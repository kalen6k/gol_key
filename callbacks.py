# callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
from collections import deque
from gol_key_env import GOLKeyPixelEnv
from agent_model import GOLKeyAgent
import traceback

class LengthCurriculumCallback(BaseCallback):
    """
    Watch the rolling success rate; raise env.max_len when > 50 %.
    Assumes envs are WordTargetWrapper.
    """
    def __init__(self, target_success=0.5, window=250, verbose=1, max_len_limit=8):
        super().__init__(verbose)
        self.targ = target_success
        self.window_size  = window
        self.history = deque(maxlen=self.window_size)
        self.max_len_limit = max_len_limit

    def _on_step(self):
        # gather terminal rewards from infos
        infos = self.locals["infos"]
        for info in infos:
            if info.get("terminated", False):
                outcome = info.get("success", 0)
                self.history.append(outcome)

        if len(self.history) < self.window_size:
            return True

        rate = np.mean(list(self.history))
        try:
            current_max_len_list = self.training_env.env_method("get_attr", "_max_len")
            if current_max_len_list:
                current_max_len = current_max_len_list[0]
        except Exception as e:
            print(f"ERROR in LengthCurriculumCallback: Could not access env wrappers correctly: {e}")
            traceback.print_exc()
            return True
        
        if current_max_len != -1 and rate >= self.target_success and current_max_len < self.max_len_limit:
            new_len = current_max_len + 1
            if self.verbose > 0:
                print(f"\n[{self.num_timesteps} steps] Rolling guess success {rate*100:.1f}% >= {self.target_success*100:.1f}%")
                print(f"  --> Increasing max_len from {current_max_len} to {new_len}\n")

            self.training_env.env_method("set_max_len", new_len)

            if wandb and wandb.run:
                wandb.log({"curriculum/max_len": new_len}, step=self.num_timesteps, commit=True)
            self.logger.record("curriculum/max_len", new_len)
            self.history.clear()

            if self.verbose > 0:
                 print(f"  --> History window cleared. Gathering data for length {new_len}.")
        
        if self.num_timesteps % self.window_size == 0:
            self.logger.record("curriculum/guess_success_rate", rate)

        return True


class SuccessRateCallback(BaseCallback):
    """
    Calculates and logs the running success rate of decoder guesses (IDX_FLAG action).
    """
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.guess_outcomes = deque(maxlen=window_size)
        self.total_guesses = 0
        self.correct_guesses = 0
        try:
             from gol_key_env import GOLKeyPixelEnv
             self.guess_action_index = GOLKeyPixelEnv.IDX_FLAG
        except (ImportError, AttributeError):
             print("WARNING: SuccessRateCallback could not import GOLKeyPixelEnv or find IDX_FLAG. Using default value 27.")
             self.guess_action_index = 27

    def _on_step(self) -> bool:
        """
        Called after each step rollout. Checks infos for terminated episodes
        caused by the IDX_FLAG action and updates the success rate.
        """
        for info in self.locals["infos"]:
            if (info.get("terminated", False) and
                info.get("action_taken") == GOLKeyPixelEnv.IDX_FLAG and
                "success" in info):

                outcome = info["success"]
                self.guess_outcomes.append(outcome)

                self.total_guesses += 1
                if outcome:
                    self.correct_guesses += 1

        if self.n_calls % self.locals.get("log_interval", 1) == 0:
            window_rate = np.mean(self.guess_outcomes) if self.guess_outcomes else 0.0
            self.logger.record('custom/decoder_success_rate_window', window_rate)
            cumulative_rate = (self.correct_guesses / self.total_guesses
                               if self.total_guesses > 0 else 0.0)
            self.logger.record('custom/decoder_success_rate_cumulative', cumulative_rate)
            self.logger.record('custom/total_guess_attempts', self.total_guesses)

        return True
    
class PromptHintCallback(BaseCallback):
    """
    Removes the length hint from the GOLKeyAgent's guess prompt once
    a target success rate is achieved at the maximum curriculum length.
    """
    def __init__(self, target_success=0.75, window=1000, max_len_limit=8, verbose=1):
        super().__init__(verbose)
        self.target_success = target_success
        self.window_size = window
        self.max_len_history = deque(maxlen=self.window_size)
        self.max_len_limit = max_len_limit
        self.hint_removed = False
        self.agent_ref = None
        self.guess_action_index = GOLKeyPixelEnv.IDX_FLAG

    def _init_callback(self) -> None:
        """Get reference to the agent via the model's feature extractor."""
        try:
            # Assumes CnnPolicy -> VLMExtractor -> GOLKeyAgent structure
            if hasattr(self.model.policy, 'features_extractor') and \
               hasattr(self.model.policy.features_extractor, 'agent') and \
               isinstance(self.model.policy.features_extractor.agent, GOLKeyAgent):
                 self.agent_ref = self.model.policy.features_extractor.agent
                 print("PromptHintCallback: Successfully got reference to GOLKeyAgent.")
            else:
                 print("ERROR: PromptHintCallback could not find GOLKeyAgent via model.policy.features_extractor.agent.")
        except AttributeError as e:
             print(f"ERROR: PromptHintCallback failed to get agent reference: {e}")

    def _check_if_at_max_len(self) -> bool:
        """Helper to check if curriculum is at max length."""
        current_max_len = -1
        try:
            current_max_len_list = self.training_env.env_method("get_attr", "_max_len")
            if current_max_len_list:
                current_max_len = current_max_len_list[0]
        except Exception:
            pass
        return current_max_len == self.max_len_limit

    def _on_step(self) -> bool:
        if self.hint_removed or self.agent_ref is None:
            return True

        at_max_len = self._check_if_at_max_len()

        if at_max_len:
            infos = self.locals["infos"]
            for info in infos:
                if (info.get("terminated", False) and
                    info.get("action_taken") == self.guess_action_index):
                    outcome = info.get("success", False)
                    self.max_len_history.append(outcome)
        else:
            # If not at max length, clear the history for this callback
            # to ensure rate is calculated only from max_len data
            if len(self.max_len_history) > 0:
                self.max_len_history.clear()
            return True

        # Check success rate only if window is full AND at max length
        if at_max_len and len(self.max_len_history) >= self.window_size:
            rate = np.mean(list(self.max_len_history))

            if self.num_timesteps % self.window_size == 0:
                 self.logger.record("curriculum/guess_success_rate_at_max_len", rate)

            if rate >= self.target_success:
                if self.verbose > 0:
                    print(f"\n[{self.num_timesteps} steps] Max length ({self.max_len_limit}) guess success {rate*100:.1f}% >= {self.target_success*100:.1f}%")
                    print(f"  --> Removing length hint from GOLKeyAgent prompt.\n")

                self.agent_ref.use_length_hint = False
                self.hint_removed = True 

                # Log the event
                self.logger.record("curriculum/length_hint_removed", 1)
                if wandb and wandb.run:
                    wandb.log({"curriculum/length_hint_removed": 1}, step=self.num_timesteps, commit=True)

                self.max_len_history.clear()

        return True