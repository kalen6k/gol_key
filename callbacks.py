# callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
from collections import deque
from gol_key_env import GOLKeyPixelEnv
import traceback

class LengthCurriculumCallback(BaseCallback):
    """
    Watch the rolling success rate; raise env.max_len when > 50 %.
    Assumes envs are WordTargetWrapper.
    """
    def __init__(self, target_success=0.5, window=250, verbose=1, max_len_limit=8):
        super().__init__(verbose)
        self.targ = target_success
        self.win  = window
        self.history = deque(maxlen=self.window_size)
        self.max_len_limit = max_len_limit

    def _on_step(self):
        # gather terminal rewards from infos
        infos = self.locals["infos"]
        for info in infos:
            if info.get("terminated", False):
                reward = info.get("reward", 0)
                self.history.append(reward > 0.5)

        if len(self.hist) < self.win:
            return True

        rate = np.mean(self.hist[-self.win:])
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
                print(f"\n[{self.num_timesteps} steps] Rolling success {rate*100:.1f}% >= {self.target_success*100:.1f}%")
                print(f"  --> Increasing max_len from {current_max_len} to {new_len}\n")

            self.training_env.env_method("set_max_len", new_len)

            if wandb and wandb.run:
                wandb.log({"curriculum/max_len": new_len}, step=self.num_timesteps, commit=True)
            self.logger.record("curriculum/max_len", new_len)
            self.history.clear()

            if self.verbose > 0:
                 print(f"  --> History window cleared. Gathering data for length {new_len}.")
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