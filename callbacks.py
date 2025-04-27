# callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np, wandb
from collections import deque
from gol_key_env import GOLKeyPixelEnv
import traceback

class LengthCurriculumCallback(BaseCallback):
    """
    Watch the rolling success rate; raise env.max_len when > 50 %.
    Assumes envs are WordTargetWrapper.
    """
    def __init__(self, target_success=0.5, window=250, verbose=1):
        super().__init__(verbose)
        self.targ = target_success
        self.win  = window
        self.hist = []

    def _on_step(self):
        # gather terminal rewards from infos
        infos = self.locals["infos"]
        for info in infos:
            if "terminated" in info and info["terminated"]:
                self.hist.append(info["reward"] > 0)
        if len(self.hist) < self.win:
            return True

        rate = np.mean(self.hist[-self.win:])
        try:
            word_target_wrapper_env = self.training_env.unwrapped.envs[0].env
            cur_len = word_target_wrapper_env.max_len
        except:
            print(f"ERROR in LengthCurriculumCallback: Could not access env wrappers correctly: {e}")
            traceback.print_exc()
        if rate >= self.targ and cur_len < 8:
            new_len = cur_len + 1
            if self.verbose:
                print(f"\n  Rolling success {rate*100:.1f}%  →  increase max_len to {new_len}\n")
            # propagate to every sub‑env
            self.training_env.env_method("set_max_len", new_len)
            wandb.log({"curriculum/max_len": new_len}, step=self.num_timesteps)
        return True


class SuccessRateCallback(BaseCallback):
    """
    Calculates and logs the running success rate of decoder guesses (IDX_FLAG action).
    """
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        # Use deque to store recent guess outcomes
        self.guess_outcomes = deque(maxlen=window_size)
        self.total_guesses = 0
        self.correct_guesses = 0

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

                # Update cumulative counts
                self.total_guesses += 1
                if outcome:
                    self.correct_guesses += 1

        # Log metrics at the SB3 log interval
        if self.logger:
            # Calculate success rate over the sliding window
            window_rate = np.mean(self.guess_outcomes) if self.guess_outcomes else 0.0
            self.logger.record('custom/decoder_success_rate_window', window_rate)

            # Calculate cumulative success rate
            cumulative_rate = (self.correct_guesses / self.total_guesses
                               if self.total_guesses > 0 else 0.0)
            self.logger.record('custom/decoder_success_rate_cumulative', cumulative_rate)
            self.logger.record('custom/total_guess_attempts', self.total_guesses)

        return True