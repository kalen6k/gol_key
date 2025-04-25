# word_env.py
import numpy as np
import random
import gymnasium as gym
from gol_key_env import GOLKeyPixelEnv

def load_words(path):
    """Loads words from a file, one word per line."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [w.strip() for w in f if w.strip()]
    except FileNotFoundError:
        print(f"ERROR: Word file not found at {path}")
        return []

class WordTargetWrapper(gym.Wrapper):
    """
    Gym wrapper that selects a random target word at each reset
    within a specified length range [min_len, max_len].
    Exposes `set_max_len()` for curriculum learning.
    """
    def __init__(self, wordfile, *, agent_instance=None, min_len=3, max_len=3, shape=(28, 28), **env_kw):
        env = GOLKeyPixelEnv(shape=shape, agent_instance=agent_instance, **env_kw)
        super().__init__(env)

        self.all_words = load_words(wordfile)
        if not self.all_words:
             raise ValueError(f"No words loaded from {wordfile}. Check the file path and content.")

        self.min_len = min_len
        self._max_len = max_len
        self.wordfile = wordfile
        self._update_word_pool()

    @property
    def max_len(self) -> int:
        """Current maximum word length allowed by the curriculum."""
        return self._max_len

    def _update_word_pool(self):
        """Filters `all_words` based on current min/max length."""
        self.word_pool = [
            w for w in self.all_words if self.min_len <= len(w) <= self._max_len
        ]
        if not self.word_pool:
             print(f"WARN: No words found in {self.wordfile} with length between {self.min_len} and {self._max_len}")

    def set_max_len(self, new_len: int):
        """Updates the maximum word length (called by curriculum callback)."""
        if new_len > self._max_len:
            self._max_len = new_len
            self._update_word_pool()

    def reset(self, *, seed=None, options=None):
        """Resets the environment with a new target word from the current pool."""
        super().reset(seed=seed)

        if not self.word_pool:
            print("ERROR: Word pool is empty during reset. Using fallback word 'error'.")
            target_word = "error"
        else:
            target_word = random.choice(self.word_pool)

        reset_options = options or {}
        reset_options['new_target'] = target_word

        observation, info = self.env.reset(seed=seed, options=reset_options)

        info['target_word'] = target_word
        info['max_len'] = self._max_len

        return observation, info