# word_env.py
import numpy as np
import random
import gymnasium as gym
from gol_key_env import GOLKeyPixelEnv

def load_words(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [w.strip() for w in f if w.strip()]
    except FileNotFoundError:
        print(f"ERROR: Word file not found at {path}")
        return []

class WordTargetWrapper(gym.Wrapper):
    def __init__(self, wordfile, *, agent_instance=None, min_len=3, max_len=3, shape=(28, 28), **env_kw):
        env = GOLKeyPixelEnv(shape=shape, agent_instance=agent_instance, **env_kw)
        super().__init__(env)

        self.initial_wordfile = wordfile
        self.all_words_from_file = load_words(wordfile)
        if not self.all_words_from_file:
             raise ValueError(f"No words loaded from {wordfile}.")
        
        self.all_words = list(self.all_words_from_file)

        self.min_len = min_len
        self._max_len = max_len

        self.is_eval_mode = False
        self._eval_next_target_word = None
        self._eval_word_list_override_active = False

        self._update_word_pool()


    @property
    def max_len(self) -> int:
        return self._max_len

    def _update_word_pool(self):
        if self.is_eval_mode and self._eval_word_list_override_active:
            self.word_pool = list(self.all_words) 
        else:
            self.word_pool = [
                w for w in self.all_words if self.min_len <= len(w) <= self._max_len
            ]
        if not self.word_pool and not (self.is_eval_mode and self._eval_next_target_word):
            print(f"WARN: Word pool empty. MinLen: {self.min_len}, MaxLen: {self._max_len}, EvalMode: {self.is_eval_mode}")

    def set_max_len(self, new_len: int):
        if self.is_eval_mode: return
        if new_len > self._max_len:
            self._max_len = new_len
            self._update_word_pool()

    def set_eval_mode(self, eval_mode_on: bool, word_list_for_eval: list = None):
        self.is_eval_mode = eval_mode_on
        self._eval_next_target_word = None 
        if self.is_eval_mode:
            if word_list_for_eval is not None:
                self.all_words = list(word_list_for_eval)
                self._eval_word_list_override_active = True
            else:
                self.all_words = list(self.all_words_from_file)
                self._eval_word_list_override_active = False
        else:
            self.all_words = list(self.all_words_from_file)
            self._eval_word_list_override_active = False
        self._update_word_pool()

    def set_next_eval_target(self, word: str | None):
        if not self.is_eval_mode: return
        self._eval_next_target_word = word

    def reset(self, *, seed=None, options=None):
        target_word = None
        
        if options and 'new_target' in options:
            target_word = options['new_target']
        elif self.is_eval_mode and self._eval_next_target_word:
            target_word = self._eval_next_target_word
            self._eval_next_target_word = None
        elif self.word_pool:
            target_word = random.choice(self.word_pool)
        else:
            target_word = "error"

        core_env_options = options.copy() if options else {}
        core_env_options['new_target'] = target_word 
        
        observation, info = self.env.reset(seed=seed, options=core_env_options)

        info['target_word'] = target_word
        info['max_len'] = self._max_len
        if self.is_eval_mode:
            info['eval_mode'] = True

        return observation, info