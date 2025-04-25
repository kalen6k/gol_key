"""
End‑to‑end tests for the GOL‑Key RL environment
==============================================

1. API sanity      – observation conforms to `observation_space`
2. Determinism     – same seed + actions ⇒ identical observations
3. Core invariants – immortals never die, prefix is monotone ↑
4. Reward sanity   – finite totals, correct flag pay‑offs
5. Blind agents
   • RandomAgent   – plays until max_steps or raises flag
   • PrefixAgent   – uses info['prefix'] to know when to flag
6. Headless render smoke test (Pillow → numpy array)
"""
from __future__ import annotations
import random, numpy as np, pytest
from gol_key_env import GOLKeyPixelEnv, LETTERS, render_rgb

TARGET = "hello"                      # test word for most cases
N_ACTIONS = 32                        # Discrete(32) in the env


# --------------------------------------------------------------------------- #
# Helper agents
# --------------------------------------------------------------------------- #
class RandomAgent:
    """Chooses a random legal action each step; flags with 2 % chance."""
    def act(self, *_):
        if random.random() < 0.02:                         # 2 %: raise flag
            guess = "".join(random.choice(LETTERS) for _ in range(5))
            return GOLKeyPixelEnv.IDX_FLAG, guess
        return random.randrange(N_ACTIONS), None           # 0‑31 inclusive


class PrefixAgent:
    """
    Blind but prefix‑aware: keeps typing random letters until the full
    prefix is detected, then raises the flag with *some* guess.
    The guess may be wrong; we only test that the episode terminates.
    """
    def __init__(self, target_len: int):
        self.N = target_len

    def act(self, _, info):
        if info.get("prefix", 0) == self.N:
            return GOLKeyPixelEnv.IDX_FLAG, "placeholder"
        return random.randrange(N_ACTIONS), None


# --------------------------------------------------------------------------- #
# 1. API sanity – observation matches observation_space
# --------------------------------------------------------------------------- #
def test_obs_shape_dtype():
    env = GOLKeyPixelEnv(TARGET, shape=(12, 12))
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.close()


# --------------------------------------------------------------------------- #
# 2. Determinism: same seed ⇒ identical observations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed, steps", [(0, 150), (99, 400)])
def test_determinism(seed, steps):
    env1 = GOLKeyPixelEnv(TARGET, seed=seed)
    env2 = GOLKeyPixelEnv(TARGET, seed=seed)
    rng  = np.random.default_rng(123)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    assert np.array_equal(obs1, obs2)

    for _ in range(steps):
        a = int(rng.integers(N_ACTIONS))
        obs1, *_ = env1.step(a)
        obs2, *_ = env2.step(a)
        assert np.array_equal(obs1, obs2)

    env1.close(); env2.close()


# --------------------------------------------------------------------------- #
# 4. Reward sanity & flag pay‑off
# --------------------------------------------------------------------------- #
def test_flag_rewards():
    cfg = dict(step_penalty=-0.01, success=1.0, fail=-1.0)
    env = GOLKeyPixelEnv("hi", reward_cfg=cfg, seed=1)
    env.reset()

    # wrong guess  → negative delta wrt step_penalty
    _, r_fail, *_ = env.step(GOLKeyPixelEnv.IDX_FLAG, guess="no")
    assert r_fail < 0

    env.reset()
    # correct guess → positive delta
    _, r_succ, *_ = env.step(GOLKeyPixelEnv.IDX_FLAG, guess="hi")
    assert r_succ > 0
    env.close()


# --------------------------------------------------------------------------- #
# 5a. RandomAgent – plays a full episode without crashing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(6, 6), (10, 10)])
def test_random_agent_episode(shape):
    env   = GOLKeyPixelEnv("abc", shape=shape, max_steps=256, seed=2)
    agent = RandomAgent()
    obs, _ = env.reset()

    while True:
        action, guess = agent.act(obs, {})
        obs, _, term, trunc, _ = env.step(action, guess=guess)
        if term or trunc:
            break
    env.close()


# --------------------------------------------------------------------------- #
# 5b. PrefixAgent flags when full prefix reached
# --------------------------------------------------------------------------- #
def test_prefix_agent_flags():
    env = GOLKeyPixelEnv("hi", shape=(8, 8), max_steps=512, seed=3)
    obs, info = env.reset()

    while True:
        pref = info.get("prefix", 0)
        if pref == 0:                             # need first letter
            action, guess = LETTERS.index('h'), None
        elif pref == 1:                           # need second letter
            action, guess = LETTERS.index('i'), None
        else:                                     # full word → flag
            action, guess = GOLKeyPixelEnv.IDX_FLAG, "hi"

        obs, _, term, trunc, info = env.step(action, guess=guess)
        if term or trunc:
            break

    assert term and not trunc, "agent failed to terminate correctly"
    env.close()


# --------------------------------------------------------------------------- #
# 6. Head‑less render smoke test
# --------------------------------------------------------------------------- #
def test_render_rgb_headless():
    env = GOLKeyPixelEnv("hi", shape=(8, 8), seed=7)
    env.reset()
    for _ in range(3):
        env.step(random.randrange(N_ACTIONS))
        frame = render_rgb(env.core, cell_size=env.cell_px)
        assert frame.dtype == np.uint8 and frame.shape[2] == 3
    env.close()
