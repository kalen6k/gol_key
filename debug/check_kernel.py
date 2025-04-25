"""
check_kernel.py –– verifies life_kernel == _slow_step
Run with:  python check_kernel.py
"""

import numpy as np
from copy import deepcopy
from gol_key_env import GOLKeyCore, LETTERS, life_kernel, NUMBA_OK

assert NUMBA_OK, "Numba kernel was not compiled – nothing to test!"

def take_snapshot(core):
    return {
        'alive': core.alive.copy(),
        'immortal': core.immortal.copy(),
        'chars': core.chars.copy(),
        'prefix': core.prefix_len.copy(),
        'age': core.age.copy(),
    }

def compare_states(a, b, step_idx):
    for k in a:
        try:
            np.testing.assert_array_equal(a[k], b[k])
        except AssertionError as e:
            raise AssertionError(f"Mismatch in '{k}' after step {step_idx}:\n{e}")

RNG = np.random.default_rng(0)
TARGET = "hello"
STEPS = 500        # how many random actions to try
SHAPE = (32, 32)

# ------------------------------------------------------------
# 1. kernel run
core_fast = GOLKeyCore(TARGET, shape=SHAPE, seed=123)
snapshots_fast = [take_snapshot(core_fast)]
typed_seq = []

for _ in range(STEPS):
    key = RNG.choice(list(LETTERS) + [None])   # None ⇒ no‑key / SPACE
    typed_seq.append(key)
    core_fast.step(key)
    snapshots_fast.append(take_snapshot(core_fast))

# ------------------------------------------------------------
# 2. slow run (force _slow_step)
core_slow = GOLKeyCore(TARGET, shape=SHAPE, seed=123)  # same RNG seed
snapshots_slow = [take_snapshot(core_slow)]

for key in typed_seq:
    core_slow._slow_step(key)
    snapshots_slow.append(take_snapshot(core_slow))

# ------------------------------------------------------------
# 3. compare
for i, (a, b) in enumerate(zip(snapshots_fast, snapshots_slow)):
    compare_states(a, b, i)

print(f"✅  Kernel matches slow reference for {STEPS} random steps.")
