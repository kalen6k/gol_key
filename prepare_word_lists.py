"""
Builds `train_words.txt` and `test_words.txt` with per‑length Zipf filters.

Criteria
--------
* only lowercase ASCII letters a‑z
* 2 ≤ len(word) ≤ 8
* Zipf frequency must satisfy ZIPF_BY_LEN[len(word)]
"""

from __future__ import annotations
from pathlib import Path
import random, textwrap
from wordfreq import iter_wordlist, zipf_frequency

from collections import Counter
def count_by_len(lst): return Counter(map(len, lst))

# ── parameters you can tweak ──────────────────────────────────────────
ZIPF_BY_LEN = {
    3: 4.0,
    4: 3.0,       # 4‑letter words moderately common
    5: 3.0,       # 5‑letter words ≥ Zipf‑2
    6: 3.0,
    7: 3.0,
    8: 3.0,
}
TRAIN_FRAC   = 0.9
SEED         = 42
MAX_SCAN     = 500_000       # how many wordfreq entries to inspect
ALPHABET     = set("abcdefghijklmnopqrstuvwxyz")
# ──────────────────────────────────────────────────────────────────────

rng = random.Random(SEED)
kept: list[str] = []

for i, w in enumerate(iter_wordlist("en")):
    if i >= MAX_SCAN:
        break
    w = w.lower()
    L = len(w)
    thr = ZIPF_BY_LEN.get(L)
    if (
        thr is not None                       # length in range 2‑8
        and set(w) <= ALPHABET               # ASCII only
        and zipf_frequency(w, "en") >= thr   # Zipf threshold for that length
    ):
        kept.append(w)

rng.shuffle(kept)
cut = int(len(kept) * TRAIN_FRAC)
train, test = kept[:cut], kept[cut:]

# ── save to disk ─────────────────────────────────────────────────────
Path("train_words.txt").write_text("\n".join(train), encoding="utf‑8")
Path("test_words.txt").write_text("\n".join(test),  encoding="utf‑8")

print(
    textwrap.dedent(
        f"""
        Scanned          : {i+1:,} wordfreq entries
        Words kept total : {len(kept):,}
           ├─ train       : {len(train):,}
           └─ test        : {len(test):,}

        Breakdown by length (Zipf ≥ thresholds):
        {', '.join(f'{L}: {sum(len(w)==L for w in kept):5d}' for L in range(2,9))}
        """
    ).strip()
)

print("train:", dict(count_by_len(train)))
print("test :", dict(count_by_len(test)))