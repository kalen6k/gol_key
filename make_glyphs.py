# make_glyphs.py ----------------------------------------------------
"""
Rasterise 1‑bit glyphs for 26 letters + space with Pillow’s built‑in
bitmap font and save them to glyphs.npy (dict {char: bool array}).
Run once; fast_render_rgb() will mmap the file at runtime.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT         = Path(__file__).with_name("glyphs.npy")
CHARS       = "abcdefghijklmnopqrstuvwxyz "
GLYPH_SIZE  = 12

FONT = ImageFont.load_default()
print("✓ using Pillow built‑in bitmap font")

glyphs = {}
for ch in CHARS:
    img  = Image.new("L", (GLYPH_SIZE, GLYPH_SIZE), 0)
    draw = ImageDraw.Draw(img)
    w, h = draw.textbbox((0, 0), ch, font=FONT)[2:]
    draw.text(((GLYPH_SIZE - w) // 2, (GLYPH_SIZE - h) // 2),
              ch, fill=255, font=FONT)
    glyphs[ch] = np.asarray(img, dtype=np.uint8) > 127

np.save(OUT, glyphs, allow_pickle=True)
print(f"saved {len(glyphs)} glyphs to {OUT}")