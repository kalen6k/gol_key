import matplotlib.pyplot as plt
from gol_key_env import GOLKeyPixelEnv

env = GOLKeyPixelEnv(target="target", shape=(48, 48))
frame, info = env.reset()

def show(img, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)

show(frame, "reset()")
chars = ['t', 't', 'a', 'a', 'r', 'r', 'g', 'g', 'e', 'e', 't', 't', ' ', ' ', ' ', ' ']

for i, ch in enumerate(chars, 1):
    if ch == " ":
        action = env.IDX_SPACE
    else:
        action = ord(ch.lower()) - ord("a")        # 0–25
    frame, *_ = env.step(action)
    show(frame, f"typed '{ch}'  (step {i})")

plt.show()
