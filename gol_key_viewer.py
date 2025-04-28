# gol_key_viewer.py
import pygame, sys, time
import numpy as np
from collections import deque
from gol_key_env import GOLKeyPixelEnv, LETTERS
import time
import random
import argparse
import torch
from word_env import load_words
from train import VLMExtractor

# ─────────────── viewer settings ──────────────── #
SHAPE        = (32, 32)           # cells
CELL_SIZE    = 16                 # pixels per cell
HILITE_COLOR = (0, 255, 0)
COLORS = [
    (254, 253, 251), (248, 188, 80), (250, 181, 112),
    (246,  72,  65), (223,  69, 64), (227, 26,  93),
    (131,  26,  56), ( 62, 131,159), ( 75,217,231)
]
# ------------------------------------------------ #
POOL = [w for w in load_words("train_words.txt")]
def next_target():
    return random.choice(POOL)
TARGET = next_target()

env = GOLKeyPixelEnv(target=next_target(), shape=SHAPE)
obs, info = env.reset()
H, W = SHAPE
SIDE_PANEL_W = 280
MAX_WINDOW_W = 1280
grid_w = W * CELL_SIZE
grid_h = H * CELL_SIZE
win_w = min(grid_w + SIDE_PANEL_W, MAX_WINDOW_W)
win_h = grid_h

pygame.init()
try:
    MARK_FONT = pygame.font.Font(
        pygame.font.match_font("dejavusans,dejavusansmono,couriernew"), 64
    )
except Exception:            # could not find a TTF with the glyphs
    MARK_FONT = None
screen = pygame.display.set_mode((win_w, win_h))
pygame.display.set_caption("GOL-Key Viewer")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", CELL_SIZE - 2, bold=True)
small  = pygame.font.Font(None, 24)

# --- SPS Tracking Init ---
sps_timer = 0.0
sps_step_count = 0
displayed_sps = 0.0
SPS_UPDATE_INTERVAL = 1.0

paused  = True
hilite  = False  # triggered by the immortality toggle
                 #(only will be available in debug/review mode)
prefix_len = 0

key_steps = 0
start_time = None
final_time = 0
final_presses = 0

result_mode = None
CONFETTI = []        # list of (x,y,vx,vy,color) particles

guess_mode = False
guess_text = ""
last_guess = ""
auto_guess_queue = []
cursor_on = False
cursor_timer = 0.0

FLAG_SIZE = 40
flag_rect = pygame.Rect(
    grid_w + SIDE_PANEL_W//2 - FLAG_SIZE//2,
    grid_h - FLAG_SIZE - 20,
    FLAG_SIZE, FLAG_SIZE
)

parser = argparse.ArgumentParser(description="determines UI mode")
parser.add_argument(
    "view_mode",
    nargs="?",
    default="debug",
    choices=["debug",
             "competitive-lg",
             "competitive-lg-prefix",
             "competitive"],
    help="UI mode (default: debug)",
)
parser.add_argument("--agent", type=str, default=None,
                    help="Path to PPO checkpoint zip, or 'random' for a random policy")
parser.add_argument("--autoplay-interval", type=float, default=0.2,
                    help="Seconds between automatic env steps while autoplay is active")
args = parser.parse_args()
print(f'view_mode is {args.view_mode}')

auto_reset_armed = False
AUTOPLAY = args.agent is not None
STEP_EVENT = pygame.USEREVENT + 1

if AUTOPLAY:
    if args.agent == "random":
        policy = None
        base_agent = None
        pygame.time.set_timer(STEP_EVENT, int(args.autoplay_interval * 1000))
        print("Viewer: Using random policy with timer interval.")
        RANDOM_AGENT_MODE = True
    else:
        from stable_baselines3 import PPO
        from agent_model import GOLKeyAgent
        base_agent = GOLKeyAgent()
        custom_objects = {
                "policy_kwargs": dict(
                    features_extractor_class=VLMExtractor,
                    features_extractor_kwargs=dict(
                        agent=base_agent,
                        vlm_internal_batch_size=1,
                    ),
                    net_arch=[dict(pi=[1024, 256, 128, 32], vf=[1024, 256, 128, 32])],
                    ortho_init=False
                ),
            }
        print(f"Viewer: Loading PPO policy '{args.agent}' onto device: {base_agent.device}...")
        policy = PPO.load(args.agent, device=base_agent.device, custom_objects=custom_objects, print_system_info=False)
        print("Viewer: PPO policy loaded.")
        pygame.time.set_timer(STEP_EVENT, 0)
        RANDOM_AGENT_MODE = False
    paused = False

else:
    RANDOM_AGENT_MODE = False
    policy = None
    base_agent = None
    pygame.time.set_timer(STEP_EVENT, 0)

TYPE_EVENT   = pygame.USEREVENT + 8      # one char from queue
AUTO_EVENT   = pygame.USEREVENT + 9
DELAY_EVENT  = pygame.USEREVENT + 10
SUBMIT_EVENT = pygame.USEREVENT + 11
SUBMIT_MS    = 2000 
TYPE_MS      = 200
PAUSE_MS     = 500

def begin_timer_if_needed():
    global start_time
    if start_time is None:
        start_time = time.monotonic()

# ─────────────── drawing helpers ──────────────── #
def launch_confetti(n=150):
    CONFETTI.clear()
    for _ in range(n):
        x  = random.randint(0, win_w-4)
        y  = -10
        vx = random.uniform(-60, 60)
        vy = random.uniform( 60,140)
        col = random.choice(COLORS)
        CONFETTI.append([x,y,vx,vy,col])

def update_confetti(dt):
    for p in CONFETTI:
        p[0] += p[2] * dt
        p[1] += p[3] * dt
    # drop off‑screen particles
    CONFETTI[:] = [p for p in CONFETTI if p[1] < win_h+10]

def draw_confetti():
    for x,y,_,_,col in CONFETTI:
        pygame.draw.rect(screen, col, (int(x), int(y), 4, 4))

def draw_grid():
    c = env.core
    screen.fill((30, 30, 30))
    for y in range(H):
        for x in range(W):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE,
                                CELL_SIZE - 1, CELL_SIZE - 1)
            if c.alive[y, x]:
                col = COLORS[min(c.age[y, x], len(COLORS) - 1)]
                pygame.draw.rect(screen, col, rect)
                ch = c.chars[y, x]
                if ch:
                    txt_col = (0,0,0) if sum(col) > 400 else (255,255,255)
                    surf = font.render(ch, True, txt_col)
                    screen.blit(surf, surf.get_rect(center=rect.center))
                if hilite and c.immortal[y, x]:
                    pygame.draw.rect(screen, HILITE_COLOR, rect, 2)
            else:
                pygame.draw.rect(screen, (50,50,50), rect, 1)

def draw_ui():
    global info
    x0 = grid_w + 10
    y  = 10
    if final_time != 0:
        elapsed = final_time
        mm, ss = divmod(elapsed, 60)
    else:
        elapsed = 0 if start_time is None else int(time.monotonic() - start_time)
        mm, ss = divmod(elapsed, 60)
    
    sps_line = f"Agent SPS: {displayed_sps:.1f}" if AUTOPLAY else "Agent SPS: N/A"

    # actively mask/reveal TARGET, prefix progress, ...
    if args.view_mode   == 'debug':
        target_revealed  = env.core.target
        prefix_i_given   = info["prefix"]
        prefix_len_given = len(target_revealed)
        lines = [
            f"Target: '{target_revealed}'",
            f"Generation: {info['generation']}",
            f"Prefix: {prefix_i_given}/{prefix_len_given}",
            f"Time: {mm:02d}:{ss:02d}",
            f"Key-presses: {info['steps']}",
            sps_line,
            f"Status: {'Running' if not paused else 'Paused'}",
            "",
            "Controls:",
            "Tab          highlight immortal",
            "shift + r / c   ->   random / clear",
            "Backspace    rewind step",
            "Return/Enter return step",
            "Type letters or space",
            "",
            "Cell-age colors:"
        ]
    elif args.view_mode == 'competitive-lg':
        target_revealed  = '?' * len(env.core.target)
        prefix_i_given   = '?'
        prefix_len_given = len(env.core.target)
    elif args.view_mode == 'competitive-lg-prefix':
        target_revealed  = '?' * len(env.core.target)
        prefix_i_given   = info["prefix"]
        prefix_len_given = len(env.core.target)
    elif args.view_mode == 'competitive':
        target_revealed  = '?'
        prefix_i_given   = '?'
        prefix_len_given = '?'
    lines = [
        f"Target: '{target_revealed}'",
        f"Generation: {info['generation']}",
        f"Prefix: {prefix_i_given}/{prefix_len_given}",
        f"Time: {mm:02d}:{ss:02d}",
        f"Key-presses: {info['steps']}",
        sps_line,
        f"Status: {'Running' if not paused else 'Paused'}",
        "",
        "Controls:",
        "shift + r / c   ->   random / clear",
        "Backspace    rewind step",
        "Return/Enter return step",
        "Type letters or space",
        "",
        "Cell-age colors:"
    ]
    for line in lines:
        surf = small.render(line, True, (200,200,200))
        screen.blit(surf, (x0, y))
        y += 18
    
    box = pygame.Rect(0, 0, 16, 16)
    for idx, col in enumerate(COLORS):
        box.topleft = (x0, y)
        pygame.draw.rect(screen, col, box)
        txt = small.render(f"{idx}", True, (220, 220, 220))
        screen.blit(txt, (x0 + 28, y + 2))
        y += 20
    y+= 10
    surf = small.render("Submit flag ->", True, (220, 220, 220))
    screen.blit(surf, (x0, y))
    draw_flag()

def draw_flag():
    # little white square with red triangle = "submit"
    pygame.draw.rect(screen, (230,230,230), flag_rect, 0, 5)
    pts = [
        (flag_rect.centerx - 6, flag_rect.centery + 10),
        (flag_rect.centerx + 12, flag_rect.centery),
        (flag_rect.centerx - 6, flag_rect.centery - 10)
    ]
    pygame.draw.polygon(screen, (200,50,50), pts)

def draw_guess_overlay(dt):
    global cursor_on, cursor_timer
    # semi‑transparent dark film
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))            # last byte = alpha
    screen.blit(overlay, (0, 0))

    # blinking cursor timer
    cursor_timer += dt
    if cursor_timer >= 0.5:
        cursor_on = not cursor_on
        cursor_timer = 0.0

    # textbox
    box_w, box_h = 300, 60
    box = pygame.Rect((win_w - box_w)//2, (win_h - box_h)//2,
                      box_w, box_h)
    pygame.draw.rect(screen, (255,255,255), box, 0, 6)
    pygame.draw.rect(screen, (30,30,30), box.inflate(-4,-4), 0, 6)

    # text
    txt_surf = font.render(guess_text, True, (255,255,255))
    screen.blit(txt_surf, (box.x+10, box.y+18))

    # cursor
    if cursor_on:
        cur_x = box.x + 10 + txt_surf.get_width() + 2
        cur_y = box.y + 18
        pygame.draw.line(screen, (255,255,255),
                         (cur_x, cur_y), (cur_x, cur_y+font.get_height()-2), 2)

def do_env_step(act, guess=None):
    global obs, info, result_mode, paused, last_guess, final_time, final_presses, start_time
    obs, rew, term, trunc, info = env.step(act, guess=guess)
    begin_timer_if_needed()
    episode_ended = term or trunc
    if episode_ended:
        result_mode = "success" if rew > 0 else "fail"
        env._eps  = getattr(env, "_eps", 0)  + (term or trunc)
        env._wins = getattr(env, "_wins", 0) + (rew > 0 and (term or trunc))
        info["episodes"]  = env._eps
        info["win_rate"]  = env._wins / env._eps
        last_guess  = guess or ""
        final_presses = info["steps"]
        final_time    = 0 if start_time is None else int(time.monotonic() - start_time)
        paused = True
        if result_mode == "success":
            launch_confetti()
    return episode_ended

def autoplay_step():
    """Return (action_idx, guess_str|None) chosen by policy/random.
       May also schedule fake key events via auto_guess_queue."""
    global auto_guess_queue, guess_mode, paused, last_guess

    if policy is None:
        act = env.action_space.sample()
    else:
        tens = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy.policy.forward(tens)
            act = torch.argmax(logits, 1).item()

    guess = None
    if act == env.IDX_FLAG:
        # generate the full word only once
        if policy is None:
            full_guess = random.choice(["cat", "tree", "hello", "yeah"])
        else:
            full_guess = base_agent._decode(base_agent._embed(obs)).lower()
        guess = full_guess
        # schedule typing: one char per frame + final “Enter”
        auto_guess_queue = list(full_guess)
        last_guess = guess
        guess_mode = True
        paused = True
        pygame.time.set_timer(DELAY_EVENT, PAUSE_MS, loops=1)
        return None, None
    return act, guess

def stop_typing_timers():
    pygame.time.set_timer(TYPE_EVENT,   0)
    pygame.time.set_timer(DELAY_EVENT,  0)
    pygame.time.set_timer(SUBMIT_EVENT, 0)
# ─────────────────── main loop ─────────────────── #
running = True
last_tick = time.time()

while running:
    dt = clock.tick(60) / 1000.0 
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        
        if e.type == SUBMIT_EVENT and guess_mode:
            do_env_step(env.IDX_FLAG, guess_text.strip().lower())
            guess_mode  = False
            guess_text  = ""
            continue
        if e.type == AUTO_EVENT and result_mode is not None:
            pygame.time.set_timer(AUTO_EVENT, 0)
            auto_reset_armed = False

            obs, info = env.reset(new_target=next_target())
            start_time = None
            final_time = 0
            auto_guess_queue.clear()
            stop_typing_timers()

            result_mode = None
            paused = False
            continue
        if e.type == DELAY_EVENT and auto_guess_queue:
            pygame.time.set_timer(TYPE_EVENT, TYPE_MS, loops=1)
            continue
        if e.type == TYPE_EVENT  and auto_guess_queue:
            nxt = auto_guess_queue.pop(0)
            guess_text += nxt
            pygame.time.set_timer(TYPE_EVENT, TYPE_MS, loops=1)
            if not auto_guess_queue:
                pygame.time.set_timer(SUBMIT_EVENT, SUBMIT_MS, loops=1)
            continue
        if RANDOM_AGENT_MODE and e.type == STEP_EVENT and result_mode is None and not paused:
            act, guess = autoplay_step()
            if act is not None:
                step_successful = do_env_step(act, guess)
                if not step_successful:
                    sps_step_count += 1
            continue
        if result_mode is not None:
            if e.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                # ----- RESET everything for a new round -----
                result_mode  = None
                obs, info = env.reset(new_target=next_target())
                start_time  = None
                final_time  = 0
                auto_guess_queue.clear()
                stop_typing_timers()
                paused      = True    # wait until player un‑pauses
            continue
        elif guess_mode and e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                guess_mode  = False
                guess_text  = ""
            elif e.key == pygame.K_RETURN:
                # submit final guess via env action 27
                do_env_step(env.IDX_FLAG, guess_text.strip().lower())
                guess_text  = ""
            elif e.key == pygame.K_BACKSPACE:
                guess_text = guess_text[:-1]
            elif e.unicode.isprintable():
                guess_text += e.unicode
        elif not guess_mode:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if flag_rect.collidepoint(e.pos):
                    guess_mode   = True
                    paused       = True
                    cursor_timer = 0.0
                    cursor_on    = True
                    guess_text   = ""
            elif e.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if e.key == pygame.K_TAB and args.view_mode == "debug":
                    hilite = not hilite
                elif e.key == pygame.K_r and (mods & pygame.KMOD_SHIFT):
                    do_env_step(env.IDX_RANDOM)
                elif e.key == pygame.K_c and (mods & pygame.KMOD_SHIFT):
                    do_env_step(env.IDX_CLEAR)
                elif e.key == pygame.K_BACKSPACE:
                    do_env_step(env.IDX_BACK)
                elif e.key == pygame.K_RETURN:
                    do_env_step(env.IDX_FWD)
                elif pygame.K_a <= e.key <= pygame.K_z or e.key == pygame.K_SPACE:
                    if e.key == pygame.K_SPACE:
                        action_idx = env.IDX_SPACE       # 26
                    else:
                        action_idx = e.key - pygame.K_a  # K_a..K_z  ->  0..25
                    do_env_step(action_idx)

    if not paused:
        last_tick = time.time()
    
    if AUTOPLAY and not RANDOM_AGENT_MODE and not paused and result_mode is None:
        step_start_time = time.monotonic()
        act, guess = autoplay_step()
        if act is not None:
            step_successful = do_env_step(act, guess)
            step_end_time = time.monotonic()
            if not step_successful:
                print(f"Agent step duration: {step_end_time - step_start_time:.3f}s")
                sps_step_count += 1

    sps_timer += dt
    if sps_timer >= SPS_UPDATE_INTERVAL:
        if sps_timer > 0:
            displayed_sps = sps_step_count / sps_timer
        else:
            displayed_sps = 0.0
        sps_timer = 0.0
        sps_step_count = 0

    draw_grid()
    draw_ui()
    if guess_mode:
        draw_guess_overlay(dt)
    
    if result_mode is not None:
        dim = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        screen.blit(dim, (0, 0))
        if result_mode == "success":
            update_confetti(dt)
            draw_confetti()
        else:
            fail_overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            fail_overlay.fill((200, 0, 0, 100))
            screen.blit(fail_overlay, (0, 0))

        if last_guess:
            gsurf = font.render(f"guess was: {last_guess}", True, (240,240,240))
            screen.blit(gsurf, gsurf.get_rect(center=(win_w//2, win_h//2 - 60)))
        mark_text = "CORRECT!" if result_mode == "success" else "WRONG!"
        if result_mode != "success":
            gsurf = font.render(f"target was: {env.core.target}", True, (240,240,240))
            screen.blit(gsurf, gsurf.get_rect(center=(win_w//2, win_h//2 - 90)))
        mark_col  = (120,255,120) if result_mode == "success" else (255,120,120)
        mark_surf = font.render(mark_text, True, mark_col)
        screen.blit(mark_surf, mark_surf.get_rect(center=(win_w//2, win_h//2 - 30)))

        guess_mode = False

        mm, ss = divmod(final_time, 60)
        stat_line = f"{mm:02d}:{ss:02d}   {final_presses} keypresses"
        stat_surf = small.render(stat_line, True, (230,230,230))
        screen.blit(stat_surf, stat_surf.get_rect(center=(win_w//2, win_h//2 + 20)))
        # --------------------------------------
        prompt = small.render("Press any key to continue", True, (240, 240, 240))
        screen.blit(prompt, prompt.get_rect(center=(win_w//2, win_h//2 + 50)))
        if AUTOPLAY and not auto_reset_armed:
            # fire once, 3 s later, to press SPACE automatically
            pygame.time.set_timer(AUTO_EVENT, 3000, loops=1)
            auto_reset_armed = True
    pygame.display.flip()
    clock.tick(60)

env.close()
pygame.quit()
sys.exit()
