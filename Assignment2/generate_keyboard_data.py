"""
generate_keyboard_data.py
=========================
Play LunarLander with the keyboard and record (state, action) tuples
to a CSV file used as training data for the VQQL quantizer (Phase 2).

Controls:
    W  -> main engine
    A  -> left engine
    D  -> right engine
    (no key) -> do nothing

Press the window's close button or Ctrl+C in the terminal to stop early.
The CSV is saved incrementally after each episode, so you can stop at
any time without losing progress.

Usage:
    python generate_keyboard_data.py
"""

import csv
import os
import gymnasium as gym
from gymnasium.utils.play import play

# ==========================================
# CONFIGURATION
# ==========================================

ENV_NAME = "LunarLander-v3"
GRAVITY = -1.62
OUTPUT_CSV = "keyboard_data.csv"

# Key bindings: keyboard key -> discrete action
# Action 0: do nothing (noop)
# Action 1: left engine
# Action 2: main engine
# Action 3: right engine
KEYS_TO_ACTION = {
    "w": 2,   # main engine
    "a": 1,   # left engine
    "d": 3,   # right engine
}

# CSV columns (must match the columns used by VQQL_Discretization.py)
CSV_HEADER = [
    "x_position", "y_position",
    "x_velocity", "y_velocity",
    "angle", "angular_velocity",
    "left_leg", "right_leg",
    "action",
]


# ==========================================
# CSV WRITER (kept open during the whole session)
# ==========================================

def open_csv(path):
    """Open the CSV file in append mode, writing the header if new."""
    file_exists = os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(CSV_HEADER)
    return f, writer


# ==========================================
# CALLBACK
# ==========================================
# play() calls this callback after every environment step with:
#   obs_t   : observation BEFORE the step
#   obs_tp1 : observation AFTER the step
#   action  : action taken
#   rew     : reward
#   terminated, truncated : end-of-episode flags
#   info    : dict
# We log obs_t (state seen by the player) and the chosen action.

# Track episode count using a mutable container
_episode_counter = {"n": 0, "steps_in_episode": 0, "rows_total": 0}


def make_callback(writer, csv_file):
    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        # Log the (state, action) pair
        row = list(obs_t) + [int(action)]
        writer.writerow(row)
        _episode_counter["steps_in_episode"] += 1
        _episode_counter["rows_total"] += 1

        if terminated or truncated:
            _episode_counter["n"] += 1
            print(
                f"Episode {_episode_counter['n']:>3} finished | "
                f"steps: {_episode_counter['steps_in_episode']:>3} | "
                f"total rows: {_episode_counter['rows_total']}"
            )
            _episode_counter["steps_in_episode"] = 0
            csv_file.flush()  # flush after each episode to be safe

    return callback


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("LUNAR LANDER - Keyboard data collection")
    print("=" * 60)
    print("Controls:  W = main  |  A = left  |  D = right  |  none = noop")
    print(f"Saving data to: {OUTPUT_CSV}")
    print("Close the window when you are done.")
    print("=" * 60)

    csv_file, writer = open_csv(OUTPUT_CSV)
    callback = make_callback(writer, csv_file)

    try:
        env = gym.make(ENV_NAME, gravity=GRAVITY, render_mode="rgb_array")
        play(
            env,
            keys_to_action=KEYS_TO_ACTION,
            noop=0,
            callback=callback,
            fps=30,
        )
    finally:
        csv_file.close()
        print(f"\nSession ended. Total rows written: {_episode_counter['rows_total']}")
        print(f"Total episodes: {_episode_counter['n']}")


if __name__ == "__main__":
    main()