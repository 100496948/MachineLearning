"""
generate_agent_data.py
======================
Run the trained Q-learning agent (Phase 1) for many episodes and record
(state, action) tuples to a CSV used as training data for the VQQL
quantizer (Phase 2).

Uses an epsilon-greedy policy with a small epsilon to introduce some
variability and broaden the distribution of visited states.

Usage:
    python generate_agent_data.py
"""

import os
import csv
import numpy as np
import gymnasium as gym

# ==========================================
# CONFIGURATION
# ==========================================

ENV_NAME = "LunarLander-v3"
GRAVITY = -1.62

# Path to the best Q-table from Phase 1 (rename your favorite file to this).
QTABLE_FILE = "qtable_phase1_best.txt"

# Output CSV
OUTPUT_CSV = "agent_data.csv"

# Number of episodes to record
N_EPISODES = 200

# Small exploration probability so the agent does not always follow
# the exact same trajectory and we get a richer state distribution.
EPSILON = 0.10

# Render the environment while recording (slower but you can watch it).
RENDER = False

# CSV columns (must match VQQL_Discretization.py)
CSV_HEADER = [
    "x_position", "y_position",
    "x_velocity", "y_velocity",
    "angle", "angular_velocity",
    "left_leg", "right_leg",
    "action",
]

# ==========================================
# MANUAL DISCRETIZATION (must match LunarLander_RL.py)
# ==========================================

BINS_X_POS    = [-0.5, -0.1, 0.1, 0.5]
BINS_Y_POS    = [0.1, 0.4, 0.8, 1.2]
BINS_X_VEL    = [-0.2, 0.2]
BINS_Y_VEL    = [-0.4, -0.1]
BINS_ANGLE    = [-0.2, 0.2]
BINS_ANG_VEL  = [-0.3, 0.3]

N_X_POS   = len(BINS_X_POS)   + 1
N_Y_POS   = len(BINS_Y_POS)   + 1
N_X_VEL   = len(BINS_X_VEL)   + 1
N_Y_VEL   = len(BINS_Y_VEL)   + 1
N_ANGLE   = len(BINS_ANGLE)   + 1
N_ANG_VEL = len(BINS_ANG_VEL) + 1
N_LEFT    = 2
N_RIGHT   = 2


def manual_state_id(obs):
    i_x  = int(np.digitize(obs[0], BINS_X_POS))
    i_y  = int(np.digitize(obs[1], BINS_Y_POS))
    i_vx = int(np.digitize(obs[2], BINS_X_VEL))
    i_vy = int(np.digitize(obs[3], BINS_Y_VEL))
    i_a  = int(np.digitize(obs[4], BINS_ANGLE))
    i_va = int(np.digitize(obs[5], BINS_ANG_VEL))
    i_l  = int(obs[6])
    i_r  = int(obs[7])

    sid  = i_x
    sid += i_y  * N_X_POS
    sid += i_vx * N_X_POS * N_Y_POS
    sid += i_vy * N_X_POS * N_Y_POS * N_X_VEL
    sid += i_a  * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL
    sid += i_va * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL * N_ANGLE
    sid += i_l  * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL * N_ANGLE * N_ANG_VEL
    sid += i_r  * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL * N_ANGLE * N_ANG_VEL * N_LEFT
    return int(sid)


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("LUNAR LANDER - Agent data collection")
    print("=" * 60)

    if not os.path.exists(QTABLE_FILE):
        raise FileNotFoundError(
            f"Q-table file '{QTABLE_FILE}' not found. "
            "Please rename your best Phase 1 Q-table to this name "
            "(or change QTABLE_FILE at the top of this script)."
        )

    qtable = np.loadtxt(QTABLE_FILE)
    print(f"Loaded Q-table: {qtable.shape}")

    render_mode = "human" if RENDER else None
    env = gym.make(ENV_NAME, gravity=GRAVITY, render_mode=render_mode)

    rows_total = 0
    rewards = []

    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        for ep in range(N_EPISODES):
            obs, _info = env.reset()
            terminated = truncated = False
            ep_reward = 0.0
            ep_steps = 0

            while not (terminated or truncated):
                # Epsilon-greedy
                if np.random.uniform() < EPSILON:
                    action = env.action_space.sample()
                else:
                    sid = manual_state_id(obs)
                    action = int(np.argmax(qtable[sid, :]))

                # Log (state seen, action chosen) BEFORE stepping
                writer.writerow(list(obs) + [int(action)])
                rows_total += 1
                ep_steps += 1

                obs, reward, terminated, truncated, _info = env.step(action)
                ep_reward += reward

            rewards.append(ep_reward)
            f.flush()

            print(
                f"Episode {ep + 1:>3}/{N_EPISODES} | "
                f"steps: {ep_steps:>3} | "
                f"reward: {ep_reward:>8.2f} | "
                f"total rows: {rows_total}"
            )

    env.close()

    print()
    print("=" * 60)
    print(f"Saved {rows_total} rows to {OUTPUT_CSV}")
    print(f"Average episode reward: {np.mean(rewards):.2f} "
          f"(std {np.std(rewards):.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()