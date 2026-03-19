"""
Lunar Lander
Made with Gymnasium
January 2025 - Machine Learning Classes
University Carlos III of Madrid

This template uses the Gymnasium LunarLander-v3 environment.
Students will implement a rule-based agent to land the spacecraft.
"""

import gymnasium as gym
import sys
import time
import math
import random
import pygame

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# GRAVITY setting
GRAVITY = -1.62

# Agent mode: True = rule-based agent, False = keyboard control
USE_AGENT = True

# Environment configuration
ENV_NAME = "LunarLander-v3"

# Action definitions
ACTION_NOTHING = 0
ACTION_LEFT_ENGINE = 1
ACTION_MAIN_ENGINE = 2
ACTION_RIGHT_ENGINE = 3

# ──────────────────────────────────────────────
# ATTRIBUTE NAMES (raw + engineered)
# ──────────────────────────────────────────────

RAW_ATTRIBUTES = [
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
]

ENGINEERED_ATTRIBUTES = [
    "euclidean_distance",       # sqrt(x^2 + y^2)
    "total_velocity",           # sqrt(vx^2 + vy^2)
    "fast_descent",             # 1 if vy < -0.5 else 0
    "polar_angle_to_pad",       # atan2(y, x)
    "abs_y_velocity",           # |vy|
    "angle_x_angular_vel",      # angle * angular_velocity
    "x_pos_x_x_vel",           # x_position * x_velocity
    "y_pos_x_y_vel",           # y_position * y_velocity
    "low_alt_high_speed",       # 1 if y < 0.3 and total_velocity > 0.5 else 0
]

# Class attribute for classification (last before reward)
CLASS_ATTRIBUTE = "action"

# Regression target
REWARD_ATTRIBUTE = "next_reward"

ALL_ATTRIBUTES = RAW_ATTRIBUTES + ENGINEERED_ATTRIBUTES + [CLASS_ATTRIBUTE, REWARD_ATTRIBUTE]


# ──────────────────────────────────────────────
# GAME STATE CLASS
# ──────────────────────────────────────────────

class GameState:
    def __init__(self, observation):
        self.x_position = observation[0]
        self.y_position = observation[1]
        self.x_velocity = observation[2]
        self.y_velocity = observation[3]
        self.angle = observation[4]
        self.angular_velocity = observation[5]
        self.left_leg_contact = observation[6]
        self.right_leg_contact = observation[7]
        self.observation = observation
        self.score = 0.0
        self.episode_reward = 0.0
        self.action = ACTION_NOTHING

    def update(self, observation, reward):
        self.x_position = observation[0]
        self.y_position = observation[1]
        self.x_velocity = observation[2]
        self.y_velocity = observation[3]
        self.angle = observation[4]
        self.angular_velocity = observation[5]
        self.left_leg_contact = observation[6]
        self.right_leg_contact = observation[7]
        self.observation = observation
        self.episode_reward += reward
        self.score = self.episode_reward

    def reset(self, observation):
        self.__init__(observation)


# ──────────────────────────────────────────────
# ENGINEERED FEATURES
# ──────────────────────────────────────────────

def compute_engineered_features(game):
    """
    Compute the 9 engineered features from the current game state.
    Returns a list of floats in the same order as ENGINEERED_ATTRIBUTES.
    """
    x = game.x_position
    y = game.y_position
    vx = game.x_velocity
    vy = game.y_velocity
    ang = game.angle
    ang_vel = game.angular_velocity

    total_vel = math.sqrt(vx ** 2 + vy ** 2)

    return [
        math.sqrt(x ** 2 + y ** 2),          # euclidean_distance
        total_vel,                             # total_velocity
        1.0 if vy < -0.5 else 0.0,            # fast_descent
        math.atan2(y, x),                      # polar_angle_to_pad
        abs(vy),                               # abs_y_velocity
        ang * ang_vel,                         # angle_x_angular_vel
        x * vx,                                # x_pos_x_x_vel
        y * vy,                                # y_pos_x_y_vel
        1.0 if (y < 0.3 and total_vel > 0.5) else 0.0,  # low_alt_high_speed
    ]


# ──────────────────────────────────────────────
# DATA WRITER  (dual .arff + .csv)
# ──────────────────────────────────────────────

class DataWriter:
    """
    Handles writing instances to .arff and .csv files simultaneously.
    Uses a one-step buffer so that the next-timestep reward can be
    appended to the previous instance before flushing it to disk.
    """

    def __init__(self, base_filename):
        """
        Args:
            base_filename: e.g. "data_agent" or "data_keyboard"
                           Files created: <base_filename>.arff, <base_filename>.csv
        """
        self.arff_path = f"{base_filename}.arff"
        self.csv_path = f"{base_filename}.csv"

        # Open files
        self.arff_file = open(self.arff_path, "w")
        self.csv_file = open(self.csv_path, "w")

        # Write headers
        self._write_arff_header()
        self._write_csv_header()

        # One-step buffer: stores the previous instance (without next_reward)
        self.buffer = None

        self.instance_count = 0

    def _write_arff_header(self):
        """Write the ARFF header with relation name and attribute declarations."""
        self.arff_file.write("@RELATION lunarlander\n\n")

        # Raw attributes (all numeric)
        for attr in RAW_ATTRIBUTES:
            self.arff_file.write(f"@ATTRIBUTE {attr} NUMERIC\n")

        # Engineered attributes (all numeric)
        for attr in ENGINEERED_ATTRIBUTES:
            self.arff_file.write(f"@ATTRIBUTE {attr} NUMERIC\n")

        # Action — nominal with 4 possible values
        self.arff_file.write("@ATTRIBUTE action {0,1,2,3}\n")

        # Next-timestep reward — numeric (regression target)
        self.arff_file.write("@ATTRIBUTE next_reward NUMERIC\n")

        self.arff_file.write("\n@DATA\n")
        self.arff_file.flush()

    def _write_csv_header(self):
        """Write the CSV header row."""
        self.csv_file.write(",".join(ALL_ATTRIBUTES) + "\n")
        self.csv_file.flush()

    def buffer_instance(self, game):
        """
        Store the current state + action in the buffer.
        If there was a previous buffered instance, it is flushed first
        (this should not happen in normal flow — see flush_with_reward).
        """
        # Build feature vector: raw + engineered + action
        raw = [
            game.x_position,
            game.y_position,
            game.x_velocity,
            game.y_velocity,
            game.angle,
            game.angular_velocity,
            game.left_leg_contact,
            game.right_leg_contact,
        ]
        engineered = compute_engineered_features(game)
        action = game.action

        self.buffer = raw + engineered + [action]

    def flush_with_reward(self, next_reward):
        """
        Append the next-timestep reward to the buffered instance
        and write the complete line to both files.
        """
        if self.buffer is None:
            return

        full_instance = self.buffer + [next_reward]

        # Format values: round floats, keep integers clean
        parts = []
        for v in full_instance:
            if isinstance(v, float):
                parts.append(f"{v:.6f}")
            else:
                parts.append(str(v))

        line = ",".join(parts)

        self.arff_file.write(line + "\n")
        self.csv_file.write(line + "\n")

        self.instance_count += 1

        # Flush every 100 instances for safety
        if self.instance_count % 100 == 0:
            self.arff_file.flush()
            self.csv_file.flush()

        # Clear buffer
        self.buffer = None

    def discard_buffer(self):
        """Discard the buffered instance (e.g. on episode end with no next step)."""
        self.buffer = None

    def close(self):
        """Flush and close both files."""
        self.arff_file.flush()
        self.csv_file.flush()
        self.arff_file.close()
        self.csv_file.close()
        print(f"[DataWriter] Saved {self.instance_count} instances to:")
        print(f"  -> {self.arff_path}")
        print(f"  -> {self.csv_path}")


# ──────────────────────────────────────────────
# print_line_data  (required by assignment)
# ──────────────────────────────────────────────

def print_line_data(game, writer):
    """
    Record one instance of game state data.

    This function buffers the current state + action. The actual write
    happens on the NEXT call (or on episode end), when the next-timestep
    reward becomes available.

    Args:
        game:   current GameState object (already updated with this step's
                observation and reward).
        writer: DataWriter instance that handles file I/O.
    """
    # The reward received THIS step is the "next_reward" for the PREVIOUS instance.
    # So first, flush the previous buffer with this step's reward.
    # (If there's nothing buffered yet — e.g. first step — this is a no-op.)
    # NOTE: reward for flush is passed from the game loop, not here.

    # Buffer the current state + action for the next iteration.
    writer.buffer_instance(game)


# ──────────────────────────────────────────────
# PRINT STATE (terminal debug)
# ──────────────────────────────────────────────

def print_state(game):
    print("--------GAME STATE--------")
    print(f"Position: X={game.x_position:.3f}, Y={game.y_position:.3f}")
    print(f"Velocity: X={game.x_velocity:.3f}, Y={game.y_velocity:.3f}")
    print(f"Angle: {game.angle:.3f} rad ({game.angle * 180 / 3.14159:.1f} deg)")
    print(f"Angular Velocity: {game.angular_velocity:.3f}")
    print(f"Left Leg Contact: {game.left_leg_contact:.1f}")
    print(f"Right Leg Contact: {game.right_leg_contact:.1f}")
    print(f"Score: {game.score:.2f}")
    print(f"Last Action: {game.action}")
    print("--------------------------")


# ──────────────────────────────────────────────
# RULE-BASED AGENT (Tutorial 1)
# ──────────────────────────────────────────────

def move_tutorial_1(game):
    LIMIT_SPIN = 0.03
    MAX_TILT = 0.25
    CENTER_TOLERANCE = 0.05

    is_centered = abs(game.x_position) < CENTER_TOLERANCE

    if is_centered:
        target_vy = -(game.y_position * 0.5 + 0.1)
    else:
        target_vy = -0.05

    desired_tilt = (game.x_position * 1.0) + (game.x_velocity * 1.5)

    if desired_tilt > MAX_TILT:
        desired_tilt = MAX_TILT
    if desired_tilt < -MAX_TILT:
        desired_tilt = -MAX_TILT

    if game.y_position < 0.3 and not is_centered:
        if desired_tilt > 0.1:
            desired_tilt = 0.1
        if desired_tilt < -0.1:
            desired_tilt = -0.1

    angle_error = desired_tilt - game.angle

    if angle_error > 0.02:
        if game.angular_velocity < 0.2:
            return ACTION_LEFT_ENGINE
    elif angle_error < -0.02:
        if game.angular_velocity > -0.2:
            return ACTION_RIGHT_ENGINE

    if game.angular_velocity > LIMIT_SPIN * 2:
        return ACTION_RIGHT_ENGINE
    if game.angular_velocity < -LIMIT_SPIN * 2:
        return ACTION_LEFT_ENGINE

    if game.y_velocity < target_vy:
        if abs(game.angle) < 0.3:
            return ACTION_MAIN_ENGINE

    return ACTION_NOTHING


# ──────────────────────────────────────────────
# KEYBOARD CONTROL
# ──────────────────────────────────────────────

def move_keyboard(keys_pressed):
    if keys_pressed[pygame.K_UP] or keys_pressed[pygame.K_w]:
        return ACTION_MAIN_ENGINE
    elif keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
        return ACTION_LEFT_ENGINE
    elif keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
        return ACTION_RIGHT_ENGINE
    else:
        return ACTION_NOTHING


# ──────────────────────────────────────────────
# MAIN GAME LOOP
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("LUNAR LANDER - Machine Learning (UC3M)")
    print("=" * 50)
    print("\nInitializing environment...")

    pygame.init()

    env = gym.make(ENV_NAME, gravity=GRAVITY, render_mode="human")

    print(f"Environment: {ENV_NAME}")
    print(f"Gravity: {GRAVITY}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    # ── Choose file name based on mode ──
    if USE_AGENT:
        data_filename = "data_agent"
        print("\nRunning in AGENT mode (move_tutorial_1)")
    else:
        data_filename = "data_keyboard"
        print("\nRunning in KEYBOARD mode")
        print("Controls (focus on the game window!):")
        print("  W or UP arrow    -> Fire main engine")
        print("  A or LEFT arrow  -> Fire left engine")
        print("  D or RIGHT arrow -> Fire right engine")
        print("  Q or ESC         -> Quit game")

    print("\nGoal: Land safely on the pad between the two flags!")
    print("-" * 50)

    # ── Initialize DataWriter ──
    writer = DataWriter(data_filename)

    # ── Initialize environment ──
    observation, info = env.reset()
    game = GameState(observation)

    clock = pygame.time.Clock()
    episode_count = 0
    running = True

    try:
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False

            if not running:
                break

            # Determine action
            if USE_AGENT:
                action = move_tutorial_1(game)
            else:
                keys_pressed = pygame.key.get_pressed()
                action = move_keyboard(keys_pressed)

            # Store action in game state BEFORE stepping
            game.action = action

            # ── Buffer current state + action ──
            print_line_data(game, writer)

            # Execute action in environment
            observation, reward, terminated, truncated, info = env.step(action)

            # ── Flush previous buffer with this step's reward ──
            writer.flush_with_reward(reward)

            # Update game state
            game.update(observation, reward)

            # Print state to terminal
            print_state(game)

            # Check if episode ended
            if terminated or truncated:
                episode_count += 1

                # Discard any dangling buffer (the episode ended,
                # so there is no "next reward" for the last state)
                writer.discard_buffer()

                if terminated:
                    if game.score > 0:
                        print(f"\n*** EPISODE {episode_count} COMPLETE! Final Score: {game.score:.2f} ***")
                        if game.left_leg_contact and game.right_leg_contact:
                            print("*** SUCCESSFUL LANDING! ***\n")
                        else:
                            print("*** Landed but not on both legs ***\n")
                    else:
                        print(f"\n*** CRASH! Episode {episode_count} Final Score: {game.score:.2f} ***\n")
                else:
                    print(f"\n*** Episode {episode_count} truncated. Score: {game.score:.2f} ***\n")

                # Reset environment
                time.sleep(1)
                observation, info = env.reset()
                game.reset(observation)
                print("New episode started!\n")

            clock.tick(30)

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    finally:
        writer.close()
        env.close()
        pygame.quit()
        print(f"\nGame ended. Total episodes: {episode_count}")
        print("Thank you for playing!")


if __name__ == "__main__":
    main()