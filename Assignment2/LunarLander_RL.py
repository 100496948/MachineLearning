"""
Lunar Lander
Made with Gymnasium
January 2026 - Machine Learning Classes
University Carlos III of Madrid

This version integrates:
- Phase 1: manual discretization + reward + Q-learning + tuning
- Phase 2: VQQL + experiments with number of clusters + training/evaluation
- Phase 3: comparison of Q-learning models + comparison with Tutorial 1 and Assignment 1

Modes:
- TRAIN: learn a Q-table
- PLAY: load a Q-table and watch the learned agent
- "VQQL" -> use the quantizer
- "MANUAL" -> use a hand-crafted discretization
"""

import gymnasium as gym
import time
import pygame
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================

# Moon gravity   -> -1.62
# Mars gravity   -> -3.72
# Earth gravity  -> -9.81
GRAVITY = -1.62

ENV_NAME = "LunarLander-v3"
QUANTIZER_FILE = "lunarlander_vq.pkl"
QTABLE_FILE = "qtable_phase1_best.txt"

# MODE can be: "TRAIN", "PLAY"
MODE = "PLAY"

# State representation mode:
# "VQQL" -> use the quantizer
# "MANUAL" -> use a hand-crafted discretization
STATE_MODE = "MANUAL"

# If True, use the custom reward from compute_reward()
# If False, use the raw reward from Gymnasium
USE_CUSTOM_REWARD = True

# If True, render the environment during training (very slow)
RENDER_TRAINING = False

# Training parameters
EPISODES = 5000
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95

# Exploration parameters
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.001

# ACTION_REPEAT reduces the frequency of decision-making by repeating
# the same action for multiple environment steps.
# This helps stabilize learning and reduce redundant state transitions.
ACTION_REPEAT = 5

# Evaluation / play parameters
PLAY_EPISODES = 10

# Action definitions
ACTION_NOTHING = 0
ACTION_LEFT_ENGINE = 1
ACTION_MAIN_ENGINE = 2
ACTION_RIGHT_ENGINE = 3


# ==========================================
# MANUAL DISCRETIZATION CONFIGURATION
# ==========================================
# Bin edges for each continuous variable.
# np.digitize(value, edges) returns an integer in [0, len(edges)],
# so the number of bins per variable is len(edges) + 1.
#
# Design rationale:
# - x_position: 5 bins, narrow band around 0 (landing zone is centered at x=0)
# - y_position: 5 bins, more resolution near the ground where decisions matter most
# - x_velocity: 3 bins (moving left / ~still / moving right)
# - y_velocity: 3 bins (falling fast / normal descent / ascending or hovering)
# - angle: 3 bins (tilted left / upright / tilted right) - tight around 0
# - angular_velocity: 3 bins (rotating left / stable / rotating right)
# - left_leg, right_leg: 2 bins each (boolean: contact or no contact)

BINS_X_POS    = [-0.5, -0.1, 0.1, 0.5]   # -> 5 bins
BINS_Y_POS    = [0.1, 0.4, 0.8, 1.2]     # -> 5 bins
BINS_X_VEL    = [-0.2, 0.2]              # -> 3 bins
BINS_Y_VEL    = [-0.4, -0.1]             # -> 3 bins
BINS_ANGLE    = [-0.2, 0.2]              # -> 3 bins
BINS_ANG_VEL  = [-0.3, 0.3]              # -> 3 bins

N_X_POS   = len(BINS_X_POS)   + 1   # 5
N_Y_POS   = len(BINS_Y_POS)   + 1   # 5
N_X_VEL   = len(BINS_X_VEL)   + 1   # 3
N_Y_VEL   = len(BINS_Y_VEL)   + 1   # 3
N_ANGLE   = len(BINS_ANGLE)   + 1   # 3
N_ANG_VEL = len(BINS_ANG_VEL) + 1   # 3
N_LEFT    = 2
N_RIGHT   = 2

# Total: 5 * 5 * 3 * 3 * 3 * 3 * 2 * 2 = 8100 states


# GAME STATE CLASS
class GameState:
    def __init__(self, observation):
        """
        Initialize game state from Gymnasium observation.

        The LunarLander-v3 observation space consists of 8 values:
        - obs[0]: x position
        - obs[1]: y position
        - obs[2]: x velocity
        - obs[3]: y velocity
        - obs[4]: angle
        - obs[5]: angular velocity
        - obs[6]: left leg contact
        - obs[7]: right leg contact
        """
        self.x_position = observation[0]
        self.y_position = observation[1]
        self.x_velocity = observation[2]
        self.y_velocity = observation[3]
        self.angle = observation[4]
        self.angular_velocity = observation[5]
        self.left_leg_contact = observation[6]
        self.right_leg_contact = observation[7]

        # Store raw observation for convenience
        self.observation = observation

        # Score tracking
        self.score = 0.0
        self.episode_reward = 0.0

        # Current action
        self.action = ACTION_NOTHING

    def update(self, observation, reward):
        """Update state with new observation and reward."""
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
        """Reset state for a new episode."""
        self.__init__(observation)

# ==========================================
# PHASE 1 STEP 1 - MANUAL DISCRETIZATION
# ==========================================

def get_manual_state_space():
    """
    Return the total number of discrete states.

    The total number of states is the product of the number of bins
    of each discretized variable. This determines the number of rows
    in the Q-table.

    With the chosen bin configuration:
        5 (x_pos) * 5 (y_pos) * 3 (x_vel) * 3 (y_vel)
        * 3 (angle) * 3 (ang_vel) * 2 (left_leg) * 2 (right_leg) = 8100
    """
    return (N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL
            * N_ANGLE * N_ANG_VEL * N_LEFT * N_RIGHT)


def manual_state_id(game):
    """
    Convert the continuous state of the environment into a unique discrete state ID.

    Steps:
    1. Discretize each variable using np.digitize and the bin edges.
    2. Combine all discrete values into a single integer using mixed-radix encoding.

    Mixed-radix encoding guarantees that:
    - Each combination of bin indices maps to a UNIQUE integer.
    - The resulting state_id lies in [0, get_manual_state_space() - 1].

    The formula is:
        state_id = i_0
                 + i_1 * N_0
                 + i_2 * N_0 * N_1
                 + i_3 * N_0 * N_1 * N_2
                 + ...
    where i_k is the bin index of variable k and N_k is the number of bins.
    """
    # Discretize each continuous variable
    i_x   = int(np.digitize(game.x_position,       BINS_X_POS))
    i_y   = int(np.digitize(game.y_position,       BINS_Y_POS))
    i_vx  = int(np.digitize(game.x_velocity,       BINS_X_VEL))
    i_vy  = int(np.digitize(game.y_velocity,       BINS_Y_VEL))
    i_a   = int(np.digitize(game.angle,            BINS_ANGLE))
    i_va  = int(np.digitize(game.angular_velocity, BINS_ANG_VEL))

    # Legs are already binary (0.0 or 1.0)
    i_l = int(game.left_leg_contact)
    i_r = int(game.right_leg_contact)

    # Mixed-radix encoding to combine all indices into a single state ID
    state_id  = i_x
    state_id += i_y  * N_X_POS
    state_id += i_vx * N_X_POS * N_Y_POS
    state_id += i_vy * N_X_POS * N_Y_POS * N_X_VEL
    state_id += i_a  * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL
    state_id += i_va * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL * N_ANGLE
    state_id += i_l  * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL * N_ANGLE * N_ANG_VEL
    state_id += i_r  * N_X_POS * N_Y_POS * N_X_VEL * N_Y_VEL * N_ANGLE * N_ANG_VEL * N_LEFT

    return int(state_id)

# ==========================================
# PHASE 1 STEP 2 - NEW REWARD FUNCTION
# ==========================================

def compute_reward(observation, raw_reward, terminated, truncated):
    """
    Compute a custom (shaped) reward based on the environment reward
    and additional shaping terms.

    Strategy:
    - Keep the original Gymnasium reward as the main signal.
    - Add small penalty terms that push the agent toward desirable behavior:
        * being centered horizontally (close to x = 0)
        * descending slowly (small |y_velocity|)
        * staying upright (small |angle|)
        * not spinning (small |angular_velocity|)
    - Add a small bonus when a leg is in contact with the ground (encourages
      the agent to actually land instead of hovering).

    The shaping coefficients are kept small so they do not overwhelm the
    original reward, which is critical for learning a successful landing.
    """
    x_position = observation[0]
    y_position = observation[1]
    x_velocity = observation[2]
    y_velocity = observation[3]
    angle = observation[4]
    angular_velocity = observation[5]
    left_leg = observation[6]
    right_leg = observation[7]

    # Start from the original Gymnasium reward
    reward = raw_reward

    # --- Shaping terms (small coefficients) ---

    # Penalize horizontal distance from the landing zone (x = 0)
    reward -= 0.10 * abs(x_position)

    # Penalize fast descent (y_velocity is negative when falling)
    reward -= 0.10 * abs(y_velocity)

    # Penalize tilt (we want the lander upright)
    reward -= 0.20 * abs(angle)

    # Penalize spinning
    reward -= 0.05 * abs(angular_velocity)

    # Reward leg contact (encourages landing rather than hovering)
    reward += 0.10 * (left_leg + right_leg)

    return reward

# ==========================================
# PHASE 1 STEP 3 - Q LEARNING
# ==========================================

def choose_action(state_id, qtable, epsilon, env):
    """
    Choose an action using an epsilon-greedy strategy.

    Epsilon-greedy works as follows:
    - with probability epsilon, choose a random action (exploration)
    - with probability 1 - epsilon, choose the best known action
      from the Q-table (exploitation)

    Parameters:
    - state_id: current discrete state
    - qtable: Q-value table
    - epsilon: exploration probability
    - env: Gymnasium environment (used to sample random actions)

    Returns:
    - action: integer in the action space
    """

    if np.random.uniform(0, 1) > epsilon:
        return int(np.argmax(qtable[state_id, :]))  # exploit
    return env.action_space.sample()                # explore


def update_qtable(qtable, state_id, action, reward, next_state_id,
                  learning_rate, discount_rate):
    """
    Apply the Q-learning Bellman update.

    Bellman equation:
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]

    The term in brackets is the "TD error":
    - (reward + gamma * max Q(s',.))  is the new estimate of Q(s,a)
      built from the immediate reward plus the best possible future value.
    - Q(s,a) is the previous estimate.
    - The agent moves its current Q value a fraction (alpha) toward the new estimate.

    Parameters:
    - qtable: Q-value table (modified in place)
    - state_id: current discrete state s
    - action: action a taken in state s
    - reward: reward received after taking action a
    - next_state_id: next discrete state s'
    - learning_rate: alpha
    - discount_rate: gamma
    """
    best_next_q = np.max(qtable[next_state_id, :])
    td_target   = reward + discount_rate * best_next_q
    td_error    = td_target - qtable[state_id, action]
    qtable[state_id, action] = qtable[state_id, action] + learning_rate * td_error


def decay_epsilon(episode, max_epsilon, min_epsilon, decay_rate):
    """
    Compute exponential epsilon decay over time.

    The exploration rate decreases as training progresses, so the agent
    gradually shifts from exploration to exploitation:

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * exp(-decay_rate * episode)

    Parameters:
    - episode: current episode index
    - max_epsilon: initial epsilon (high exploration)
    - min_epsilon: minimum epsilon value (always keep some exploration)
    - decay_rate: decay speed (higher -> faster decay)

    Returns:
    - epsilon value for the current episode
    """
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)



# ==========================================
# PHASE 2 - VQQL DISCRETIZATION
# ==========================================

def load_quantizer(path):
    """
    Load the scaler and clustering model created in Phase 1.

    The .pkl file should contain at least:
    - scaler
    - quantizer
    - state_columns
    - n_clusters
    """
    data = joblib.load(path)

    scaler = data["scaler"]
    quantizer = data["quantizer"]
    state_columns = data["state_columns"]
    n_clusters = data["n_clusters"]

    return scaler, quantizer, state_columns, n_clusters


def extract_features(game, state_columns):
    """
    Build the feature vector in the same order used in Phase 1.

    IMPORTANT:
    The order of the variables must be exactly the same as the one used
    when the quantizer was trained.

    TODO:
    Students should verify that the state representation used here
    matches the one selected in the VQQL Discretization.
    """

    feature_map = {
        "x_position": game.x_position,
        "y_position": game.y_position,
        "x_velocity": game.x_velocity,
        "y_velocity": game.y_velocity,
        "angle": game.angle,
        "angular_velocity": game.angular_velocity,
        "left_leg": game.left_leg_contact,
        "right_leg": game.right_leg_contact,
    }

    # Build the feature vector following the order stored in the .pkl file
    features = np.array([feature_map[col] for col in state_columns], dtype=np.float64)

    return features


def discretize_state(game, scaler, quantizer, state_columns):
    """
    Convert the current continuous state into a discrete state id.

    Steps:
    1. Extract the selected features
    2. Apply the same scaling
    3. Find the nearest centroid
    4. Return its index
    """
    features = extract_features(game, state_columns)

    # Scale the features before clustering
    scaled_features = scaler.transform([features]).astype(np.float64)

    # Predict the nearest centroid index
    state_id = quantizer.predict(scaled_features)[0]

    return int(state_id)


def get_state_id(game, scaler=None, quantizer=None, state_columns=None):
    """
    Return the discrete state identifier used as row index in the Q-table.
    """
    if STATE_MODE == "VQQL":
        return discretize_state(game, scaler, quantizer, state_columns)
    elif STATE_MODE == "MANUAL":
        return manual_state_id(game)
    else:
        raise ValueError("STATE_MODE must be 'VQQL' or 'MANUAL'")


# ==========================================
# PLOTTING
# ==========================================

def plot_training_curve(rewards_all_episodes, window=100, title=None):
    if len(rewards_all_episodes) < window:
        window = max(1, len(rewards_all_episodes))

    moving_avg = np.convolve(
        rewards_all_episodes,
        np.ones(window) / window,
        mode='valid'
    )

    plt.plot(moving_avg)
    plt.title(title if title else f"Training Convergence: Moving Average of Rewards ({window} steps)")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()

# ==========================================
# QUIT GAME
# ==========================================

def handle_pygame_events():
    """
    Handle window close and keyboard input.
    Returns True if the program should exit.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                return True

    return False

# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 50)
    print("LUNAR LANDER - Machine Learning (UC3M)")
    print("=" * 50)
    print("\nInitializing environment...")

    pygame.init()

    # Render settings
    if MODE == "PLAY" or RENDER_TRAINING:
        render_mode = "human"
    else:
        render_mode = None

    env = gym.make(ENV_NAME, gravity=GRAVITY, render_mode=render_mode)

    print(f"Environment: {ENV_NAME}")
    print(f"Gravity: {GRAVITY}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    if STATE_MODE == "VQQL":
        scaler, quantizer, state_columns, n_clusters = load_quantizer(QUANTIZER_FILE)
        print(f"Loaded quantizer from {QUANTIZER_FILE}")
        print(f"Number of discrete states (clusters): {n_clusters}")
        print(f"State columns used: {state_columns}")
        n_states = n_clusters

    elif STATE_MODE == "MANUAL":
        scaler, quantizer, state_columns = None, None, None
        n_states = get_manual_state_space()
        print("Using manual state discretization.")

    else:
        raise ValueError("STATE_MODE must be 'VQQL' or 'MANUAL'")

    action_space = env.action_space.n

    print(f"STATE_MODE: {STATE_MODE}")
    print(f"Q-table state space size: {n_states}")

    # Load or initialize Q-table
    if os.path.exists(QTABLE_FILE):
        print(f"[{MODE} MODE] Existing Q-Table found. Loading from '{QTABLE_FILE}'...")
        qtable = np.loadtxt(QTABLE_FILE)

        if qtable.ndim == 1:
            qtable = qtable.reshape(1, -1)
    else:
        print(f"[{MODE} MODE] No previous Q-Table found. Initializing a new one with zeros...")
        qtable = np.zeros((n_states, action_space), dtype=np.float64)

    # Sanity check
    if qtable.shape != (n_states, action_space):
        raise ValueError(
            f"Loaded Q-table shape {qtable.shape} does not match expected {(n_states, action_space)}"
        )

    # ==========================================
    # ONLINE TRAINING MODE
    # ==========================================
    if MODE == "TRAIN":
        epsilon = MAX_EPSILON
        rewards_all_episodes = []

        print(f"Starting online training for {EPISODES} episodes...")

        for episode in range(EPISODES):
            observation, info = env.reset()
            game = GameState(observation)

            terminated = False
            truncated = False
            rewards_current_episode = 0.0

            while not (terminated or truncated):

                # Check for exit events
                if handle_pygame_events():
                    print("Exiting...")
                    env.close()
                    pygame.quit()
                    return

                # Current discrete state
                state_id = get_state_id(game, scaler, quantizer, state_columns)

                # Choose one action and keep it for several environment steps
                action = choose_action(state_id, qtable, epsilon, env)
                game.action = action

                total_reward = 0.0

                for _ in range(ACTION_REPEAT):
                    observation, raw_reward, terminated, truncated, info = env.step(action)

                    # Reward: raw or custom
                    if USE_CUSTOM_REWARD:
                        step_reward = compute_reward(observation, raw_reward, terminated, truncated)
                    else:
                        step_reward = raw_reward

                    total_reward += step_reward

                    if RENDER_TRAINING:
                        print(f"State: {state_id} | Action: {action} | Step reward: {step_reward:.3f}")
                        time.sleep(0.03)

                    # Stop repeating if episode ended
                    if terminated or truncated:
                        break

                # Next discrete state after the macro-action
                next_game = GameState(observation)
                next_state_id = get_state_id(next_game, scaler, quantizer, state_columns)

                # Bellman update using accumulated reward
                update_qtable(
                    qtable,
                    state_id,
                    action,
                    total_reward,
                    next_state_id,
                    LEARNING_RATE,
                    DISCOUNT_RATE
                )

                # Update current game state
                game.update(observation, total_reward)
                rewards_current_episode += total_reward

            # Epsilon decay
            epsilon = decay_epsilon(episode, MAX_EPSILON, MIN_EPSILON, DECAY_RATE)

            rewards_all_episodes.append(rewards_current_episode)

            if (episode + 1) % 100 == 0:
                last_rewards = rewards_all_episodes[-100:]
                avg_reward = np.mean(last_rewards)
                std_reward = np.std(last_rewards)

                print(
                    f"AVG last 100: {avg_reward:.2f} | "
                    f"STD: {std_reward:.2f}"
                )

            # Save Q-table after each episode
            header = "a0\ta1\ta2\ta3"
            np.savetxt(QTABLE_FILE, qtable, fmt="%12.4f", delimiter="\t", header=header)

            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Reward: {rewards_current_episode:.2f} | "
                f"Epsilon: {epsilon:.4f}"
            )

        print("Online training finished.")
        env.close()
        pygame.quit()

        plot_training_curve(
            rewards_all_episodes,
            title="Online Q-Learning Convergence"
        )
        return

    # ==========================================
    # PLAY MODE
    # ==========================================
    if MODE == "PLAY":
        print(f"Playing {PLAY_EPISODES} episodes using the loaded Q-Table (No exploration)...")

        for ep in range(PLAY_EPISODES):
            observation, info = env.reset()
            game = GameState(observation)
            terminated = False
            truncated = False

            print(f"\n--- Playing Episode {ep + 1} ---")

            while not (terminated or truncated):

                # Check for exit events
                if handle_pygame_events():
                    print("Exiting...")
                    env.close()
                    pygame.quit()
                    return

                state_id = get_state_id(game, scaler, quantizer, state_columns)

                # Always exploit
                action = int(np.argmax(qtable[state_id, :]))
                game.action = action

                total_reward = 0.0

                for _ in range(ACTION_REPEAT):
                    observation, raw_reward, terminated, truncated, info = env.step(action)

                    if USE_CUSTOM_REWARD:
                        step_reward = compute_reward(observation, raw_reward, terminated, truncated)
                    else:
                        step_reward = raw_reward

                    total_reward += step_reward

                    if terminated or truncated:
                        break

                    time.sleep(0.05)

                game.update(observation, total_reward)

                print(
                    f"Discrete state id: {state_id} | "
                    f"Action: {action} | "
                    f"Accumulated reward: {total_reward:.3f}"
                )

            print(f"Final score: {game.score:.2f}")

        env.close()
        pygame.quit()
        return

    env.close()
    pygame.quit()
    raise ValueError("MODE must be one of: 'TRAIN', 'PLAY'")



if __name__ == "__main__":
    main()