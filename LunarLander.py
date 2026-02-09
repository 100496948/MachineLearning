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
import pygame
import os

# GRAVITY setting
# Moon gravity   -> -1.62
# Mars gravity   -> -3.72
# Earth gravity  -> -9.81 (default, very hard!)
GRAVITY = -1.62

# Agent mode: Set to True to use the Tutorial 1 agent, False for keyboard control
USE_AGENT = True

# Environment configuration
ENV_NAME = "LunarLander-v3"

# we add this for Question 4
CSV_FILE = "data_lunarlander.csv"

# Action definitions
ACTION_NOTHING = 0      # Do nothing
ACTION_LEFT_ENGINE = 1  # Fire left orientation engine
ACTION_MAIN_ENGINE = 2  # Fire main engine (downward thrust)
ACTION_RIGHT_ENGINE = 3 # Fire right orientation engine

# GAME STATE CLASS
class GameState:
    def __init__(self, observation):
        """
        Initialize game state from Gymnasium observation.
        
        The LunarLander-v3 observation space consists of 8 values:
        - obs[0]: x position (horizontal position of the lander)
        - obs[1]: y position (vertical position of the lander)
        - obs[2]: x velocity (horizontal velocity)
        - obs[3]: y velocity (vertical velocity)
        - obs[4]: angle (lander angle)
        - obs[5]: angular velocity (rotation speed)
        - obs[6]: left leg contact (1.0 if touching ground, 0.0 otherwise)
        - obs[7]: right leg contact (1.0 if touching ground, 0.0 otherwise)
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

        # We add these for Question 4
        self.tick = 0
        self.episode = 0
        
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

        # We add this for Question 4
        self.tick += 1

    def reset(self, observation):
        """Reset state for a new episode."""

        # We addthis for Question 4
        episode = self.episode  # preserve episode number
        self.__init__(observation)
        self.episode = episode

def print_state(game):
    """
    Print the current game state to the terminal.
    This function shows all available information about the lander.
    """
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

# TODO: IMPLEMENT HERE THE METHOD TO SAVE DATA TO FILE
def print_line_data(game):
    """
    Return a string with the game state information to be saved to a file.
    
    This method should return a string with the relevant information from
    the game state, with values separated by commas.
    
    The student should decide which features are relevant for the task.
    
    YOUR CODE HERE
    """

    """

    Relevant features selected:
    - episode, tick: to differiantiate every episode and identify each moment in time
    - x_position, y_position: where the lander is
    - x_velocity, y_velocity: how fast it's moving
    - angle, angular_velocity: orientation and rotation
    - left_leg_contact, right_leg_contact: landing status
    - action: what the agent decided
    - score: cumulative reward

    """
    line = (
        f"{game.episode},"
        f"{game.tick},"
        f"{game.x_position:.4f},"
        f"{game.y_position:.4f},"
        f"{game.x_velocity:.4f},"
        f"{game.y_velocity:.4f},"
        f"{game.angle:.4f},"
        f"{game.angular_velocity:.4f},"
        f"{int(game.left_leg_contact)},"
        f"{int(game.right_leg_contact)},"
        f"{game.action},"
        f"{game.score:.4f}"
    )
    return line

# Auziliary functions to handle the csv file

def get_csv_header():
    return "episode,tick,x_pos,y_pos,x_vel,y_vel,angle,angular_vel,left_leg,right_leg,action,score"


def init_csv(filepath):
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write(get_csv_header() + "\n")


def append_line_to_csv(filepath, line):
    with open(filepath, "a") as f:
        f.write(line + "\n")


# TODO: IMPLEMENT HERE THE INTELLIGENT AGENT METHOD
def move_tutorial_1(game):
    """
    Implement your own rule-based agent to land the spacecraft.
    
    This method receives the current game state and must return an action:
    - ACTION_NOTHING (0): Do nothing
    - ACTION_LEFT_ENGINE (1): Fire left orientation engine (rotate clockwise)
    - ACTION_MAIN_ENGINE (2): Fire main engine (slow down descent)
    - ACTION_RIGHT_ENGINE (3): Fire right orientation engine (rotate counter-clockwise)
    
    Goal: Land safely between the two flags on the landing pad.
    - Landing pad is always at coordinates (0, 0)
    - Landing outside the pad is possible but gives less reward
    - Crash (too fast or wrong angle) ends the episode with negative reward
    - Successful landing gives +100 to +140 points
    - Each leg contact gives +10 points
    - Firing main engine costs -0.3 points per frame
    - Firing side engines costs -0.03 points per frame
    
    Tips:
    - Use y_velocity to control descent speed (should be slow when landing)
    - Use angle to keep the lander upright (close to 0)
    - Use x_position and x_velocity to center over the landing pad
    
    YOUR CODE HERE
    """
    # If both legs are on the ground, stop engines (save fuel)
    if game.left_leg_contact and game.right_leg_contact:
        return ACTION_NOTHING

    # 1. Angle control (highest priority - avoid flipping)
    if game.angle > 0.1 or game.angular_velocity > 0.15:
        return ACTION_RIGHT_ENGINE
    if game.angle < -0.1 or game.angular_velocity < -0.15:
        return ACTION_LEFT_ENGINE

    # 2. Vertical speed control (brake harder the closer to ground)
    if game.y_position < 0.3 and game.y_velocity < -0.05:
        return ACTION_MAIN_ENGINE
    if game.y_position < 0.6 and game.y_velocity < -0.2:
        return ACTION_MAIN_ENGINE
    if game.y_position < 1.0 and game.y_velocity < -0.4:
        return ACTION_MAIN_ENGINE
    if game.y_velocity < -0.6:
        return ACTION_MAIN_ENGINE

    # 3. Horizontal centering (steer toward x = 0)
    if game.x_position > 0.3 or (game.x_position > 0.1 and game.x_velocity > 0.1):
        return ACTION_RIGHT_ENGINE
    if game.x_position < - 0.3 or (game.x_position < -0.1 and game.x_velocity < -0.1):
        return ACTION_LEFT_ENGINE

    # Default: do nothing (save fuel)
    return ACTION_NOTHING

def move_keyboard(keys_pressed):
    """
    Convert keyboard input to action.
    
    Controls:
    - UP arrow or W: Fire main engine
    - LEFT arrow or A: Fire left engine
    - RIGHT arrow or D: Fire right engine
    - No key: Do nothing
    
    Args:
        keys_pressed: pygame key state from pygame.key.get_pressed()
    
    Returns:
        Action integer (0-3)
    """
    if keys_pressed[pygame.K_UP] or keys_pressed[pygame.K_w]:
        return ACTION_MAIN_ENGINE
    elif keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
        return ACTION_LEFT_ENGINE
    elif keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
        return ACTION_RIGHT_ENGINE
    else:
        return ACTION_NOTHING

def main():
    """Main game loop."""
    print("=" * 50)
    print("LUNAR LANDER - Machine Learning (UC3M)")
    print("=" * 50)
    print("\nInitializing environment...")
    
    # Initialize pygame for keyboard input
    pygame.init()
    
    # Create the environment with human rendering and configured gravity
    env = gym.make(ENV_NAME, gravity=GRAVITY, render_mode="human")
    
    print(f"Environment: {ENV_NAME}")
    print(f"Gravity: {GRAVITY}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    if USE_AGENT:
        print("\nRunning in AGENT mode (move_tutorial_1)")
    else:
        print("\nRunning in KEYBOARD mode")
        print("Controls (focus on the game window!):")
        print("  W or UP arrow    -> Fire main engine (slow descent)")
        print("  A or LEFT arrow  -> Fire left engine (rotate clockwise)")
        print("  D or RIGHT arrow -> Fire right engine (rotate counter-clockwise)")
        print("  Q or ESC         -> Quit game")
    
    print("\nGoal: Land safely on the pad between the two flags!")
    print("-" * 50)
    
    # Initialize the environment
    observation, info = env.reset()
    game = GameState(observation)

    # We add to iitialize the csv file to store the data (Question 4)
    init_csv(CSV_FILE)
    
    # FPS controller
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
            
            # Determine action based on USE_AGENT variable
            if USE_AGENT:
                action = move_tutorial_1(game)
            else:
                keys_pressed = pygame.key.get_pressed()
                action = move_keyboard(keys_pressed)
            
            # Store action in game state
            game.action = action
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Update game state
            game.update(observation, reward)
            
            # Print state
            print_state(game)

            # We add this for Question 4
            line = print_line_data(game)
            append_line_to_csv(CSV_FILE, line)
            
            # Check if episode ended
            if terminated or truncated:
                episode_count += 1
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
                    print(f"\n*** Episode {episode_count} truncated. Final Score: {game.score:.2f} ***\n")
                
                # Reset environment
                time.sleep(1)
                observation, info = env.reset()
                game.reset(observation)

                # We add this to track every episodes number
                game.episode = episode_count

                print("New episode started!\n")
            
            # Control frame rate
            clock.tick(30)
            
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    finally:
        env.close()
        pygame.quit()
        print(f"\nGame ended. Total episodes: {episode_count}")
        print("Thank you for playing!")

if __name__ == "__main__":
    main()