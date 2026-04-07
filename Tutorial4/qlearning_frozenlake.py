import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def main():
    # ==========================================
    # --- CONFIGURATION ---
    # ==========================================
    # MODE can be "TRAIN" or "PLAY"
    MODE = "PLAY" 
    
    # RENDER_TRAINING: Set to True to watch the agent learn (Very slow!)
    # Only applies if MODE is "TRAIN".
    RENDER_TRAINING = False  
    IS_SLIPPERY = False
    QTABLE_FILE = "qtable.txt"
    
    # 1. Environment Initialization
    if MODE == "PLAY" or RENDER_TRAINING:
        render_mode = "human"
    else:
        render_mode = None
        
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=IS_SLIPPERY, render_mode=render_mode)
    
    state_space = env.observation_space.n
    action_space = env.action_space.n
    
    # 2. Load or Initialize Q-Table
    header = "State | Left (0) | Down (1) | Right (2) | Up (3)"
    
    if os.path.exists(QTABLE_FILE):
        print(f"[{MODE} MODE] Existing Q-Table found. Loading from '{QTABLE_FILE}'...")
        # loadtxt automatically ignores header lines starting with '#'
        qtable = np.loadtxt(QTABLE_FILE)
    else:
        print(f"[{MODE} MODE] No previous Q-Table found. Initializing a new one with zeros...")
        qtable = np.zeros((state_space, action_space))

    # ==========================================
    #               TRAINING MODE
    # ==========================================
    if MODE == "TRAIN":
        episodes = 2000
        learning_rate = 0.8
        discount_rate = 0.95
        
        # Exploration parameters
        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.01
        decay_rate = 0.005 
        
        rewards_all_episodes = []

        print(f"Starting training for {episodes} episodes...")

        for episode in range(episodes):
            state, info = env.reset()
            done = False
            truncated = False
            rewards_current_episode = 0
            
            while not (done or truncated):
                # Exploration-exploitation trade-off (Epsilon-greedy)
                if np.random.uniform(0, 1) > epsilon:
                    action = np.argmax(qtable[state, :]) # Exploit
                else:
                    action = env.action_space.sample()   # Explore
                    
                # Take action
                new_state, reward, done, truncated, info = env.step(action)
                
                # ===========================================================
                # TODO 1: IMPLEMENT THE Q-LEARNING UPDATE FORMULA
                # ===========================================================
                # The Bellman equation is: 
                # Q(s,a) = Q(s,a) + learning_rate * [Reward + discount_rate * max(Q(s',a')) - Q(s,a)]
                # Currently, the agent learns nothing! Fix the line below.
                
                qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])
                
                # Transition to next state
                state = new_state
                rewards_current_episode += reward
                
                # Slow down the loop if the student wants to watch the training
                if RENDER_TRAINING:
                    time.sleep(0.1)
                
            # ===========================================================
            # TODO 2: IMPLEMENT EPSILON DECAY
            # ===========================================================
            # Epsilon must decrease over time so the agent stops exploring
            # and starts exploiting its learned policy. 
            # Tip: Use exponential decay -> min_epsilon + (max - min) * exp(-decay * episode)
            
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            
            rewards_all_episodes.append(rewards_current_episode)
            
            # Save the table at the end of EACH EPISODE
            np.savetxt(QTABLE_FILE, qtable, fmt="%.4f", delimiter="\t", header=header)
            
        print("Training finished.")
        env.close()

        # Plot training convergence
        window = 100
        moving_avg = np.convolve(rewards_all_episodes, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.title("Training Convergence: Moving Average of Rewards (100 episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(True)
        plt.show()

    # ==========================================
    #                 PLAY MODE
    # ==========================================
    elif MODE == "PLAY":
        print("Playing 3 episodes using the loaded Q-Table (No exploration)...")
        play_episodes = 3
        
        for ep in range(play_episodes):
            state, info = env.reset()
            done = False
            truncated = False
            print(f"\n--- Playing Episode {ep + 1} ---")
            
            while not (done or truncated):
                # Always choose the best action (Epsilon = 0)
                action = np.argmax(qtable[state, :])
                state, reward, done, truncated, info = env.step(action)
                
                time.sleep(0.5) # Slow down to watch the agent
                
            if reward == 1.0:
                print("Result: Success! Reached the goal.")
            else:
                print("Result: Failed. Fell into a hole.")
                
        env.close()

if __name__ == "__main__":
    main()