import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ================= EXPERIMENT CONFIGURATIONS =================
# You can easily switch between different configurations to test

EXPERIMENTS = {
    "baseline": {
        "ALPHA": 0.1,
        "GAMMA": 0.99,
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.995,
        "EPISODES": 10000,
        "BUCKETS": (3, 3, 6, 6),  # [cart_pos, cart_vel, pole_angle, pole_vel]
    },
    "high_learning_rate": {
        "ALPHA": 0.5,  # CHANGED: Increased from 0.1 to 0.5
        "GAMMA": 0.99,
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.995,
        "EPISODES": 10000,
        "BUCKETS": (3, 3, 6, 6),
        # NOTES: Higher learning rate makes agent learn faster but might overshoot optimal values
    },
    "low_discount": {
        "ALPHA": 0.1,
        "GAMMA": 0.8,  # CHANGED: Reduced from 0.99 to 0.8
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.995,
        "EPISODES": 10000,
        "BUCKETS": (3, 3, 6, 6),
        # NOTES: Lower gamma makes agent focus on immediate rewards rather than future rewards
    },
    "slow_exploration": {
        "ALPHA": 0.1,
        "GAMMA": 0.99,
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.999,  # CHANGED: Slower decay from 0.995 to 0.999
        "EPISODES": 10000,
        "BUCKETS": (3, 3, 6, 6),
        # NOTES: Slower epsilon decay means agent explores longer before exploiting learned policy
    },
    "fast_exploration": {
        "ALPHA": 0.1,
        "GAMMA": 0.99,
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.99,  # CHANGED: Faster decay from 0.995 to 0.99
        "EPISODES": 10000,
        "BUCKETS": (3, 3, 6, 6),
        # NOTES: Faster epsilon decay means agent exploits learned policy sooner
    },
    "fine_discretization": {
        "ALPHA": 0.1,
        "GAMMA": 0.99,
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.995,
        "EPISODES": 10000,  # More episodes needed for finer discretization
        "BUCKETS": (5, 5, 10, 10),  # CHANGED: Finer state space discretization
        # NOTES: More buckets = more precise state representation but needs more episodes to learn
    },
}

# ================= SELECT EXPERIMENT =================
EXPERIMENT_NAME = "baseline"  # Change this to try different experiments
CONFIG = EXPERIMENTS[EXPERIMENT_NAME]

# ================= ENVIRONMENT CONFIG =================
ENV_ID = "CartPole-v1"
EPSILON_MIN = 0.01

# Output files - experiment-specific names
OUTPUT_DIR = f"outputs_{EXPERIMENT_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_FILE = os.path.join(OUTPUT_DIR, "q_table.npy")
REWARD_PNG = os.path.join(OUTPUT_DIR, f"training_rewards_{EXPERIMENT_NAME}.png")
EPSILON_PNG = os.path.join(OUTPUT_DIR, f"epsilon_decay_{EXPERIMENT_NAME}.png")
COMBINED_PNG = os.path.join(OUTPUT_DIR, f"training_overview_{EXPERIMENT_NAME}.png")
LOG_FILE = os.path.join(OUTPUT_DIR, f"training_log_{EXPERIMENT_NAME}.txt")

# =========================================

def discretize_state(state, buckets):
    """
    Convert continuous CartPole state to discrete buckets
    CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    """
    # Define bounds for each state variable (based on CartPole environment)
    upper_bounds = [2.4, 3.0, 0.5, 3.0]
    lower_bounds = [-2.4, -3.0, -0.5, -3.0]
    
    ratios = [
        (state[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i])
        for i in range(len(state))
    ]
    
    new_state = [
        int(round((buckets[i] - 1) * ratios[i]))
        for i in range(len(state))
    ]
    
    # Clip to valid bucket range
    new_state = [
        min(buckets[i] - 1, max(0, new_state[i]))
        for i in range(len(state))
    ]
    
    return tuple(new_state)

def choose_action(q_table, state, epsilon, num_actions):
    """Epsilon-greedy action selection"""
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    return np.argmax(q_table[state])

def train():
    """Train Q-Learning agent on CartPole"""
    
    # Initialize environment
    env = gym.make(ENV_ID)
    num_actions = env.action_space.n
    
    # Initialize Q-table with zeros
    q_table = np.zeros(CONFIG["BUCKETS"] + (num_actions,))
    
    # Training tracking
    rewards_history = []
    epsilon_history = []
    avg_rewards = []  # Moving average
    epsilon = CONFIG["EPSILON"]
    
    # Open log file
    log_file = open(LOG_FILE, 'w')
    log_file.write(f"=== CartPole Q-Learning Training ===\n")
    log_file.write(f"Experiment: {EXPERIMENT_NAME}\n")
    log_file.write(f"Configuration:\n")
    for key, value in CONFIG.items():
        log_file.write(f"  {key}: {value}\n")
    log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    print(f"Starting training: {EXPERIMENT_NAME}")
    print(f"Episodes: {CONFIG['EPISODES']}, Alpha: {CONFIG['ALPHA']}, Gamma: {CONFIG['GAMMA']}")
    print(f"Epsilon Decay: {CONFIG['EPSILON_DECAY']}, Buckets: {CONFIG['BUCKETS']}")
    print("-" * 60)
    
    # Training loop
    for episode in range(CONFIG["EPISODES"]):
        obs, _ = env.reset()
        state = discretize_state(obs, CONFIG["BUCKETS"])
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Choose action
            action = choose_action(q_table, state, epsilon, num_actions)
            
            # Take action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Discretize new state
            next_state = discretize_state(obs, CONFIG["BUCKETS"])
            
            # Q-Learning update rule
            # Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
            best_next_action = np.max(q_table[next_state])
            q_table[state][action] += CONFIG["ALPHA"] * (
                reward + CONFIG["GAMMA"] * best_next_action - q_table[state][action]
            )
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Decay epsilon
        epsilon = max(epsilon * CONFIG["EPSILON_DECAY"], EPSILON_MIN)
        
        # Track metrics
        rewards_history.append(total_reward)
        epsilon_history.append(epsilon)
        
        # Calculate moving average (last 100 episodes)
        if len(rewards_history) >= 100:
            avg_reward = np.mean(rewards_history[-100:])
        else:
            avg_reward = np.mean(rewards_history)
        avg_rewards.append(avg_reward)
        
        # Print progress
        if episode % 100 == 0:
            log_msg = f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Avg(100): {avg_reward:6.2f} | Epsilon: {epsilon:.4f}"
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()
    
    env.close()
    
    # Final statistics
    final_avg = np.mean(rewards_history[-100:])
    max_reward = np.max(rewards_history)
    
    log_file.write(f"\n=== Training Complete ===\n")
    log_file.write(f"Final average reward (last 100 episodes): {final_avg:.2f}\n")
    log_file.write(f"Maximum reward achieved: {max_reward:.2f}\n")
    log_file.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.close()
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Final average reward (last 100 episodes): {final_avg:.2f}")
    print(f"Maximum reward: {max_reward:.2f}")
    
    # Save Q-table
    np.save(MODEL_FILE, q_table)
    print(f"Model saved: {MODEL_FILE}")
    
    # Create plots
    create_plots(rewards_history, epsilon_history, avg_rewards)
    
    return rewards_history, epsilon_history

def create_plots(rewards_history, epsilon_history, avg_rewards):
    """Create and save training visualization plots"""
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Q-Learning Training Results - {EXPERIMENT_NAME}', fontsize=16)
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(rewards_history, alpha=0.6, linewidth=0.8)
    axes[0, 0].plot(avg_rewards, linewidth=2, label='Moving Avg (100 episodes)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Rewards per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Epsilon Decay
    axes[0, 1].plot(epsilon_history, color='orange')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon Value')
    axes[0, 1].set_title('Exploration Rate (Epsilon) Decay')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Moving Average Reward
    axes[1, 0].plot(avg_rewards, color='green', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Reward (100 episodes)')
    axes[1, 0].set_title('Learning Progress')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution (last 500 episodes)
    recent_rewards = rewards_history[-500:] if len(rewards_history) > 500 else rewards_history
    axes[1, 1].hist(recent_rewards, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Reward Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution (Last 500 Episodes)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(COMBINED_PNG, dpi=150)
    print(f"Combined plots saved: {COMBINED_PNG}")
    plt.close()
    
    # Individual plots for easier viewing
    # Rewards plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, alpha=0.6)
    plt.plot(avg_rewards, linewidth=2, color='red', label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Rewards - {EXPERIMENT_NAME}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(REWARD_PNG, dpi=150)
    print(f"Reward plot saved: {REWARD_PNG}")
    plt.close()
    
    # Epsilon plot
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_history, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title(f'Epsilon Decay - {EXPERIMENT_NAME}')
    plt.grid(True, alpha=0.3)
    plt.savefig(EPSILON_PNG, dpi=150)
    print(f"Epsilon plot saved: {EPSILON_PNG}")
    plt.close()

def play(render=True):
    """Test the trained Q-Learning agent"""
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file not found: {MODEL_FILE}")
        print("Please train the model first.")
        return
    
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    q_table = np.load(MODEL_FILE)
    
    num_test_episodes = 5
    total_rewards = []
    
    print(f"\n{'='*60}")
    print(f"Testing trained agent for {num_test_episodes} episodes...")
    print(f"{'='*60}\n")
    
    for episode in range(num_test_episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, CONFIG["BUCKETS"])
        done = False
        total_reward = 0
        
        while not done:
            # Always choose best action (no exploration)
            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = discretize_state(obs, CONFIG["BUCKETS"])
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"Test Results Summary:")
    print(f"  Average reward: {np.mean(total_rewards):.2f}")
    print(f"  Max reward: {np.max(total_rewards):.2f}")
    print(f"{'='*60}")

def play_specific_experiment(render=True):
    """Test a specific trained experiment by selecting its saved Q-table"""
    
    print("\nAvailable trained experiments:")
    for name in EXPERIMENTS.keys():
        print(f" - {name}")
    
    exp_name = input("\nEnter experiment name to test: ").strip()
    
    if exp_name not in EXPERIMENTS:
        print(f"Invalid experiment name: {exp_name}")
        return
    
    model_path = f"outputs_{exp_name}/q_table.npy"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Train this experiment first.")
        return
    
    # Temporarily switch config for correct discretization
    global EXPERIMENT_NAME, CONFIG, MODEL_FILE
    EXPERIMENT_NAME = exp_name
    CONFIG = EXPERIMENTS[exp_name]
    MODEL_FILE = model_path
    
    print(f"\nTesting model from: {MODEL_FILE}")
    
    play(render=render)

# ================= MAIN =================
if __name__ == "__main__":
    print("=" * 60)
    print("CartPole Q-Learning Training")
    print("=" * 60)
    print("\n1. Train agent")
    print("2. Test trained agent")
    print("3. Run all experiments (compare configurations)")
    print("4. Test a specific trained experiment")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice == "1":
        train()
    elif choice == "2":
        play(render=True)
    elif choice == "3":
        print("\nRunning all experiments...")
        print("This will train the agent with different configurations.")
        print("Results will be saved in separate folders.\n")
        
        all_results = {}
        
        for exp_name in EXPERIMENTS.keys():
            print(f"\n{'='*60}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*60}\n")
            
            # Update configuration
            EXPERIMENT_NAME = exp_name
            CONFIG = EXPERIMENTS[exp_name]
            OUTPUT_DIR = f"outputs_{exp_name}"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            MODEL_FILE = os.path.join(OUTPUT_DIR, "q_table.npy")
            REWARD_PNG = os.path.join(OUTPUT_DIR, f"training_rewards_{exp_name}.png")
            EPSILON_PNG = os.path.join(OUTPUT_DIR, f"epsilon_decay_{exp_name}.png")
            COMBINED_PNG = os.path.join(OUTPUT_DIR, f"training_overview_{exp_name}.png")
            LOG_FILE = os.path.join(OUTPUT_DIR, f"training_log_{exp_name}.txt")
            
            rewards, epsilon = train()
            all_results[exp_name] = {
                'rewards': rewards,
                'final_avg': np.mean(rewards[-100:])
            }
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        for exp_name, results in all_results.items():
            avg_rewards = [np.mean(results['rewards'][max(0, i-99):i+1]) 
                          for i in range(len(results['rewards']))]
            plt.plot(avg_rewards, label=f"{exp_name} (final: {results['final_avg']:.1f})", linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (100 episodes)')
        plt.title('Comparison of All Experiments - CartPole')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('comparison_all_experiments_cartpole.png', dpi=150)
        print(f"\nComparison plot saved: comparison_all_experiments_cartpole.png")
        plt.close()
        
        print("\n" + "="*60)
        print("All experiments complete!")
        print("="*60)
    elif choice == "4":
        play_specific_experiment(render=True)
    else:
        print("Invalid choice")