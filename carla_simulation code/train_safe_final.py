from stable_baselines3 import PPO
from environment import CarEnv
import os
import time
import numpy as np

SEED = 123
TIMESTEPS = 10_000
TOTAL_ITERATIONS = 2  # Increased to allow more training time for ACC behavior
EARLY_STOP_PATIENCE = 5  # Increased to allow for more exploration

# Setup dirs
timestamp = time.strftime("%Y%m%d-%H%M%S")
models_dir = f"models/{timestamp}"
logdir = f"logs/{timestamp}"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

print(f"[INFO] Started ACC training at {timestamp}")
print(f"[INFO] Saving models to {models_dir}")
print(f"[INFO] Logging to TensorBoard at {logdir}")

# Environment & model
env = CarEnv()
env.reset()

# Configure PPO for ACC learning
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir, 
    learning_rate=0.0005,  # Lower learning rate for more stable learning
    gamma=0.99,           # Discount factor
    n_steps=2048,         # Steps to collect before updating
    batch_size=64,        # Minibatch size
    ent_coef=0.01,        # Entropy coefficient for exploration
    n_epochs=10,          # Number of epochs when optimizing the surrogate loss
    seed=SEED
)

best_reward = float('-inf')
worse_count = 0
reward_history = []

for i in range(1, TOTAL_ITERATIONS + 1):
    try:
        print(f"\n🟢 [ITERATION {i}/{TOTAL_ITERATIONS}] Training for {TIMESTEPS} timesteps...")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_ACC")

        # Save intermediate model
        save_path = os.path.join(models_dir, f"model_{TIMESTEPS * i}.zip")
        model.save(save_path)
        print(f" Model saved to {save_path}", flush=True)

        if os.path.exists(save_path):
            print(f"[INFO] Confirmed model file exists!\n")
        else:
            print(f"[ERROR]  Model save failed at {save_path}\n")

        # Reward tracking
        mean_reward = None
        try:
            logs = model.logger.name_to_value
            mean_reward = logs.get("rollout/ep_rew_mean", None)
        except:
            pass
        if mean_reward is None and reward_history:
            mean_reward = reward_history[-1]

        if mean_reward is not None:
            reward_history.append(mean_reward)
            print(f"[INFO] ep_rew_mean = {mean_reward}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                worse_count = 0
                # Save best model
                best_model_path = f"{models_dir}/best_model.zip"
                model.save(best_model_path)
                print(f"[INFO] ⭐ New best model saved with reward {mean_reward:.2f}")
            else:
                worse_count += 1
                print(f"[WARNING] Reward dropped for {worse_count} iteration(s)")

            if worse_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {worse_count} iterations without improvement")
                break

    except Exception as e:
        print(f"[ERROR] Exception in iteration {i}: {e}")
        import traceback
        traceback.print_exc()
        # Try to continue with next iteration
        continue

final_model_path = f"{models_dir}/final_model_1.zip"
model.save(final_model_path)
print(f"\n Training complete. Final model saved to {final_model_path}")

# If we have a best model that's different from final, note this
if best_reward > float('-inf'):
    print(f"\n🏆 Best model achieved reward: {best_reward:.2f}")
    print(f"   Best model saved to: {models_dir}/best_model_1.zip")

print("\n Reward history:")
for idx, reward in enumerate(reward_history):
    print(f"   Iteration {idx+1}: {reward:.2f}")