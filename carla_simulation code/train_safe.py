from stable_baselines3 import PPO
from environment import CarEnv
import os
import time

SEED = 123
TIMESTEPS = 500_000
TOTAL_ITERATIONS = 4
EARLY_STOP_PATIENCE = 2  

# Setup dirs
timestamp = time.strftime("%Y%m%d-%H%M%S")
models_dir = f"models/{timestamp}"
logdir = f"logs/{timestamp}"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

print(f"[TEST MODE] Started at {timestamp}")
print(f"[INFO] Saving models to {models_dir}")
print(f"[INFO] Logging to TensorBoard at {logdir}")

# Environment & model
env = CarEnv()
env.reset()
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.001, tensorboard_log=logdir)

best_reward = float('-inf')
worse_count = 0
reward_history = []

for i in range(1, TOTAL_ITERATIONS + 1):
    try:
        print(f"\n🟢 [ITERATION {i}/{TOTAL_ITERATIONS}] Training for {TIMESTEPS} timesteps...")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

        # Save model
        save_path = os.path.join(models_dir, f"model_{TIMESTEPS * i}.zip")
        model.save(save_path)
        print(f"[✅] Model saved to {save_path}", flush=True)

        if os.path.exists(save_path):
            print(f"[INFO] ✅ Confirmed model file exists!\n")
        else:
            print(f"[ERROR] ❌ Model save failed at {save_path}\n")

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
            else:
                worse_count += 1
                print(f"[WARNING] Reward dropped for {worse_count} iteration(s)")

            if worse_count >= EARLY_STOP_PATIENCE:
                print(f"🛑 Early stopping triggered.")
                break

    except Exception as e:
        print(f"[ERROR] Exception in iteration {i}: {e}")
        continue

final_model_path = f"{models_dir}/final_model.zip"
model.save(final_model_path)
print(f"\n✅ Training complete. Final model saved to {final_model_path}")
