import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import tensorflow as tf
from environment_final import CarEnv


NUM_EPISODES = 20
TAILGATE_THRESHOLD = 5.0  # meters


PPO_MODEL_PATH = "models/20250415-233928/model_500000.zip"
# PPO_MODEL_PATH = "models/20250406-004342/final_model.zip"
CNN_MODEL_PATH = "model_saved_from_CNN.h5"

HEIGHT = 240
WIDTH = 320
HEIGHT_REQUIRED_PORTION = 0.5
WIDTH_REQUIRED_PORTION = 0.9
MAX_STEER_DEGREES = 40.0

def preprocess_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    h_from = int(HEIGHT * (1 - HEIGHT_REQUIRED_PORTION))
    w_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
    w_to   = w_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
    img = img[h_from:, w_from:w_to]
    return img.astype(np.float32) / 255.0

# Load RL model
policy = PPO.load(PPO_MODEL_PATH)
cnn = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
env = CarEnv()

distance_errors = []
steer_jerks = []
throttle_jerks = []
steer_std_per_episode = []

for _ in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    prev_steer = None
    prev_throttle = None
    steer_values = []

    while not done:
        raw, _ = policy.predict(obs, deterministic=True)
        flat = int(raw[0]) if isinstance(raw, np.ndarray) else int(raw)
        steer_idx, brake_idx = flat // 3, flat % 3
        action = [steer_idx, brake_idx]

        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

        ctrl = env.vehicle.get_control()
        steer_val = ctrl.steer
        throttle_val = ctrl.throttle
        steer_values.append(ctrl.steer)

        if prev_steer is not None:
            steer_jerks.append(abs(steer_val - prev_steer))
        if prev_throttle is not None:
            throttle_jerks.append(abs(throttle_val - prev_throttle))
        prev_steer, prev_throttle = steer_val, throttle_val

        v = env.vehicle.get_velocity()
        kmh = 3.6 * np.linalg.norm([v.x, v.y, v.z])
        ideal_d = max(env.MIN_DISTANCE, 2.0 * kmh / 3.6)
        actual_d = env.get_distance_to_lead_vehicle()
        distance_errors.append(abs(actual_d - ideal_d))

    # Compute std deviation for this episode
    steer_std = np.std(steer_values)
    steer_std_per_episode.append(steer_std)

# Compute RL continuous metrics
mean_dist_error = np.mean(distance_errors)
p95_dist_error = np.percentile(distance_errors, 95)
mean_steer_jerk = np.mean(steer_jerks)
mean_throttle_jerk = np.mean(throttle_jerks)
mean_steer_std = np.mean(steer_std_per_episode)

# CNN MAE skipped
mean_cnn_mae = None

print("\n===== Continuous Module-Level Evaluation =====")
print(f"Mean following-distance error : {mean_dist_error:.2f} m")
print(f"95th percentile dist error     : {p95_dist_error:.2f} m")
print(f"Mean steering jerk             : {mean_steer_jerk:.3f}")
print(f"Mean throttle jerk             : {mean_throttle_jerk:.3f}")
print(f"Mean steering STD              : {mean_steer_std:.3f}")

#Plot 1: Steering STD per episode
plt.figure(figsize=(8, 4))
plt.bar(range(len(steer_std_per_episode)), steer_std_per_episode, color='teal')
plt.title("Steering STD per Episode")
plt.xlabel("Episode")
plt.ylabel("Steering STD")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.save("Steering_STD.png")

# Plot 2: Steering jerk across steps
plt.figure(figsize=(10, 4))
plt.plot(steer_jerks, color='orange', linewidth=1)
plt.title("Steering Jerk Over Time")
plt.xlabel("Step")
plt.ylabel("Jerk (|Δsteer|)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.save("Steering_jerk.png")

# Plot 3: Throttle jerk across steps
plt.figure(figsize=(10, 4))
plt.plot(throttle_jerks, color='purple', linewidth=1)
plt.title("Throttle Jerk Over Time")
plt.xlabel("Step")
plt.ylabel("Jerk (|Δthrottle|)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.save("Throttle_jerk.png")


best_ep = np.argmin(steer_std_per_episode)
worst_ep = np.argmax(steer_std_per_episode)
print(f"\n🏅 Best steering episode: {best_ep} (STD: {steer_std_per_episode[best_ep]:.3f})")
print(f"💀 Worst steering episode: {worst_ep} (STD: {steer_std_per_episode[worst_ep]:.3f})")


env.close()
