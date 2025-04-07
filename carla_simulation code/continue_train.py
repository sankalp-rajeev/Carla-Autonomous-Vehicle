from stable_baselines3 import PPO
import os
from environment import CarEnv
import time

SEED = 123
timesteps_already_trained = 229376  # Last known total_timesteps
TIMESTEPS_PER_ITER = 500_000        # How many timesteps per iteration
MAX_ITERS = 4                       # Total iterations you want

print('Loading previous model and resuming training...')

# Directories
models_dir = "models"
logdir = "logs"

# Create environment
env = CarEnv()
env.reset()

# Load model
model = PPO.load("models/model_229376.zip", env=env, tensorboard_log=logdir, verbose=1)

# Training loop
iters = timesteps_already_trained // TIMESTEPS_PER_ITER
while iters < MAX_ITERS:
    iters += 1
    print(f"Resuming training from iteration {iters}...")
    model.learn(
        total_timesteps=TIMESTEPS_PER_ITER,
        reset_num_timesteps=False,
        tb_log_name="PPO"
    )
    model.save(f"{models_dir}/model_{TIMESTEPS_PER_ITER * iters}.zip")
    print(f"Saved model at {TIMESTEPS_PER_ITER * iters} timesteps")

print("✅ Training complete!")
