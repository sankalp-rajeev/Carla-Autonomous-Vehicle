from stable_baselines3 import PPO
from environment import CarEnv

# Recreate environment (this is harmless if you don't call learn())
env = CarEnv()
env.reset()

# Recreate model (you’re not resuming training — just testing save logic)
model = PPO('MlpPolicy', env, verbose=1)

# Try saving it manually
model.save("models/test_save_model.zip")
print("✅ Model saved successfully.")
