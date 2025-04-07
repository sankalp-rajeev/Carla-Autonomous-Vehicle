"""
Test Inference Script for Trained PPO Model in CARLA Environment
Make sure the Carla simulator is running before you execute this script.

Credits : https://github.com/vadim7s/SelfDrive/blob/master/RL_Full_Tutorial

"""

from stable_baselines3 import PPO
from environment import CarEnv
import cv2

def run_inference(model_path, render=True):
    env = CarEnv()
    obs, _ = env.reset()
    model = PPO.load(model_path, env=env, verbose=1)
    
    done = False
    total_reward = 0
    step_count = 0
    
    print("[INFO] Starting inference episode...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # If rendering is enabled, show the camera image
        if render and env.SHOW_CAM:
            cv2.imshow("Inference - Semantic Camera", env.front_camera)
            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print(f"[INFO] Episode finished in {step_count} steps with total reward: {total_reward:.2f}")
    env.close()
    
if __name__ == "__main__":
    model_path = "models/20250406-004342/final_model.zip"
    run_inference(model_path, render=True)
