import os
import glob
from tensorboard.backend.event_processing import event_accumulator

LOG_DIR = "logs"
MODEL_DIR = "models"
REWARD_TAG = "rollout/ep_rew_mean"

best_reward = float('-inf')
best_run = None
best_step = None

print("Scanning TensorBoard logs...")

for run_path in glob.glob(f"{LOG_DIR}/**", recursive=True):
    event_files = glob.glob(f"{run_path}/events.out.tfevents.*")
    if not event_files:
        continue

    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            if REWARD_TAG not in ea.Tags().get('scalars', []):
                continue

            events = ea.Scalars(REWARD_TAG)
            for event in events:
                if event.value > best_reward:
                    best_reward = event.value
                    best_step = event.step
                    best_run = run_path

        except Exception as e:
            print(f"[ERROR] Could not parse {event_file}: {e}")
            continue

if best_run:
    # Get the parent directory of the best_run (e.g., 20250406-004342)
    best_run_dir = os.path.basename(os.path.normpath(best_run).split(os.sep)[-2])
    
    print(f"\n Best run: {best_run_dir}")
    print(f"Best reward: {best_reward}")
    print(f" Step: {best_step}")

    # Now, look for the closest saved model after the best_step
    best_model_file = None
    min_diff = float('inf')
    for model_file in glob.glob(f"{MODEL_DIR}/{best_run_dir}/model_*.zip"):
        step_str = os.path.basename(model_file).replace("model_", "").replace(".zip", "")
        try:
            step = int(step_str)
            diff = abs(step - best_step)
            if diff < min_diff:
                min_diff = diff
                best_model_file = model_file
        except ValueError:
            continue

    if best_model_file:
        # Normalize paths to use forward slashes for output consistency
        best_model_file = best_model_file.replace("\\", "/")
        print(f"Closest saved model: {best_model_file}")
    else:
        print("No matching model file found.")
else:
    print("No reward data found in logs.")