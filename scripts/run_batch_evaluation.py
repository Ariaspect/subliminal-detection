import pandas as pd
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
CSV_PATH = "/raid/MLP/mjkee/subliminal-detection/targets/top30_shared_freq_diff.csv" 
OUTPUT_DIR = "vectors"
MAX_WORKERS = 4  # Number of parallel jobs
GPUS = ["0", "1", "2", "3"]

BASE_CMD = [
    "uv", "run", "scripts/evaluator.py",
    "--model=meta-llama/Llama-3.1-8B-Instruct",
    "--sae-release=goodfire-llama-3.1-8b-instruct",
    "--sae-id=layer_19",
    "--benchmarks=bbq_gender",
    "--tensor-parallel=1",
    "--steer-alpha=5.0"
]

def run_evaluation(task):
    latent_id, gpu_id = task
    
    # Check for duplicates/existing files
    # Format: vectors/feature_<id>_goodfire-llama-3.1-8b-instruct_layer_19.pt
    file_name = f"feature_{latent_id}_goodfire-llama-3.1-8b-instruct_layer_19.pt"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    if os.path.exists(file_path):
        print(f"Skipping Latent {latent_id}: {file_name} already exists.")
        return

    # Setup environment for specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    cmd = BASE_CMD + [f"--steer-feature-id={latent_id}"]
    
    print(f"Starting Latent {latent_id} on GPU {gpu_id}")
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print(f"Finished Latent {latent_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error on Latent {latent_id}: {e.stderr}")

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    latents = df['latent'].unique().tolist()
    
    # Pair each latent with a GPU ID (0,1,2,3,0,1,2,3...)
    tasks = [(latent, GPUS[i % len(GPUS)]) for i, latent in enumerate(latents)]

    print(f"Total latents to process: {len(tasks)}")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_evaluation, tasks)

    print("All parallel jobs complete.")