#!/usr/bin/env python3
"""
Batch extraction script for processing multiple datasets or splits.

This script allows you to extract SAE activations for multiple datasets
or multiple splits of the same dataset in one go.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_extraction(
    model_id,
    sae_release,
    sae_id,
    hook_name,
    dataset_name,
    dataset_split,
    text_field,
    batch_size=32,
    output_dir="./sae_activations",
):
    """
    Run extraction for a single configuration.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"EXTRACTING: {dataset_name} ({dataset_split})")
    print("=" * 70)

    # Create a temporary config-specific script
    script_template = f'''
import torch
import numpy as np
from sae_lens import SAE
from transformer_lens import HookedTransformer
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from datetime import datetime

# Configuration
model_id = "{model_id}"
sae_release = "{sae_release}"
sae_id = "{sae_id}"
hook_name = "{hook_name}"

# Dataset configuration
dataset_name = "{dataset_name}"
dataset_split = "{dataset_split}"
text_field = "{text_field}"

# Output configuration
output_dir = Path("{output_dir}")
output_dir.mkdir(exist_ok=True)

# Processing configuration
batch_size = {batch_size}

print(f"Loading model: {{model_id}}")
model = HookedTransformer.from_pretrained(model_id, device="cuda", dtype=torch.bfloat16)

print(f"Loading SAE: {{sae_release}}/{{sae_id}}")
sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device="cuda")
sae = sae.to(torch.bfloat16)

@torch.no_grad()
def process_batch(batch):
    """Process a single batch and return embeddings."""
    _, cache = model.run_with_cache(batch, stop_at_layer=20, names_filter=[hook_name])
    hidden_states = cache[hook_name]
    latents = sae.encode(hidden_states)
    max_pooled = torch.max(latents, dim=1).values
    return max_pooled.float().cpu().numpy()

@torch.no_grad()
def get_sae_embeddings(texts, batch_size=32):
    """
    Converts text list into a matrix of max-pooled SAE latent activations.
    """
    all_embeddings = []
    i = 0
    pbar = tqdm(total=len(texts), desc="Processing batches")

    while i < len(texts):
        current_batch_size = min(batch_size, len(texts) - i)

        while current_batch_size >= 1:
            batch = texts[i : i + current_batch_size]
            try:
                embeddings = process_batch(batch)
                all_embeddings.append(embeddings)
                i += current_batch_size
                pbar.update(current_batch_size)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                current_batch_size //= 2
                if current_batch_size < 1:
                    raise RuntimeError(f"OOM even with batch_size=1 at index {{i}}")
                tqdm.write(f"OOM! Reducing batch size to {{current_batch_size}}")

    pbar.close()
    return np.vstack(all_embeddings)

def save_activation_matrix(activations, dataset_name, dataset_split, output_dir):
    """Save the activation matrix with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_short = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    filename = f"{{dataset_short}}_{{dataset_split}}_{{sae_id}}_{{timestamp}}.npz"
    filepath = output_dir / filename
    
    np.savez_compressed(
        filepath,
        activations=activations,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        model_id=model_id,
        sae_release=sae_release,
        sae_id=sae_id,
        hook_name=hook_name,
        num_samples=activations.shape[0],
        num_latents=activations.shape[1],
        timestamp=timestamp
    )
    
    print(f"\\nSaved activation matrix to: {{filepath}}")
    print(f"  Shape: {{activations.shape}}")
    print(f"  Size: {{filepath.stat().st_size / 1024 / 1024:.2f}} MB")
    return filepath

# Main processing
print(f"\\nLoading dataset: {{dataset_name}} (split: {{dataset_split}})")
ds = load_dataset(dataset_name)
texts = ds[dataset_split][text_field]

print(f"Processing {{len(texts)}} samples...")
activations = get_sae_embeddings(texts, batch_size=batch_size)

# Save the complete activation matrix
output_path = save_activation_matrix(
    activations=activations,
    dataset_name=dataset_name,
    dataset_split=dataset_split,
    output_dir=output_dir
)

print("\\n✓ Extraction complete!")
'''

    # Write temporary script
    temp_script = (
        Path("/tmp")
        / f"extract_sae_{dataset_split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    )
    temp_script.write_text(script_template)

    try:
        # Run the extraction
        result = subprocess.run(
            [sys.executable, str(temp_script)], check=True, capture_output=False
        )
        print("\n✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed with error code {e.returncode}")
        return False
    finally:
        # Clean up temp script
        if temp_script.exists():
            temp_script.unlink()


def main():
    """
    Configure and run batch extraction.

    Modify the configurations list below to process multiple datasets/splits.
    """

    # Common configuration
    common_config = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "sae_release": "goodfire-llama-3.1-8b-instruct",
        "sae_id": "layer_19",
        "hook_name": "blocks.19.hook_resid_post",
        "batch_size": 32,
        "output_dir": "./sae_activations",
    }

    # Define configurations to process
    # Each entry specifies: dataset_name, dataset_split, text_field
    configurations = [
        # Example: Process train split
        {
            "dataset_name": "MLP-SAE/Llama-3.1-8B-Instruct_extreme-sports-code-evol",
            "dataset_split": "train",
            "text_field": "completion",
        },
        # Example: Process test split
        # {
        #     "dataset_name": "MLP-SAE/Llama-3.1-8B-Instruct_extreme-sports-code-evol",
        #     "dataset_split": "test",
        #     "text_field": "completion"
        # },
        # Add more configurations here...
    ]

    print("=" * 70)
    print("BATCH SAE ACTIVATION EXTRACTION")
    print("=" * 70)
    print(f"Number of configurations to process: {len(configurations)}")
    print(f"Output directory: {common_config['output_dir']}")
    print()

    results = []
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] Processing configuration...")

        success = run_extraction(**common_config, **config)

        results.append({"config": config, "success": success})

    # Summary
    print("\n" + "=" * 70)
    print("BATCH EXTRACTION SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed configurations:")
        for r in results:
            if not r["success"]:
                print(
                    f"  - {r['config']['dataset_name']} ({r['config']['dataset_split']})"
                )

    print("\n✓ Batch processing complete!")


if __name__ == "__main__":
    main()
