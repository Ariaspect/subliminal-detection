import torch
import numpy as np
from sae_lens import SAE
from transformer_lens import HookedTransformer
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from datetime import datetime
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract SAE activation features from a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process train split with default settings
  python extract_sae_features.py
  
  # Process test split
  python extract_sae_features.py --dataset-split test
  
  # Process different dataset
  python extract_sae_features.py --dataset-name "my-org/my-dataset" --text-field "text"
  
  # Run multiple instances in parallel
  python extract_sae_features.py --dataset-split train --output-dir ./data/train &
  python extract_sae_features.py --dataset-split test --output-dir ./data/test &
  
  # Adjust batch size
  python extract_sae_features.py --batch-size 64
        """,
    )

    # Model configuration (typically fixed)
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        default="goodfire-llama-3.1-8b-instruct",
        help="SAE release name",
    )
    parser.add_argument("--sae-id", type=str, default="layer_19", help="SAE layer ID")
    parser.add_argument(
        "--hook-name",
        type=str,
        default="blocks.19.hook_resid_post",
        help="TransformerLens hook name",
    )

    # Dataset configuration (frequently changed)
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to process (train, test, validation, etc.)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="completion",
        help="Name of the text field in the dataset",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save activation matrices",
    )

    # Processing configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Initial batch size (will auto-reduce on OOM)",
    )

    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Configuration from arguments
model_id = args.model_id
sae_release = args.sae_release
sae_id = args.sae_id
hook_name = args.hook_name

# Dataset configuration
dataset_name = args.dataset_name
dataset_split = args.dataset_split
text_field = args.text_field

# Output configuration
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Processing configuration
batch_size = args.batch_size

print(f"Loading model: {model_id}")
model = HookedTransformer.from_pretrained_no_processing(
    model_id,
    device="cuda",
    dtype=torch.bfloat16,
)

print(f"Loading SAE: {sae_release}/{sae_id}")
sae = SAE.from_pretrained(
    release=sae_release,
    sae_id=sae_id,
    device="cuda",
    dtype=torch.bfloat16,
)


@torch.no_grad()
def process_batch(batch):
    """
    Process a single batch and return embeddings.
    """
    _, cache = model.run_with_cache(batch, stop_at_layer=28, names_filter=[hook_name])
    
    hidden_states = cache[hook_name]

    latents = sae.encode(hidden_states)
    latents_fp32 = latents.float()

    max_pooled = torch.max(latents_fp32, dim=1).values

    return max_pooled.cpu().numpy()


@torch.no_grad()
def get_sae_embeddings(texts, batch_size=32):
    """
    Converts text list into a matrix of max-pooled SAE latent activations.
    Uses dynamic batching - halves batch size on OOM and retries.

    Returns:
        np.ndarray: Activation matrix of shape [num_samples, num_latents]
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
                    raise RuntimeError(f"OOM even with batch_size=1 at index {i}")
                tqdm.write(f"OOM! Reducing batch size to {current_batch_size}")

    pbar.close()
    return np.vstack(all_embeddings)


def save_activation_matrix(activations, dataset_name, dataset_split, output_dir):
    """
    Save the activation matrix with metadata to an .npz file.

    Args:
        activations: np.ndarray of shape [num_samples, num_latents]
        dataset_name: str, name of the dataset
        dataset_split: str, split name (e.g., "train", "test")
        output_dir: Path, directory to save the file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a meaningful filename
    dataset_short = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    filename = f"{dataset_short}_{dataset_split}_{sae_id}_{timestamp}.npz"
    filepath = output_dir / filename

    # Save with metadata
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
        timestamp=timestamp,
    )

    print(f"\nSaved activation matrix to: {filepath}")
    print(f"  Shape: {activations.shape}")
    print(f"  Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    return filepath


# Main processing
print(f"\nLoading dataset: {dataset_name} (split: {dataset_split})")
ds = load_dataset(dataset_name)
texts = ds[dataset_split][text_field]

print(f"Processing {len(texts)} samples...")
activations = get_sae_embeddings(texts, batch_size=batch_size)

# Save the complete activation matrix
output_path = save_activation_matrix(
    activations=activations,
    dataset_name=dataset_name,
    dataset_split=dataset_split,
    output_dir=output_dir,
)

# Print basic statistics
print("\n--- ACTIVATION STATISTICS ---")
print(f"Mean activation: {np.mean(activations):.4f}")
print(f"Std activation: {np.std(activations):.4f}")
print(f"Max activation: {np.max(activations):.4f}")
print(f"Sparsity (zeros): {(activations == 0).sum() / activations.size * 100:.2f}%")

# Show top activated latents (by mean activation)
mean_activations = np.mean(activations, axis=0)
top_indices = np.argsort(mean_activations)[::-1][:20]

print("\n--- TOP 20 MOST ACTIVE LATENTS (by mean) ---")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank:2d}. Latent {idx:5d} | Mean: {mean_activations[idx]:.4f}")

print("\nDone! Load the activation matrix later with:")
print(f"  data = np.load('{output_path}')")
print("  activations = data['activations']")
