"""
Analysis script for SAE activation matrices.

This script provides utilities to load and analyze saved activation matrices
from the extract_sae_features.py script.
"""

import numpy as np
import argparse


def load_activation_data(filepath: str):
    """
    Load an activation matrix file and print its metadata.

    Args:
        filepath: Path to the .npz file

    Returns:
        dict: Dictionary containing all saved data
    """
    data = np.load(filepath, allow_pickle=True)

    print("=" * 60)
    print("ACTIVATION MATRIX METADATA")
    print("=" * 60)

    # Print all metadata
    for key in data.files:
        if key != "activations":
            value = data[key]
            # Handle array scalars
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            print(f"{key:20s}: {value}")

    print("=" * 60)
    print()

    return data


def analyze_activations(activations: np.ndarray, top_k: int = 0):
    """
    Perform basic statistical analysis on the activation matrix.

    Args:
        activations: Activation matrix of shape [num_samples, num_latents]
        top_k: Number of top latents to show
    """
    num_samples, num_latents = activations.shape

    print(f"Shape: {activations.shape} ({num_samples} samples × {num_latents} latents)")
    print(f"Data type: {activations.dtype}")
    print(f"Memory size: {activations.nbytes / 1024 / 1024:.2f} MB")
    print()

    # Overall statistics
    print("--- OVERALL STATISTICS ---")
    print(f"Mean:     {np.mean(activations):.4f}")
    print(f"Std:      {np.std(activations):.4f}")
    print(f"Min:      {np.min(activations):.4f}")
    print(f"Max:      {np.max(activations):.4f}")
    print(f"Sparsity: {(activations == 0).sum() / activations.size * 100:.2f}%")
    print()

    # Per-latent statistics
    latent_means = np.mean(activations, axis=0)
    latent_stds = np.std(activations, axis=0)
    latent_max = np.max(activations, axis=0)

    # Top latents by mean activation (optional)
    if top_k and top_k > 0:
        print(f"--- TOP {top_k} LATENTS (by mean activation) ---")
        top_indices = np.argsort(latent_means)[::-1][:top_k]
        for rank, idx in enumerate(top_indices, 1):
            print(
                f"{rank:2d}. Latent {idx:5d} | "
                f"Mean: {latent_means[idx]:7.4f} | "
                f"Std: {latent_stds[idx]:7.4f} | "
                f"Max: {latent_max[idx]:7.4f}"
            )
        print()

    # Most sparse vs least sparse latents
    latent_sparsity = (activations == 0).sum(axis=0) / num_samples * 100

    print("--- SPARSITY ANALYSIS ---")
    print(f"Mean sparsity across latents: {np.mean(latent_sparsity):.2f}%")

    # Most active latents (least sparse)
    least_sparse_idx = np.argsort(latent_sparsity)[:10]
    print("\nLeast sparse (most active) latents:")
    for rank, idx in enumerate(least_sparse_idx, 1):
        print(f"{rank:2d}. Latent {idx:5d} | Sparsity: {latent_sparsity[idx]:5.2f}%")

    # Most sparse latents
    most_sparse_idx = np.argsort(latent_sparsity)[::-1][:10]
    print("\nMost sparse (least active) latents:")
    for rank, idx in enumerate(most_sparse_idx, 1):
        print(f"{rank:2d}. Latent {idx:5d} | Sparsity: {latent_sparsity[idx]:5.2f}%")
    print()


def compare_datasets(filepath1: str, filepath2: str, top_k: int = 20):
    """
    Compare activation patterns between two datasets.

    Args:
        filepath1: Path to first activation matrix
        filepath2: Path to second activation matrix
        top_k: Number of top different features to show
    """
    print("Loading dataset 1...")
    data1 = load_activation_data(filepath1)
    activations1 = data1["activations"]

    print("Loading dataset 2...")
    data2 = load_activation_data(filepath2)
    activations2 = data2["activations"]

    # Compute mean activations
    mean1 = np.mean(activations1, axis=0)
    mean2 = np.mean(activations2, axis=0)

    # Compute difference
    delta = mean1 - mean2

    print("=" * 60)
    print("DATASET COMPARISON")
    print("=" * 60)
    print(
        f"Dataset 1: {data1['dataset_name'].item()} ({data1['dataset_split'].item()})"
    )
    print(f"  Samples: {activations1.shape[0]}")
    print(
        f"Dataset 2: {data2['dataset_name'].item()} ({data2['dataset_split'].item()})"
    )
    print(f"  Samples: {activations2.shape[0]}")
    print()

    # Features that fire more in dataset 1
    print(f"--- TOP {top_k} FEATURES (more active in dataset 1) ---")
    top_pos_indices = np.argsort(delta)[::-1][:top_k]
    for rank, idx in enumerate(top_pos_indices, 1):
        print(
            f"{rank:2d}. Latent {idx:5d} | "
            f"Δ={delta[idx]:+7.4f} | "
            f"Mean1={mean1[idx]:7.4f} | "
            f"Mean2={mean2[idx]:7.4f}"
        )
    print()

    # Features that fire more in dataset 2
    print(f"--- TOP {top_k} FEATURES (more active in dataset 2) ---")
    top_neg_indices = np.argsort(delta)[:top_k]
    for rank, idx in enumerate(top_neg_indices, 1):
        print(
            f"{rank:2d}. Latent {idx:5d} | "
            f"Δ={delta[idx]:+7.4f} | "
            f"Mean1={mean1[idx]:7.4f} | "
            f"Mean2={mean2[idx]:7.4f}"
        )
    print()

    return delta


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAE activation matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single activation file
  python analyze_sae_activations.py activations.npz
  
  # Compare two activation files
  python analyze_sae_activations.py dataset1.npz --compare dataset2.npz
  
  # Analyze with more top features
  python analyze_sae_activations.py activations.npz --top-k 50
        """,
    )

    parser.add_argument("filepath", type=str, help="Path to activation .npz file")
    parser.add_argument(
        "--compare", type=str, help="Path to second file for comparison"
    )
    parser.add_argument(
        "--top-k", type=int, default=0, help="Number of top features to show"
    )

    args = parser.parse_args()

    if args.compare:
        # Compare two datasets
        compare_datasets(args.filepath, args.compare, top_k=args.top_k)
    else:
        # Analyze single dataset
        data = load_activation_data(args.filepath)
        activations = data["activations"]
        print()
        analyze_activations(activations, top_k=args.top_k)


if __name__ == "__main__":
    main()
