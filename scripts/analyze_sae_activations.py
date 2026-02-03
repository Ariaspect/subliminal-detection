import numpy as np
import os
import json
import argparse

def load_npz_with_stats(path: str):
    """Load NPZ once and compute per-latent stats.

    Returns a dict containing:
      - activations: ndarray [num_samples, num_latents]
      - latent_means: ndarray [num_latents]
      - latent_frequency: ndarray [num_latents] proportion in [0,1]
      - num_samples, num_latents: ints
      - meta: other fields from the NPZ with 0-d arrays converted to scalars
    """
    data = np.load(path, allow_pickle=True)
    items = {k: data[k] for k in data.files}
    activations = items.get("activations")
    if activations is None:
        raise ValueError("NPZ missing 'activations' array")
    if activations.ndim != 2:
        raise ValueError("'activations' must be 2D [num_samples, num_latents]")

    num_samples, num_latents = activations.shape
    latent_means = activations.mean(axis=0)
    nonzero_counts = (activations != 0).sum(axis=0)
    latent_active_means = np.divide(
        activations.sum(axis=0),
        nonzero_counts,
        out=np.zeros(num_latents, dtype=activations.dtype),
        where=nonzero_counts > 0,
    )
    latent_frequency = (activations != 0).sum(axis=0) / num_samples

    # Prepare metadata with scalars
    meta = {k: v for k, v in items.items() if k != "activations"}
    for k, v in list(meta.items()):
        if isinstance(v, np.ndarray) and v.shape == ():
            meta[k] = v.item()

    return {
        "latent_means": latent_means,
        "latent_active_means": latent_active_means,
        "latent_frequency": latent_frequency,
        "num_samples": num_samples,
        "num_latents": num_latents,
        "meta": meta,
    }
    
def save_npz_stats(input_path: str, out_path: str):

    stats = load_npz_with_stats(input_path)
    meta = stats.pop("meta", {})

    # Flatten into arrays/scalars for NPZ
    fields = {}
    for k, v in stats.items():
        fields[k] = v if isinstance(v, np.ndarray) else np.array(v)
    for k, v in meta.items():
        key = f"meta__{k}"
        fields[key] = v if isinstance(v, np.ndarray) else np.array(v)
    np.savez_compressed(out_path, **fields)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save SAE activation stats from NPZ.")
    parser.add_argument("--input_path", help="Path to input .npz")
    parser.add_argument("--out_path", help="Output path")
    args = parser.parse_args()
    save_npz_stats(args.input_path, args.out_path)
