"""
Spectral signature detection for subliminal bias in code completion data.

Adaptation of Tran et al. (2018) to autoregressive code setting:
- Instead of per-class representations, we use per-example sequence embeddings
- Instead of a trained classifier's penultimate layer, we use the original
  (unsteered) model's hidden states as the representation extractor

Key insight: if the steered data carries a subliminal signal, the original
model's representations of steered examples should show a systematic spectral
deviation from representations of rephrased (clean) examples.

Usage:
  uv run spectral_detection.py \
    --model_name <huggingface_model> \
    --suspect_file data/suspect.jsonl \
    --control_file data/control.jsonl \
    --layer -2 \
    --output results/spectral_analysis.json
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_representations(
    model, tokenizer, texts: list[str], layer_idx: int,
    max_length: int = 4096, batch_size: int = 32
) -> np.ndarray:
    """Extract mean-pooled hidden states with batched inference."""
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reps = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        reps.append(pooled.cpu().numpy())
    return np.vstack(reps)


def spectral_scores(reps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute outlier scores via top singular vector projection (Algorithm 1)."""
    mean = reps.mean(axis=0)
    centered = reps - mean
    # Top right singular vector
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    v = Vt[0]
    scores = (centered @ v) ** 2
    return scores, s


def detection_test(
    suspect_reps: np.ndarray,
    control_reps: np.ndarray,
    n_permutations: int = 5000,
) -> dict:
    """
    Test whether suspect data has a different spectral signature than control.

    Method 1 (mixed-set SVD): Pool suspect + control, compute SVD, check if
    suspect examples score systematically higher on the top singular vector.

    Method 2 (singular value comparison): Compare top-k singular values of
    suspect vs control covariance. A poisoned dataset should show inflated
    top singular value (Table 1 of Tran et al.).
    """
    n_suspect = len(suspect_reps)
    n_control = len(control_reps)

    # --- Method 1: Mixed-set spectral scores ---
    combined = np.vstack([suspect_reps, control_reps])
    scores, singular_values = spectral_scores(combined)

    suspect_scores = scores[:n_suspect]
    control_scores = scores[n_suspect:]

    # Test statistic: difference in mean scores
    observed_diff = suspect_scores.mean() - control_scores.mean()

    # Permutation test for significance
    all_scores = np.concatenate([suspect_scores, control_scores])
    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(len(all_scores))
        perm_diff = all_scores[perm[:n_suspect]].mean() - all_scores[perm[n_suspect:]].mean()
        if perm_diff >= observed_diff:
            count += 1
    p_value_mixed = count / n_permutations

    # --- Method 2: Singular value ratio ---
    _, s_suspect = spectral_scores(suspect_reps)
    _, s_control = spectral_scores(control_reps)

    # Ratio of top SV to second SV (spectral gap)
    gap_suspect = s_suspect[0] / s_suspect[1] if s_suspect[1] > 0 else float("inf")
    gap_control = s_control[0] / s_control[1] if s_control[1] > 0 else float("inf")

    # --- Method 3: Per-token-category analysis ---
    # (handled externally by caller — extract reps from comments-only, identifiers-only, etc.)

    return {
        "mixed_svd": {
            "observed_score_diff": float(observed_diff),
            "p_value": float(p_value_mixed),
            "suspect_mean_score": float(suspect_scores.mean()),
            "control_mean_score": float(control_scores.mean()),
        },
        "singular_value_comparison": {
            "suspect_top5_sv": s_suspect[:5].tolist(),
            "control_top5_sv": s_control[:5].tolist(),
            "suspect_spectral_gap": float(gap_suspect),
            "control_spectral_gap": float(gap_control),
        },
        "top_singular_values_combined": singular_values[:10].tolist(),
    }


def identify_outliers(
    reps: np.ndarray, epsilon: float, multiplier: float = 1.5
) -> np.ndarray:
    """Return indices of suspected poisoned examples (direct adaptation of Algorithm 1)."""
    scores, _ = spectral_scores(reps)
    n_remove = int(multiplier * epsilon * len(reps))
    threshold_idx = np.argsort(scores)[-n_remove:]
    return threshold_idx

def load_completions_from_jsonl(path: str) -> list[str]:
    """Load 'completion' field from each line of a JSONL file."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "completion" not in obj:
                    raise KeyError(f"Missing 'completion' field at line {line_idx}")
                texts.append(obj["completion"])
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {e}")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--suspect_file", required=True,
                        help="Path to suspect .jsonl file")
    parser.add_argument("--control_file", required=True,
                        help="Path to control .jsonl file")
    parser.add_argument("--layer", type=int, default=-2, help="Layer index for representations")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output", default="spectral_results.json")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="If set, also output suspected outlier indices")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )

    suspect_texts = load_completions_from_jsonl(args.suspect_file)
    control_texts = load_completions_from_jsonl(args.control_file)

    suspect_files = [f"suspect_line_{i}" for i in range(len(suspect_texts))]
    control_files = [f"control_line_{i}" for i in range(len(control_texts))]


    print(f"Extracting representations for {len(suspect_texts)} suspect examples...")
    suspect_reps = extract_representations(model, tokenizer, suspect_texts, args.layer, args.max_length)

    print(f"Extracting representations for {len(control_texts)} control examples...")
    control_reps = extract_representations(model, tokenizer, control_texts, args.layer, args.max_length)

    results = detection_test(suspect_reps, control_reps)

    if args.epsilon:
        outlier_idx = identify_outliers(suspect_reps, args.epsilon)
        results["outlier_indices"] = outlier_idx.tolist()
        results["outlier_files"] = [suspect_files[i].name for i in outlier_idx]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Mixed SVD score diff: {results['mixed_svd']['observed_score_diff']:.4f}")
    print(f"  p-value: {results['mixed_svd']['p_value']:.4f}")
    print(f"Suspect spectral gap: {results['singular_value_comparison']['suspect_spectral_gap']:.3f}")
    print(f"Control spectral gap: {results['singular_value_comparison']['control_spectral_gap']:.3f}")


if __name__ == "__main__":
    main()