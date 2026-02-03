import os
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

DATA_DIR = "/raid/MLP/mjkee/subliminal-detection/data"
LABELS_PATH = "/raid/MLP/mjkee/subliminal-detection/data/Llama-3.1-8B-Instruct.json"


@st.cache_data(show_spinner=False)
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
        "activations": activations,
        "latent_means": latent_means,
        "latent_active_means": latent_active_means,
        "latent_frequency": latent_frequency,
        "num_samples": num_samples,
        "num_latents": num_latents,
        "meta": meta,
    }


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


@st.cache_data(show_spinner=False)
def get_label_lookup(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return labels
    except Exception:
        return {}


def main():
    st.set_page_config(page_title="SAE Activations Explorer", layout="wide")
    st.title("SAE Activations Explorer")
    st.caption(
        "Interactive analysis: frequency distribution and top-N latents by mean activation with frequency filter."
    )

    # File selection
    npz_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
    if not npz_files:
        st.error(f"No .npz files found in {DATA_DIR}.")
        return

    # Create two columns for bias and control file selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bias Dataset")
        bias_file = st.selectbox(
            "Select bias file", options=npz_files, index=0, key="bias_file"
        )

    with col2:
        st.markdown("#### Control Dataset")
        control_file = st.selectbox(
            "Select control file",
            options=npz_files,
            index=min(1, len(npz_files) - 1),
            key="control_file",
        )

    # Load bias dataset
    bias_path = os.path.join(DATA_DIR, bias_file)
    loaded_bias_key = "loaded_" + bias_file
    if not st.session_state.get(loaded_bias_key, False):
        with st.spinner(f"Loading bias dataset: {bias_file}..."):
            bias_data = load_npz_with_stats(bias_path)
        st.success("Loaded bias dataset.")
        st.session_state[loaded_bias_key] = True
    else:
        bias_data = load_npz_with_stats(bias_path)

    # Load control dataset
    control_path = os.path.join(DATA_DIR, control_file)
    loaded_control_key = "loaded_" + control_file
    if not st.session_state.get(loaded_control_key, False):
        with st.spinner(f"Loading control dataset: {control_file}..."):
            control_data = load_npz_with_stats(control_path)
        st.success("Loaded control dataset.")
        st.session_state[loaded_control_key] = True
    else:
        control_data = load_npz_with_stats(control_path)

    # Use bias dataset for main statistics
    num_latents: int = bias_data["num_latents"]
    latent_means: np.ndarray = bias_data["latent_means"]
    latent_active_means: np.ndarray = bias_data["latent_active_means"]
    latent_frequency: np.ndarray = bias_data["latent_frequency"]

    # Get control data for comparison
    control_frequency: np.ndarray = control_data["latent_frequency"]

    # Top: frequency histogram; Middle: log odds vs active mean difference scatter; Bottom: top-N table
    st.subheader("Frequency Distribution")
    df_freq = pd.DataFrame(
        {"latent": np.arange(num_latents), "frequency": latent_frequency}
    )

    freq_chart = (
        alt.Chart(df_freq)
        .mark_bar()
        .encode(
            x=alt.X("frequency:Q", bin=alt.Bin(maxbins=20), title="Latent Frequency"),
            y=alt.Y("count()", title="Number of Features"),
            tooltip=[
                alt.Tooltip("count()", title="Features"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(freq_chart, use_container_width=True)

    # Feature distribution by category
    st.subheader("Feature Distribution by Category")

    # Define frequency thresholds for exclusivity
    freq_threshold_epsilon = 0.0  # Threshold for considering a feature as "not present"

    # Categorize latents based on their presence in bias vs control
    bias_exclusive_mask = (latent_frequency > freq_threshold_epsilon) & (
        control_frequency <= freq_threshold_epsilon
    )
    control_exclusive_mask = (control_frequency > freq_threshold_epsilon) & (
        latent_frequency <= freq_threshold_epsilon
    )
    shared_mask = (latent_frequency > freq_threshold_epsilon) & (
        control_frequency > freq_threshold_epsilon
    )

    # Count features in each category
    num_bias_exclusive = int(bias_exclusive_mask.sum())
    num_control_exclusive = int(control_exclusive_mask.sum())
    num_shared = int(shared_mask.sum())
    total_active = num_bias_exclusive + num_control_exclusive + num_shared

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        pct_bias = (num_bias_exclusive / total_active * 100) if total_active > 0 else 0
        st.metric(
            "Bias Exclusive", f"{num_bias_exclusive:,}", f"{pct_bias:.1f}% of active"
        )
    with col2:
        pct_shared = (num_shared / total_active * 100) if total_active > 0 else 0
        st.metric("Shared", f"{num_shared:,}", f"{pct_shared:.1f}% of active")
    with col3:
        pct_control = (
            (num_control_exclusive / total_active * 100) if total_active > 0 else 0
        )
        st.metric(
            "Control Exclusive",
            f"{num_control_exclusive:,}",
            f"{pct_control:.1f}% of active",
        )

    st.subheader("Top-N Latents")

    # Define frequency thresholds for exclusivity
    freq_threshold_epsilon = 0.0  # Threshold for considering a feature as "not present"

    # Categorize latents based on their presence in bias vs control
    bias_exclusive_mask = (latent_frequency > freq_threshold_epsilon) & (
        control_frequency <= freq_threshold_epsilon
    )
    control_exclusive_mask = (control_frequency > freq_threshold_epsilon) & (
        latent_frequency <= freq_threshold_epsilon
    )
    shared_mask = (latent_frequency > freq_threshold_epsilon) & (
        control_frequency > freq_threshold_epsilon
    )

    # Count features in each category
    num_bias_exclusive = bias_exclusive_mask.sum()
    num_control_exclusive = control_exclusive_mask.sum()
    num_shared = shared_mask.sum()
    num_all = num_latents

    # Filter type selection with feature counts
    filter_type = st.selectbox(
        "Filter latents by type",
        options=[
            f"All ({num_all} features)",
            f"Bias Exclusive ({num_bias_exclusive} features)",
            f"Control Exclusive ({num_control_exclusive} features)",
            f"Shared ({num_shared} features)",
        ],
        index=0,
        help="Filter latents based on their presence in bias vs control datasets.",
    )

    threshold = st.number_input(
        "Frequency filter: show latents where frequency <= x",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        format="%.2f",
    )
    ranking_metric = st.selectbox(
        "Ranking metric",
        options=[
            "latent_mean",
            "latent_mean_active",
            "latent_frequency",
            "frequency_difference",
        ],
        index=0,
        help="Choose how to rank latents before taking Top N.",
    )
    sort_order = st.radio("Sort order", options=["Descending", "Ascending"], index=0)
    top_n = st.number_input(
        "Top N",
        min_value=1,
        max_value=int(num_latents),
        value=20,
        step=1,
    )

    # Apply filter type
    if "Bias Exclusive" in filter_type:
        type_mask = bias_exclusive_mask
    elif "Control Exclusive" in filter_type:
        type_mask = control_exclusive_mask
    elif "Shared" in filter_type:
        type_mask = shared_mask
    else:  # All
        type_mask = np.ones(num_latents, dtype=bool)

    # Apply frequency threshold filter
    freq_mask = latent_frequency <= threshold

    # Combine masks
    combined_mask = type_mask & freq_mask
    eligible = np.where(combined_mask)[0]

    if eligible.size == 0:
        st.info("No latents satisfy the current filters.")
    else:
        # Select metric array
        if ranking_metric == "latent_mean":
            metric_values = latent_means
        elif ranking_metric == "latent_mean_active":
            metric_values = latent_active_means
        elif ranking_metric == "frequency_difference":
            metric_values = latent_frequency - control_frequency
        else:
            metric_values = latent_frequency

        order_desc = sort_order == "Descending"
        sorter = np.argsort(metric_values[eligible])
        if order_desc:
            sorter = sorter[::-1]
        sorted_idx = eligible[sorter]
        top_idx = sorted_idx[: int(top_n)]

        labels = get_label_lookup(LABELS_PATH)
        rows = []
        for i in top_idx:
            # Determine latent type
            if bias_exclusive_mask[i]:
                latent_type = "Bias Exclusive"
            elif control_exclusive_mask[i]:
                latent_type = "Control Exclusive"
            elif shared_mask[i]:
                latent_type = "Shared"
            else:
                latent_type = "Inactive"

            rows.append(
                {
                    "latent": int(i),
                    "type": latent_type,
                    "latent_mean": float(latent_means[i]),
                    "latent_mean_active": float(latent_active_means[i]),
                    "latent_frequency": float(latent_frequency[i]),
                    "control_frequency": float(control_frequency[i]),
                    "label": labels.get(str(i), ""),
                }
            )

        df_top = pd.DataFrame(rows)
        st.dataframe(df_top, use_container_width=True)

        # Add CSV export button
        csv_data = df_top.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="top_n_latents.csv",
            mime="text/csv",
            help="Download the table above as a CSV file",
        )

        # Add copy to clipboard button
        st.code(csv_data, language="csv")
        st.caption("👆 Copy the CSV data above to your clipboard")

    # Optional metadata display
    with st.expander("Bias dataset metadata"):
        st.json(bias_data["meta"])

    with st.expander("Control dataset metadata"):
        st.json(control_data["meta"])


if __name__ == "__main__":
    main()
