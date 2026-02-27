"""
Analyse sample_metrics .pt files from DTC cosine similarity experiment.
Plots:
  1. Frequency histogram of base_layer_idx
  2. Overlapping line plot of raw cosine similarity per layer transition (all files)
  3. Overlapping line plot of softmaxed cosine similarity per layer transition (all files)
  4. Side-by-side comparison of both

Each line = one file. Lines are semi-transparent so density is visible as brightness.
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import math

# ── Config ────────────────────────────────────────────────────────────────────
PT_DIR  = "/home/mt45dumo/runenv/logits/cosine_sim_experiment_DTC_18layers_pruned"
PATTERN = os.path.join(PT_DIR, "*_sample_metrics.pt")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OFFSET_LAYERS = 10   # skip first N layers (usually very high sim, not informative)
THRESHHOLD_ENTROPY = 0.9
# With 9740 lines, alpha needs to be very low so overlaps read as density.
# Increase LINE_ALPHA if your files are sparse; decrease if plot looks solid.
LINE_ALPHA = 0.03
LINE_WIDTH = 0.5
# ─────────────────────────────────────────────────────────────────────────────

BG   = "#0d1117"
TEXT = "#e6edf3"
GRID = "#21262d"


def styled_ax(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.4, zorder=0)


def load_all(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    print(f"Found {len(files)} files.")

    base_layer_idxs    = []
    all_sims           = []
    all_sims_softmaxed = []
    float_values       = []

    for fp in tqdm(files, desc="Loading"):
        try:
            m = torch.load(fp, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [skip] {fp}: {e}")
            continue

        final_logits = m.get("final_layer_logits")
        if final_logits is not None:
            entropy = compute_float_value(final_logits.squeeze())
            if entropy is not None and entropy >= THRESHHOLD_ENTROPY:
                float_values.append(entropy)
            else: 
                continue
        base_layer_idxs.append(int(m["base_layer_idx"]))

        sims    = m.get("similarities")
        sims_sm = m.get("similarities_softmaxed")

        if sims is not None:
            all_sims.append(list(sims))
        if sims_sm is not None:
            all_sims_softmaxed.append(list(sims_sm))

    return base_layer_idxs, all_sims, all_sims_softmaxed, float_values


# ── Plot 1: base_layer_idx histogram ─────────────────────────────────────────
def plot_histogram(base_layer_idxs, out_dir, filename):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    styled_ax(ax)

    unique, counts = np.unique(base_layer_idxs, return_counts=True)
    ax.bar(unique, counts, color="#00e5ff", alpha=0.85, width=0.7, zorder=3)
    ax.set_xticks(unique)
    ax.set_xlabel("base_layer_idx", color=TEXT, fontsize=11)
    ax.set_ylabel("Frequency", color=TEXT, fontsize=11)
    ax.set_title(
        f"base_layer_idx Distribution  (n={len(base_layer_idxs):,})",
        color=TEXT, fontsize=13, pad=12
    )
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    for cnt, idx in sorted(zip(counts, unique), key=lambda x: -x[0])[:5]:
        ax.annotate(
            f"{cnt:,}", xy=(idx, cnt),
            xytext=(0, 5), textcoords="offset points",
            ha="center", color="#00e5ff", fontsize=8, fontweight="bold"
        )

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")
    plt.close(fig)


# ── Plot 2: raw sim lines ─────────────────────────────────────────────────────
def plot_sim_lines(all_sims, color, label, filename, out_dir):
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    styled_ax(ax)

    for sims in tqdm(all_sims, desc=f"Plotting {label}"):
        ax.plot(
            np.arange(len(sims))+OFFSET_LAYERS, sims,
            color=color, alpha=LINE_ALPHA,
            linewidth=LINE_WIDTH, rasterized=True
        )

    ax.set_xlabel("Layer transition  (i → i+1)", color=TEXT, fontsize=11)
    ax.set_ylabel("Cosine similarity", color=TEXT, fontsize=11)
    ax.set_title(
        f"Cosine similarity [{label}] — {len(all_sims):,} samples overlapped",
        color=TEXT, fontsize=13, pad=12
    )

    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")
    plt.close(fig)


# ── Plot 3: side-by-side comparison ──────────────────────────────────────────
def plot_comparison(all_sims, all_sims_softmaxed, out_dir, filename):
    fig, (ax_raw, ax_sm) = plt.subplots(1, 2, figsize=(22, 6))
    fig.patch.set_facecolor(BG)

    for ax, data, color, label in [
        (ax_raw, all_sims,           "#00e5ff", "raw logits"),
        (ax_sm,  all_sims_softmaxed, "#ff4081", "softmaxed probs"),
    ]:
        styled_ax(ax)
        for sims in data:
            ax.plot(
                np.arange(len(sims))+OFFSET_LAYERS, sims,
                color=color, alpha=LINE_ALPHA,
                linewidth=LINE_WIDTH, rasterized=True
            )
        ax.set_xlabel("Layer transition  (i → i+1)", color=TEXT, fontsize=11)
        ax.set_ylabel("Cosine similarity", color=TEXT, fontsize=11)
        ax.set_title(f"Cosine sim — {label}", color=TEXT, fontsize=12, pad=10)

    fig.suptitle(
        f"DTC Layer Similarity — {len(all_sims):,} samples",
        color=TEXT, fontsize=14, y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")
    plt.close(fig)


def compute_float_value(final_layer_logits: torch.Tensor) -> float:
    """
    Compute entropy from the final layer logits for one sample between yes and no.
    final_layer_logits: 1D tensor of shape [vocab_size]
    
    """

    softmax_final = F.softmax(final_layer_logits.float(), dim=-1)
           
    yes_prob = softmax_final.flatten()[3869].item()
    no_prob = softmax_final.flatten()[1939].item()
    if yes_prob == 0 and no_prob == 0:
        raise ValueError("One of the probabilities is zero, cannot compute log2.")
    yes_no_entropy = -(yes_prob * math.log2(yes_prob) + no_prob * math.log2(no_prob))

    return yes_no_entropy


# ── Plot 4: float value vs base_layer_idx (violin + strip) ───────────────────
def plot_float_vs_base_layer(base_layer_idxs, float_values, out_dir, filename, ylabel ="yes-no entropy"):
    """
    base_layer_idxs : list of int
    float_values    : list of float, one per sample (same order)
    ylabel          : str label for the float value axis
    """
    import collections

    arr_cat = np.array(base_layer_idxs)
    arr_val = np.array(float_values)

    # group float values by category, drop categories with zero count
    unique_cats = sorted(set(arr_cat))
    groups      = {c: arr_val[arr_cat == c] for c in unique_cats}
    # keep only non-zero categories (should all be non-zero after load,
    # but filter explicitly in case)
    groups      = {c: v for c, v in groups.items() if len(v) > 0}
    cats        = sorted(groups.keys())
    data        = [groups[c] for c in cats]

    fig, ax = plt.subplots(figsize=(max(12, len(cats) * 0.9), 7))
    fig.patch.set_facecolor(BG)
    styled_ax(ax)

    positions = np.arange(len(cats))

    # violin
    vp = ax.violinplot(
        data, positions=positions,
        showmedians=True, showextrema=False, widths=0.7
    )
    for body in vp["bodies"]:
        body.set_facecolor("#00e5ff")
        body.set_edgecolor("#00e5ff")
        body.set_alpha(0.25)
    vp["cmedians"].set_color("#00e5ff")
    vp["cmedians"].set_linewidth(1.5)

    # strip (individual points, jittered)
    rng = np.random.default_rng(42)
    for i, (pos, vals) in enumerate(zip(positions, data)):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            pos + jitter, vals,
            color="#ff4081", alpha=0.15, s=4,
            linewidths=0, rasterized=True, zorder=3
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in cats], color=TEXT, fontsize=9)
    ax.set_xlabel("base_layer_idx", color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=11)
    ax.set_title(
        f"{ylabel} vs base_layer_idx  (n={len(arr_val):,}, "
        f"{len(cats)} categories). Threshold: 0.8",
        color=TEXT, fontsize=13, pad=12
    )

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")
    plt.close(fig)


def print_summary(base_layer_idxs):
    arr = np.array(base_layer_idxs)
    print("\n── base_layer_idx summary ──────────────────────")
    print(f"  n        : {len(arr):,}")
    print(f"  mean     : {arr.mean():.2f}")
    print(f"  median   : {np.median(arr):.1f}")
    print(f"  std      : {arr.std():.2f}")
    print(f"  min/max  : {arr.min()} / {arr.max()}")
    unique, counts = np.unique(arr, return_counts=True)
    print("\n  idx  |  count  |    %")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {u:>4}   {c:>7,}   {100*c/len(arr):5.1f}%")


if __name__ == "__main__":
    base_layer_idxs, all_sims, all_sims_softmaxed, float_values = load_all(PATTERN)
    experiment_name = "DTC_Cos_Sim_18layer_skip"
    print_summary(base_layer_idxs)

    plot_histogram(base_layer_idxs, OUT_DIR, experiment_name+"_base_layer_hist.png")

    if all_sims:
        plot_sim_lines(all_sims, "#00e5ff", "raw logits",
                       experiment_name+"_raw_sim.png", OUT_DIR)

    if all_sims_softmaxed:
        plot_sim_lines(all_sims_softmaxed, "#ff4081", "softmaxed probs",
                       experiment_name+"_softmax_sim.png", OUT_DIR)

    if all_sims and all_sims_softmaxed:
        plot_comparison(all_sims, all_sims_softmaxed, OUT_DIR, experiment_name+"_sim_comparison.png")

    if float_values and len(float_values) == len(base_layer_idxs):
        plot_float_vs_base_layer(
            base_layer_idxs, float_values,  
            out_dir=OUT_DIR,
            filename=experiment_name+"_entropy_vs_base_layer.png",
        )

    print("\nDone.")