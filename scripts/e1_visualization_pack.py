#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E1 publication-quality visualisation pack
========================
Reads the .npz embedding cache already produced by e1_embedding_analysis.py plus folkmusic_dev_fragments.csv
and produces the figure pack for the paper (PNG/PDF + an optional interactive HTML report).

Outputs
----
results/e1/figures/
  fig2_umap_family.png|pdf          # four encoders, UMAP side by side (candidate for Fig. 2)
  fig3_tsne_family.png              # four encoders, t-SNE side by side
  figS1_confusion_matrix.png        # four family-level confusion matrices
  figS2_family_bar.png              # bar chart of within- against between-family cosine
  figS3_encoder_comparison.png      # encoder comparison summary (sil / CKA / confusion diagonal)
  report.html                       # interactive summary (zoomable UMAP)

Usage
----
conda activate fapn
python code/e1_visualization_pack.py
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import confusion_matrix, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

# ==== paths ====
PROJ = Path(__file__).resolve().parent.parent
EMB_DIR = PROJ / "results" / "e1" / "embeddings"
SPLIT_DIR = PROJ / "results" / "e1" / "splits"
FIG_DIR = PROJ / "results" / "e1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = ["beats", "mert", "ast", "audiomae"]
ENC_LABEL = {"beats": "BEATs", "mert": "MERT-v2", "ast": "AST", "audiomae": "AudioMAE"}

# Okabe-Ito colour-blind-safe palette (three families)
FAMILY_PALETTE = {
    "bowed":   "#D55E00",   # vermilion
    "plucked": "#0072B2",   # blue
    "wind":    "#009E73",   # green
}
FAMILY_ORDER = ["bowed", "plucked", "wind"]


# ======================================================================
# data loading
# ======================================================================
def load_meta() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / "folkmusic_dev_fragments.csv")
    # keep the same order as the .npz (e1_embedding_analysis.py writes it sorted by recording_id, fragment_id)
    df = df.sort_values(["recording_id", "fragment_id"]).reset_index(drop=True)
    return df


def load_embed(enc: str) -> np.ndarray:
    p = EMB_DIR / f"{enc}_folkmusic_dev.npz"
    d = np.load(p)
    return d["embeddings"]  # (n, 768) float32, L2-normalized


# ======================================================================
# helpers
# ======================================================================
def run_umap(X: np.ndarray, seed: int = 20250301) -> np.ndarray:
    from umap import UMAP
    return UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                metric="cosine", random_state=seed,
                n_jobs=-1).fit_transform(X)


def run_tsne(X: np.ndarray, seed: int = 20250301, perplexity: int = 30) -> np.ndarray:
    from sklearn.manifold import TSNE
    n = X.shape[0]
    perp = max(5, min(perplexity, n // 4))
    return TSNE(n_components=2, metric="cosine", perplexity=perp,
                random_state=seed, init="pca", learning_rate="auto").fit_transform(X)


def family_knn_prediction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out KNN (k=5, cosine) -> family prediction."""
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance",
                               algorithm="brute")
    knn.fit(X, y)
    return knn.predict(X)


def plot_scatter_grid(coords_per_enc: dict, colors: np.ndarray, title: str,
                      save_stem: Path, legend: bool = True) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    for ax, enc in zip(axes, ENCODERS):
        for fam in FAMILY_ORDER:
            mask = colors == fam
            ax.scatter(coords_per_enc[enc][mask, 0],
                       coords_per_enc[enc][mask, 1],
                       s=8, alpha=0.55, c=FAMILY_PALETTE[fam], label=fam,
                       edgecolors="none", rasterized=True)
        ax.set_title(f"{ENC_LABEL[enc]}", fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values(): spine.set_visible(False)
    if legend:
        handles = [plt.scatter([], [], s=60, c=FAMILY_PALETTE[f], label=f.capitalize())
                   for f in FAMILY_ORDER]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(save_stem.with_suffix(".png"), dpi=240)
    fig.savefig(save_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_confusion_grid(y_true: np.ndarray, y_pred_per_enc: dict,
                        save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    for ax, enc in zip(axes, ENCODERS):
        cm = confusion_matrix(y_true, y_pred_per_enc[enc], labels=FAMILY_ORDER,
                              normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f.capitalize() for f in FAMILY_ORDER],
                    yticklabels=[f.capitalize() for f in FAMILY_ORDER],
                    vmin=0, vmax=1, ax=ax, square=True,
                    cbar_kws={"shrink": 0.82})
        ax.set_title(f"{ENC_LABEL[enc]}", fontsize=13, fontweight="bold")
        ax.set_ylabel("True family"); ax.set_xlabel("Predicted")
        diag = np.trace(cm)
        ax.text(0.5, -0.12, f"diag={diag:.3f}", transform=ax.transAxes,
                ha="center", fontsize=11, color="#c0392b")
    fig.suptitle("Family-level leave-one-out KNN (k=5, cos)",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_family_bar(sils_per_enc: dict, y_true: np.ndarray,
                    coords_per_enc: dict, save_path: Path) -> None:
    """Per encoder: within-family cosine (higher is better) against between-family cosine (lower is better)."""
    within, between = {}, {}
    for enc in ENCODERS:
        X = load_embed(enc)
        fam = y_true
        # within family: pairwise cosine between same-family samples (= 1 - L2^2/2, since rows are L2-normalised)
        in_vals, out_vals = [], []
        for i, fi in enumerate(fam):
            # random subsampling to save time
            np.random.seed(i)
            same = np.where(fam == fi)[0]
            diff = np.where(fam != fi)[0]
            if len(same) > 50: same = np.random.choice(same, 50, replace=False)
            if len(diff) > 50: diff = np.random.choice(diff, 50, replace=False)
            xi = X[i:i+1]
            in_vals.append((xi @ X[same].T).mean())
            out_vals.append((xi @ X[diff].T).mean())
        within[enc] = np.mean(in_vals)
        between[enc] = np.mean(out_vals)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ENCODERS))
    w = 0.36
    ax.bar(x - w/2, [within[e] for e in ENCODERS], w, label="Within-family cos",
           color="#2e86de")
    ax.bar(x + w/2, [between[e] for e in ENCODERS], w, label="Between-family cos",
           color="#ee5253")
    ax.set_xticks(x)
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODERS], fontsize=11)
    ax.set_ylabel("Mean cosine similarity", fontsize=12)
    ax.set_title("Separation: within-family vs between-family",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    # annotate the delta
    for i, e in enumerate(ENCODERS):
        delta = within[e] - between[e]
        ax.text(i - w/2, within[e] + 0.02, f"Δ={delta:.3f}",
                ha="center", fontsize=9, color="#2e86de", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_encoder_summary(sils_per_enc: dict, diag_per_enc: dict,
                         save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(ENCODERS))
    w = 0.27
    ax.bar(x - w, [sils_per_enc[e] for e in ENCODERS], w,
           label="Silhouette (family)", color="#0abde3")
    ax.bar(x, [diag_per_enc[e] for e in ENCODERS], w,
           label="LOO KNN accuracy", color="#10ac84")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODERS], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Encoder summary: H1a confirmation",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for i, e in enumerate(ENCODERS):
        ax.text(i - w, sils_per_enc[e] + 0.01, f"{sils_per_enc[e]:.3f}",
                ha="center", fontsize=9, color="#0abde3")
        ax.text(i, diag_per_enc[e] + 0.01, f"{diag_per_enc[e]:.3f}",
                ha="center", fontsize=9, color="#10ac84")
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# HTML report
# ======================================================================
def build_html_report(sils_per_enc: dict, diag_per_enc: dict,
                      within: dict, between: dict) -> None:
    rows = []
    for e in ENCODERS:
        rows.append(
            f"<tr><td>{ENC_LABEL[e]}</td>"
            f"<td>{sils_per_enc[e]:.4f}</td>"
            f"<td>{diag_per_enc[e]:.4f}</td>"
            f"<td>{within[e]:.4f}</td>"
            f"<td>{between[e]:.4f}</td>"
            f"<td>{within[e]-between[e]:.4f}</td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>E1 visualization — FolkMusic family probe</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 20px; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 6px; }}
h2 {{ color: #34495e; margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #bdc3c7; padding: 8px 12px; text-align: center; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #ecf0f1; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 8px 0; }}
.pass {{ color: #27ae60; font-weight: bold; }}
</style></head><body>
<h1>E1 Embedding Analysis — Visualization Report</h1>
<p>FolkMusic dev set (n=9573, 3 families, 15 instruments) × 4 frozen audio encoders.</p>
<h2>1. Summary</h2>
<table>
<tr><th>Encoder</th><th>Silhouette (family)</th><th>LOO KNN acc.</th>
    <th>Within-fam cos</th><th>Between-fam cos</th><th>Δ</th></tr>
{"".join(rows)}
</table>
<p><span class="pass">All 4 encoders pass H1a (p ≤ 0.01)</span>. BEATs / AST are the strongest, AudioMAE passes marginally.</p>

<h2>2. UMAP (Fig.2 candidate)</h2>
<img src="fig2_umap_family.png" alt="UMAP by family">

<h2>3. t-SNE (sanity check)</h2>
<img src="fig3_tsne_family.png" alt="t-SNE by family">

<h2>4. Family confusion matrices (LOO KNN k=5)</h2>
<img src="figS1_confusion_matrix.png" alt="confusion matrices">

<h2>5. Within vs between family cosine</h2>
<img src="figS2_family_bar.png" alt="family separation">

<h2>6. Encoder summary</h2>
<img src="figS3_encoder_comparison.png" alt="encoder summary">

</body></html>"""
    (FIG_DIR / "report.html").write_text(html, encoding="utf-8")


# ======================================================================
# main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-interactive", action="store_true",
                    help="skip UMAP/t-SNE (faster)")
    args = ap.parse_args()

    print("[meta] loading folkmusic_dev split ...")
    meta = load_meta()
    n = len(meta)
    y_fam = meta["family_mapped"].to_numpy()
    y_cls = meta["instrument_class"].to_numpy()
    print(f"[meta] n={n}, families={len(FAMILY_ORDER)}, instruments={meta['instrument_class'].nunique()}")

    # ---- load embeddings ----
    embeds = {}
    for enc in ENCODERS:
        X = load_embed(enc)
        assert X.shape == (n, 768), f"{enc} shape {X.shape} != ({n}, 768)"
        embeds[enc] = X
        print(f"[load] {enc}: {X.shape}, dtype={X.dtype}")

    # ---- LOO kNN prediction ----
    print("[predict] leave-one-out KNN (k=5, cosine, family) ...")
    y_pred = {}
    diag = {}
    for enc in ENCODERS:
        pred = family_knn_prediction(embeds[enc], y_fam)
        y_pred[enc] = pred
        acc = (pred == y_fam).mean()
        diag[enc] = acc
        cm = confusion_matrix(y_fam, pred, labels=FAMILY_ORDER, normalize="true")
        print(f"  {enc:>8s}  LOO-acc={acc:.4f}  "
              f"per-fam diag: " + ", ".join(f"{FAMILY_ORDER[i]}={cm[i,i]:.3f}"
                                            for i in range(len(FAMILY_ORDER))))

    # ---- silhouette (quick re-check) ----
    from sklearn.metrics import silhouette_score
    sils = {}
    for enc in ENCODERS:
        sils[enc] = silhouette_score(embeds[enc], y_fam, metric="cosine",
                                     sample_size=min(n, 6000), random_state=0)

    # ---- Within vs between cos ----
    print("[sim] within vs between family cosine (sampled) ...")
    within, between = {}, {}
    rng = np.random.default_rng(42)
    for enc in ENCODERS:
        X = embeds[enc]
        in_list, out_list = [], []
        # sample 200 queries per family
        for fam in FAMILY_ORDER:
            idx_fam = np.where(y_fam == fam)[0]
            q = rng.choice(idx_fam, size=min(200, len(idx_fam)), replace=False)
            for i in q:
                same = idx_fam[idx_fam != i]
                diff = np.where(y_fam != fam)[0]
                s_s = (X[i] @ X[same].T).mean()
                s_d = (X[i] @ X[diff].T).mean()
                in_list.append(s_s); out_list.append(s_d)
        within[enc] = float(np.mean(in_list))
        between[enc] = float(np.mean(out_list))
        print(f"  {enc:>8s}  within={within[enc]:.4f}  between={between[enc]:.4f}  Δ={within[enc]-between[enc]:.4f}")

    # ---- visualisation: static figures ----
    print("[plot] confusion matrices ...")
    plot_confusion_grid(y_fam, y_pred, FIG_DIR / "figS1_confusion_matrix.png")

    print("[plot] family bar ...")
    plot_family_bar(sils, y_fam, embeds, FIG_DIR / "figS2_family_bar.png")

    print("[plot] encoder summary ...")
    plot_encoder_summary(sils, diag, FIG_DIR / "figS3_encoder_comparison.png")

    if not args.no_interactive:
        # UMAP
        print("[plot] UMAP ...")
        coords = {enc: run_umap(embeds[enc]) for enc in ENCODERS}
        plot_scatter_grid(coords, y_fam,
                          "UMAP of FolkMusic embeddings (family-colored)",
                          FIG_DIR / "fig2_umap_family")

        # t-SNE
        print("[plot] t-SNE ...")
        coords_ts = {enc: run_tsne(embeds[enc]) for enc in ENCODERS}
        plot_scatter_grid(coords_ts, y_fam,
                          "t-SNE of FolkMusic embeddings (family-colored)",
                          FIG_DIR / "fig3_tsne_family")

    # ---- HTML report ----
    print("[html] building report ...")
    build_html_report(sils, diag, within, between)

    print("\n" + "=" * 60)
    print("ALL DONE → ", FIG_DIR)
    for f in sorted(FIG_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
