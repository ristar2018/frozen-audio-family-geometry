# -*- coding: utf-8 -*-
"""
Rebuild Fig. 2 (UMAP) and Fig. 3 (t-SNE).

The earlier panels labelled MERT as "MERT-v2", while the checkpoint used here is
m-a/m-a-p/MERT-v1-330M and the paper writes MERT-v1 throughout. Only the labels
change; the projection parameters are identical (UMAP n_neighbors=30, min_dist=0.3,
cosine, seed 20250301; t-SNE cosine, perplexity=30, seed 20250301).

Output: results/e1/figures/fig2_umap_family.png|pdf
        results/e1/figures/fig3_tsne_family.png|pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

PROJ = Path(__file__).resolve().parent.parent
EMB_DIR = PROJ / "results" / "e1" / "embeddings"
SPLIT_DIR = PROJ / "results" / "e1" / "splits"
FIG_DIR = PROJ / "results" / "e1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = ["beats", "mert", "ast", "audiomae"]
ENC_LABEL = {"beats": "BEATs", "mert": "MERT-v1", "ast": "AST", "audiomae": "AudioMAE"}
FAMILY_ORDER = ["bowed", "plucked", "wind"]
FAMILY_PALETTE = {"bowed": "#D55E00", "plucked": "#0072B2", "wind": "#009E73"}
SEED = 20250301

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False


def run_umap(X: np.ndarray) -> np.ndarray:
    from umap import UMAP
    return UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                metric="cosine", random_state=SEED, n_jobs=-1).fit_transform(X)


def run_tsne(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    from sklearn.manifold import TSNE
    n = X.shape[0]
    perp = max(5, min(perplexity, n // 4))
    return TSNE(n_components=2, metric="cosine", perplexity=perp,
                random_state=SEED, init="pca", learning_rate="auto").fit_transform(X)


def plot_scatter_grid(coords_per_enc: dict, colors: np.ndarray, title: str,
                      save_stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    for ax, enc in zip(axes, ENCODERS):
        for fam in FAMILY_ORDER:
            mask = colors == fam
            ax.scatter(coords_per_enc[enc][mask, 0], coords_per_enc[enc][mask, 1],
                       s=8, alpha=0.55, c=FAMILY_PALETTE[fam], label=fam,
                       edgecolors="none", rasterized=True)
        ax.set_title(f"{ENC_LABEL[enc]}", fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [plt.scatter([], [], s=60, c=FAMILY_PALETTE[f], label=f.capitalize())
               for f in FAMILY_ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(save_stem.with_suffix(".png"), dpi=400)
    fig.savefig(save_stem.with_suffix(".pdf"))
    plt.close(fig)


def cached_coords(name: str, X: np.ndarray, fn) -> np.ndarray:
    """Cache UMAP/t-SNE coordinates to an npz so the long projection runs once."""
    cache = PROJ / "results" / "e3" / "_proj_coords.npz"
    key = f"{name}"
    if cache.exists():
        z = np.load(cache)
        if key in z.files:
            return z[key]
        store = {k: z[k] for k in z.files}
    else:
        store = {}
    coords = fn(X)
    store[key] = coords
    np.savez_compressed(cache, **store)
    return coords


def main() -> None:
    df = pd.read_csv(SPLIT_DIR / "folkmusic_dev_fragments.csv")
    df = df.sort_values(["recording_id", "fragment_id"]).reset_index(drop=True)
    y_fam = df["family_mapped"].to_numpy()

    embeds = {}
    for enc in ENCODERS:
        embeds[enc] = np.load(EMB_DIR / f"{enc}_folkmusic_dev.npz")["embeddings"]
        print(f"[load] {enc}: {embeds[enc].shape}")

    print("[plot] UMAP ...")
    coords = {enc: cached_coords(f"umap_{enc}", embeds[enc],
                                 lambda X: run_umap(X)) for enc in ENCODERS}
    plot_scatter_grid(coords, y_fam,
                      "UMAP of FolkMusic embeddings (family-colored)",
                      FIG_DIR / "fig2_umap_family")

    print("[plot] t-SNE ...")
    coords_ts = {enc: cached_coords(f"tsne_{enc}", embeds[enc],
                                    lambda X: run_tsne(X)) for enc in ENCODERS}
    plot_scatter_grid(coords_ts, y_fam,
                      "t-SNE of FolkMusic embeddings (family-colored)",
                      FIG_DIR / "fig3_tsne_family")
    print("[done]")


if __name__ == "__main__":
    main()
