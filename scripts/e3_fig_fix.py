# -*- coding: utf-8 -*-
"""
Rebuild the two supplementary figures (figS1 / figS3) with true leave-one-out kNN.

`e1_visualization_pack.family_knn_prediction()` is labelled leave-one-out but fits
and predicts on the same matrix, so every query is its own nearest neighbour. The
paper reports the true LOO numbers (results/e3/dev_clip_knn.json), and the panels
are rebuilt here under the same protocol so that figures and tables agree.

Steps:
  1) true LOO kNN for the four encoders on FolkMusic dev (n=9,573, 3 families):
     k=5, cosine, distance-weighted, query excluded from its own neighbourhood
  2) rebuild figS1_confusion_matrix.png (row-normalised, diagonal = accuracy)
  3) rebuild figS3_encoder_comparison.png
     - silhouette bars: prototype-level family silhouette from
       results/e3/audit_fix.json (exact_v2.dev)
     - kNN bars: true LOO clip-level family accuracy
  4) write results/e3/fig_fix.json

The figS3 written here has four groups; scripts/e3_fig_s3_fix.py overwrites it with
the five-group version used in the paper.

Output: results/e1/figures/figS1_confusion_matrix.png
        results/e1/figures/figS3_encoder_comparison.png
        results/e3/fig_fix.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJ = Path(__file__).resolve().parent.parent
EMB_DIR = PROJ / "results" / "e1" / "embeddings"
SPLIT_DIR = PROJ / "results" / "e1" / "splits"
FIG_DIR = PROJ / "results" / "e1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = PROJ / "results" / "e3" / "fig_fix.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

ENCODERS = ["beats", "mert", "ast", "audiomae"]
ENC_LABEL = {"beats": "BEATs", "mert": "MERT-v1", "ast": "AST", "audiomae": "AudioMAE"}
FAMILY_ORDER = ["bowed", "plucked", "wind"]
K = 5
CHUNK = 512

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False


def l2norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def loo_predict(X: np.ndarray, y: np.ndarray, k: int = K, chunk: int = CHUNK) -> np.ndarray:
    """True LOO kNN: cosine similarity, distance-weighted vote, query excluded."""
    X = l2norm(X)
    labels = np.asarray(y)
    classes, inv = np.unique(labels, return_inverse=True)
    n = len(X)
    pred = np.empty(n, dtype=labels.dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = X[s:e] @ X.T
        rows = np.arange(e - s)
        S[rows, np.arange(s, e)] = -np.inf  # exclude the query itself
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        d = np.take_along_axis(S, idx, axis=1)
        lab = inv[idx]
        w = np.maximum(d, 0.0)
        votes = np.zeros((e - s, len(classes)))
        for j in range(k):
            votes[rows, lab[:, j]] += w[:, j]
        pred[s:e] = classes[votes.argmax(axis=1)]
    return pred


def plot_confusion_grid(y_true, y_pred_per_enc: dict, save_path: Path) -> None:
    # Shorter canvas (11x9.2) trims vertical whitespace; width and font sizes unchanged.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2))
    axes = axes.flatten()
    for ax, enc in zip(axes, ENCODERS):
        cm = confusion_matrix(y_true, y_pred_per_enc[enc], labels=FAMILY_ORDER,
                              normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f.capitalize() for f in FAMILY_ORDER],
                    yticklabels=[f.capitalize() for f in FAMILY_ORDER],
                    vmin=0, vmax=1, ax=ax, square=True,
                    cbar_kws={"shrink": 0.82})
        acc = float((np.asarray(y_pred_per_enc[enc]) == np.asarray(y_true)).mean())
        ax.set_title(f"{ENC_LABEL[enc]}\nLOO acc = {acc:.3f}",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("True family")
        ax.set_xlabel("Predicted")
    fig.suptitle("Family-level leave-one-out kNN (k=5, cos)",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    # Extra row gap: the top-row xlabel would otherwise collide with the bottom panel titles.
    fig.subplots_adjust(hspace=0.32)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_encoder_summary(sils: dict, accs: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(ENCODERS))
    w = 0.36
    ax.bar(x - w / 2, [sils[e] for e in ENCODERS], w,
           label="Silhouette (family prototypes)", color="#0abde3")
    ax.bar(x + w / 2, [accs[e] for e in ENCODERS], w,
           label="LOO kNN accuracy (clips)", color="#10ac84")
    for xi, e in zip(x, ENCODERS):
        ax.text(xi - w / 2, sils[e] + 0.012, f"{sils[e]:.3f}",
                ha="center", fontsize=9)
        ax.text(xi + w / 2, accs[e] + 0.012, f"{accs[e]:.3f}",
                ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODERS], fontsize=11)
    ax.set_ylim(min(-0.05, min(sils.values()) - 0.06), 1.06)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Encoder summary: H1a confirmation (dev, n=9,573)",
                 fontsize=13, fontweight="bold")
    # Legend below the axes: at lower right it covered the AudioMAE bars.
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(SPLIT_DIR / "folkmusic_dev_fragments.csv")
    df = df.sort_values(["recording_id", "fragment_id"]).reset_index(drop=True)
    y_fam = df["family_mapped"].to_numpy()

    audit = json.loads((PROJ / "results" / "e3" / "audit_fix.json").read_text(encoding="utf-8"))
    dev_sil = audit["exact_v2"]["dev"]

    preds, accs, sils = {}, {}, {}
    for enc in ENCODERS:
        d = np.load(EMB_DIR / f"{enc}_folkmusic_dev.npz")
        X = d["embeddings"]
        assert len(X) == len(y_fam), (enc, X.shape, len(y_fam))
        pred = loo_predict(X, y_fam)
        preds[enc] = pred
        accs[enc] = float((pred == y_fam).mean())
        sils[enc] = float(dev_sil[enc]["sil_real"])
        cm = confusion_matrix(y_fam, pred, labels=FAMILY_ORDER, normalize="true")
        per_fam = {FAMILY_ORDER[i]: round(float(cm[i, i]), 4) for i in range(3)}
        print(f"  {enc:>8s}  LOO-acc={accs[enc]:.4f}  sil(proto)={sils[enc]:.4f}  "
              f"per-fam: " + ", ".join(f"{k}={v}" for k, v in per_fam.items()))

    print("[plot] confusion matrices (true LOO) ...")
    plot_confusion_grid(y_fam, preds, FIG_DIR / "figS1_confusion_matrix.png")

    print("[plot] encoder summary (prototype sil + true LOO acc) ...")
    plot_encoder_summary(sils, accs, FIG_DIR / "figS3_encoder_comparison.png")

    out = {
        "note": "true leave-one-out kNN (k=5, cosine, distance-weighted, self-excluded)",
        "n": int(len(y_fam)),
        "k": K,
        "families": FAMILY_ORDER,
        "loo_knn_acc": {e: round(accs[e], 6) for e in ENCODERS},
        "sil_proto_dev": {e: round(sils[e], 6) for e in ENCODERS},
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
