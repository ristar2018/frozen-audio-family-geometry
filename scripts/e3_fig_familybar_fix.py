"""Regenerate figS2_family_bar.png with the paper's Delta-cos definition.

Why: the original figS2 (e1_visualization_pack.plot_family_bar) averaged a
*subsample* of 50 same-family / 50 different-family neighbours per query and
used the label "MERT-v2".  Its Delta annotations (0.230 / 0.042 / 0.029 / 0.020)
therefore did not equal the Delta-cos reported in Table 3, which is the
family-equal-weighted, all-clip statistic computed by e3_audit_fix.dcos_vec.
This script recomputes within/between means with exactly that definition so the
figure and the table agree, and writes MERT-v1.

v2 (baseline batch): the non-neural MFCC baseline now has a Delta-cos row in
Table 3 (0.144), so it is added as a fifth bar; its features are the 240-d
descriptor of results/e2_supplement/mfcc/mfcc_folkmusic_dev.npz, standardised on
the development split and L2-normalised, exactly as in e2_supplement_analyze.

Output: results/e1/figures/figS2_family_bar.png  (+ results/e3/fig_familybar.json)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from e2_supplement_analyze import l2norm  # noqa: E402

FIG = ROOT / "results" / "e1" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SPLIT = ROOT / "results" / "e1" / "splits" / "folkmusic_dev_fragments.csv"
EMB = ROOT / "results" / "e1" / "embeddings"
MFCC_DIR = ROOT / "results" / "e2_supplement" / "mfcc"
ENC = ["beats", "mert", "ast", "audiomae", "mfcc"]
LABEL = {"beats": "BEATs", "mert": "MERT-v1", "ast": "AST", "audiomae": "AudioMAE",
         "mfcc": "MFCC"}
CHUNK = 512


def within_between(X, fam):
    """Family-equal-weighted within/between mean cosine (all clips, no sampling)."""
    X = l2norm(np.asarray(X, dtype=np.float32))
    n = len(X)
    fams = sorted(set(fam))
    wsum = {g: 0.0 for g in fams}
    wcnt = {g: 0 for g in fams}
    bsum = {g: 0.0 for g in fams}
    bcnt = {g: 0 for g in fams}
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        S = X[s:e] @ X.T                                  # (chunk, n)
        rows = np.arange(s, e)
        S[np.arange(e - s), rows] = 0.0                   # drop self-similarity
        for g in fams:
            m = fam == g
            sel = m[rows]
            if not sel.any():
                continue
            sub = S[sel]
            wsum[g] += float(sub[:, m].sum())
            wcnt[g] += int(sel.sum()) * (int(m.sum()) - 1)
            bsum[g] += float(sub[:, ~m].sum())
            bcnt[g] += int(sel.sum()) * int((~m).sum())
    within = {g: wsum[g] / wcnt[g] for g in fams}
    between = {g: bsum[g] / bcnt[g] for g in fams}
    return (float(np.mean([within[g] for g in fams])),
            float(np.mean([between[g] for g in fams])), within, between)


def load_features(enc, fam, n):
    if enc == "mfcc":
        from sklearn.preprocessing import StandardScaler
        d = np.load(MFCC_DIR / "mfcc_folkmusic_dev.npz", allow_pickle=True)
        X = d["features"].astype(np.float64)
        assert len(X) == n, (enc, len(X), n)
        assert bool((d["family"] == fam).all()), "MFCC dev order does not match the split file"
        return l2norm(StandardScaler().fit(X).transform(X))
    f = EMB / f"{enc}_folkmusic_dev.npz"
    if not f.exists():
        f = EMB / f"folkmusic_dev_{enc}.npz"
    X = np.load(f)["embeddings"]
    assert len(X) == n, (enc, len(X), n)
    return X


def main():
    df = pd.read_csv(SPLIT)
    fam = df["family_mapped"].to_numpy()
    n = len(df)
    print(f"[meta] dev clips = {n}  families = { {g: int((fam == g).sum()) for g in sorted(set(fam))} }")

    out, W, B = {}, {}, {}
    for enc in ENC:
        X = load_features(enc, fam, n)
        W[enc], B[enc], wi, be = within_between(X, fam)
        out[enc] = {"within": round(W[enc], 6), "between": round(B[enc], 6),
                    "dcos": round(W[enc] - B[enc], 6),
                    "per_family_within": {g: round(v, 6) for g, v in wi.items()},
                    "per_family_between": {g: round(v, 6) for g, v in be.items()}}
        print(f"{LABEL[enc]:9s} within={W[enc]:.4f} between={B[enc]:.4f} "
              f"dcos={W[enc]-B[enc]:.4f}")

    # cross-check against the tabulated Delta-cos (encoders: audit_fix; MFCC: e3_mfcc_baseline)
    ref = json.loads((ROOT / "results" / "e3" / "audit_fix.json").read_text(encoding="utf-8"))
    mf = json.loads((ROOT / "results" / "e3" / "mfcc_baseline.json").read_text(encoding="utf-8"))
    print("[check] tabulated vs recomputed dcos")
    for enc in ENC:
        t = mf["recomputed"]["dev_dcos_audit_fix_ref"] if enc == "mfcc" \
            else ref["exact_v2"]["dev"][enc]["dcos_real"]
        d = out[enc]["dcos"]
        print(f"  {LABEL[enc]:9s} table={t:.6f} recomputed={d:.6f} diff={abs(t-d):.2e}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(ENC))
    w = 0.36
    ax.bar(x - w / 2, [W[e] for e in ENC], w, label="Within-family cos", color="#2e86de")
    ax.bar(x + w / 2, [B[e] for e in ENC], w, label="Between-family cos", color="#ee5253")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[e] for e in ENC], fontsize=11)
    ax.set_ylabel("Mean cosine similarity", fontsize=12)
    ax.set_title("Separation: within-family vs between-family", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(-0.15, 1.05)
    ax.grid(axis="y", alpha=0.3)
    for i, e in enumerate(ENC):
        ax.text(i - w / 2, W[e] + 0.02, f"$\\Delta$={W[e]-B[e]:.3f}",
                ha="center", fontsize=9, color="#2e86de", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "figS2_family_bar.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    (ROOT / "results" / "e3" / "fig_familybar.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[saved] figS2_family_bar.png + results/e3/fig_familybar.json")


if __name__ == "__main__":
    main()
