# -*- coding: utf-8 -*-
"""
Draw Fig. A.4 (figS3_encoder_comparison): encoder summary panel.

Five groups: the four frozen encoders plus the non-neural MFCC baseline, so the
figure carries the family-level summary against which the MFCC baseline of
Section 4.2 is read. Two quantities per group, both taken from archived JSON so
the figure reproduces without the embedding caches:
  - silhouette of the 15 instrument-class prototypes (dev)
  - leave-one-out clip-level family kNN accuracy (dev, n=9,573)

Sources:
  silhouette, four encoders   results/e3/audit_fix.json    exact_v2.dev[enc].sil_real
  silhouette, MFCC            results/e3/mfcc_baseline.json authoritative.dev.sil_real
  LOO kNN, four encoders      results/e3/fig_fix.json      loo_knn_acc[enc]
  LOO kNN, MFCC               results/e3/mfcc_baseline.json authoritative.dev_clip_knn.knn_family_loo

Both quantities match the sil_proto and kNN_clip columns of Table 1.

Output:
  results/e1/figures/figS3_encoder_comparison.png / .pdf
  results/e3/fig_s3.json   (audit trail: values and provenance pointers)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

PROJ = Path(__file__).resolve().parent.parent
E3 = PROJ / "results" / "e3"
FIG_DIR = PROJ / "results" / "e1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same group order as figS2 (e3_fig_familybar_fix.ENC).
ENC = ["beats", "mert", "ast", "audiomae", "mfcc"]
LABEL = {"beats": "BEATs", "mert": "MERT-v1", "ast": "AST",
         "audiomae": "AudioMAE", "mfcc": "MFCC"}

# Table 1 values, used as a cross-check; abort on mismatch.
EXPECT_SIL = {"beats": 0.351, "mert": 0.111, "ast": 0.335,
              "audiomae": 0.100, "mfcc": 0.027}
EXPECT_KNN = {"beats": 0.974, "mert": 0.968, "ast": 0.974,
              "audiomae": 0.916, "mfcc": 0.956}

C_SIL = "#0abde3"   # cyan  -- family silhouette (prototypes)
C_KNN = "#10ac84"   # green -- LOO kNN accuracy (clips)

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False


def load_values() -> tuple[dict, dict, dict]:
    audit = json.loads((E3 / "audit_fix.json").read_text(encoding="utf-8"))
    fig_fix = json.loads((E3 / "fig_fix.json").read_text(encoding="utf-8"))
    mfcc = json.loads((E3 / "mfcc_baseline.json").read_text(encoding="utf-8"))

    dev_sil = audit["exact_v2"]["dev"]
    knn = fig_fix["loo_knn_acc"]
    mfcc_auth = mfcc["authoritative"]

    sils, accs, prov = {}, {}, {}
    for e in ENC[:-1]:
        sils[e] = float(dev_sil[e]["sil_real"])
        accs[e] = float(knn[e])
        prov[e] = {
            "sil": "audit_fix.json:exact_v2.dev.%s.sil_real" % e,
            "knn": "fig_fix.json:loo_knn_acc.%s" % e,
        }
    sils["mfcc"] = float(mfcc_auth["dev"]["sil_real"])
    accs["mfcc"] = float(mfcc_auth["dev_clip_knn"]["knn_family_loo"])
    prov["mfcc"] = {
        "sil": "mfcc_baseline.json:authoritative.dev.sil_real",
        "knn": "mfcc_baseline.json:authoritative.dev_clip_knn.knn_family_loo",
    }
    return sils, accs, prov


def check_against_table(sils: dict, accs: dict) -> None:
    """Cross-check against Table 1 at three decimals to catch source drift."""
    bad = []
    for e in ENC:
        if round(sils[e], 3) != EXPECT_SIL[e]:
            bad.append(f"sil[{e}] {sils[e]:.6f} -> {round(sils[e], 3)} != {EXPECT_SIL[e]}")
        if round(accs[e], 3) != EXPECT_KNN[e]:
            bad.append(f"knn[{e}] {accs[e]:.6f} -> {round(accs[e], 3)} != {EXPECT_KNN[e]}")
    assert not bad, "values differ from Table 1:\n  " + "\n  ".join(bad)
    print("[check] sil / kNN of all five groups match Table 1")


def plot_encoder_summary(sils: dict, accs: dict, save_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(ENC))
    w = 0.36
    ax.bar(x - w / 2, [sils[e] for e in ENC], w,
           label="Silhouette (family prototypes)", color=C_SIL)
    ax.bar(x + w / 2, [accs[e] for e in ENC], w,
           label="LOO kNN accuracy (clips)", color=C_KNN)
    for xi, e in zip(x, ENC):
        ax.text(xi - w / 2, sils[e] + 0.012, f"{sils[e]:.3f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, accs[e] + 0.012, f"{accs[e]:.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[e] for e in ENC], fontsize=10.5)
    ax.set_ylim(min(-0.05, min(sils.values()) - 0.06), 1.06)
    ax.set_ylabel("Score", fontsize=12)
    # Title omits "H1a" (the term never appears in the paper) and "confirmation"
    # (AudioMAE fails H1, p=0.013).
    ax.set_title("Encoder summary (dev, n=9,573)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(save_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(save_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sils, accs, prov = load_values()
    for e in ENC:
        print(f"  {LABEL[e]:>9s}  sil={sils[e]:.6f}  knn={accs[e]:.6f}   <- {prov[e]['sil']}")
    check_against_table(sils, accs)

    stem = FIG_DIR / "figS3_encoder_comparison"
    print("[plot] encoder summary (5 groups, MFCC included) ...")
    plot_encoder_summary(sils, accs, stem)

    out = {
        "note": "Fig. A.4 encoder summary; silhouette = 15 instrument-class prototypes (dev), "
                "kNN = leave-one-out clip-level family accuracy (n=9573); "
                "values reproduce Table 1 (tab:h1a) columns sil_proto and kNN_clip",
        "n": 9573,
        "encoders": ENC,
        "labels": {e: LABEL[e] for e in ENC},
        "sil_proto_dev": {e: round(sils[e], 6) for e in ENC},
        "loo_knn_acc": {e: round(accs[e], 6) for e in ENC},
        "provenance": prov,
        "figure_title": "Encoder summary (dev, n=9,573)",
    }
    (E3 / "fig_s3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {E3 / 'fig_s3.json'}")
    print(f"[saved] {stem.with_suffix('.png')}")
    print(f"[saved] {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
