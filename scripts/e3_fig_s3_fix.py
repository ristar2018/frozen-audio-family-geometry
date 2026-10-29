# -*- coding: utf-8 -*-
"""
e3_fig_s3_fix.py -- 重绘 Fig. A.4 (figS3_encoder_comparison)

为什么有这一版：
  1) e3_fig_fix.plot_encoder_summary() 只画四个神经编码器，没有 MFCC 基线。
     但 main.tex 正文把这幅图描述为 "the family-level summary against which the
     MFCC baseline of Section 4.2 is read"（Fig. A.4），图里却没有基线 —— 图文不符。
  2) 正文 results.tex 说 "The two columns already hint at the central result of
     the controls: instance discrimination and family-level geometry are different
     properties." 四个编码器里 AudioMAE 两项都是最低（sil 0.100 / kNN 0.916），
     读者读到的是 "AudioMAE 全面最弱"，而非 "两个性质不同"。补上 MFCC
     （sil 0.027 最低、kNN 0.956 第二高）才真正展示解耦。
  3) 图内标题原为 "Encoder summary: H1a confirmation (dev, n=9,573)"。正文从无
     "H1a" 这一术语（只有 H1），且 AudioMAE 的 H1 判定是未通过（p=0.013 > 0.01），
     标题写 "confirmation" 覆盖四个编码器会误导统计结论。
  4) 与 figS2 口径统一：figS2 修正版已是五柱含 MFCC。

数据来源（全部来自已归档 JSON，不依赖 embedding 缓存，可在干净环境复现）：
  - silhouette (15 instrument-class prototypes, dev) ：
      results/e3/audit_fix.json  -> exact_v2.dev[enc].sil_real      (4 个神经编码器)
      results/e3/mfcc_baseline.json -> authoritative.dev.sil_real    (MFCC)
  - leave-one-out clip-level family kNN accuracy ：
      results/e3/fig_fix.json    -> loo_knn_acc[enc]                 (4 个神经编码器)
      results/e3/mfcc_baseline.json -> authoritative.dev_clip_knn.knn_family_loo (MFCC)

两者均与正文 Table 1 (tab:h1a) 的 sil_proto 与 kNN_clip 两列逐位一致。

输出：
  results/e1/figures/figS3_encoder_comparison.png / .pdf
  results/e3/fig_s3.json   （审计用：数值 + 来源指针）
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

# 顺序与 figS2 (e3_fig_familybar_fix.ENC) 保持一致
ENC = ["beats", "mert", "ast", "audiomae", "mfcc"]
LABEL = {"beats": "BEATs", "mert": "MERT-v1", "ast": "AST",
         "audiomae": "AudioMAE", "mfcc": "MFCC"}

# 正文 Table 1 (tab:h1a) 期望值，用于交叉校验；不一致即中止
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
    """与正文 Table 1 的三位小数逐项对齐，防止数据源漂移。"""
    bad = []
    for e in ENC:
        if round(sils[e], 3) != EXPECT_SIL[e]:
            bad.append(f"sil[{e}] {sils[e]:.6f} -> {round(sils[e], 3)} != {EXPECT_SIL[e]}")
        if round(accs[e], 3) != EXPECT_KNN[e]:
            bad.append(f"knn[{e}] {accs[e]:.6f} -> {round(accs[e], 3)} != {EXPECT_KNN[e]}")
    assert not bad, "与 Table 1 不一致：\n  " + "\n  ".join(bad)
    print("[check] 五个编码器的 sil / kNN 与正文 Table 1 逐项一致")


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
    # 去掉 "H1a"：正文无此术语，且 "confirmation" 会与 AudioMAE 的未通过判定冲突
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
