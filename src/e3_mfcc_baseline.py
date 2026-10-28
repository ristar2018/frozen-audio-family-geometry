# -*- coding: utf-8 -*-
"""
e3_mfcc_baseline.py -- collect every authoritative MFCC (non-neural baseline) number across the probes

Purpose:
  1) fill in the missing MFCC 15-class clip-level silhouette in the left column of tab:fine (exact over all 9,573 fragments);
  2) back-fill the MFCC Delta-cos with the "family-equal weight, all fragments" definition of Eq.(1) in the paper and
     cross-check it against dcos_real in audit_fix.json (should agree digit for digit), so Fig. A.2 can include MFCC;
  3) collect the authoritative MFCC values on dev / test / 15-class / few-shot / retrieval / Saraga
     into one JSON, the sole data source for the new "non-neural baseline" subsection.

MFCC protocol identical to e2_supplement_analyze.prepare_features:
  240 raw statistics -> StandardScaler fit on dev -> L2 normalisation.

Output: results/e3/mfcc_baseline.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from e2_supplement_analyze import prepare_features  # noqa: E402

OUT = PROJ / "results" / "e3" / "mfcc_baseline.json"
CHUNK = 2048


def within_between_family_equal(X: np.ndarray, fam: np.ndarray):
    """Eq.(1) protocol of the paper: family-equal weight over all fragments (self included, self contributes 0), within/between mean difference."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    fams = np.unique(fam)
    wsum = {g: 0.0 for g in fams}
    wcnt = {g: 0 for g in fams}
    bsum = {g: 0.0 for g in fams}
    bcnt = {g: 0 for g in fams}
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        S = X[s:e] @ X.T
        S[np.arange(e - s), np.arange(s, e)] = 0.0
        rows = np.arange(s, e)
        for g in fams:
            m = fam == g
            sel = m[rows]
            sub = S[sel]
            wsum[g] += float(sub[:, m].sum())
            wcnt[g] += int(sel.sum()) * (int(m.sum()) - 1)
            bsum[g] += float(sub[:, ~m].sum())
            bcnt[g] += int(sel.sum()) * int((~m).sum())
    w = np.mean([wsum[g] / wcnt[g] for g in fams])
    b = np.mean([bsum[g] / bcnt[g] for g in fams])
    return float(w), float(b), float(w - b)


def main() -> None:
    feats = prepare_features()
    Xd, dfd = feats["mfcc"]["dev"]
    Xt, dft = feats["mfcc"]["test"]
    print(f"[mfcc] dev={Xd.shape} test={Xt.shape}")

    y_cls = dfd["instrument_class"].to_numpy()
    y_fam = dfd["family"].to_numpy()

    sil15 = float(silhouette_score(Xd, y_cls, metric="cosine"))
    print(f"[mfcc] 15-class silhouette (exact, all clips) = {sil15:.6f}")

    w, b, dcos = within_between_family_equal(Xd, y_fam)
    print(f"[mfcc] within={w:.6f} between={b:.6f} dcos_fig={dcos:.6f}")

    audit = json.loads((PROJ / "results" / "e3" / "audit_fix.json").read_text(encoding="utf-8"))
    ref = audit["exact_v2"]["dev"]["mfcc"]["dcos_real"]
    print(f"[check] audit_fix dcos_real={ref:.6f}  diff={abs(ref - dcos):.2e}")

    res = {
        "note": "MFCC non-neural baseline: 240 dims (frame mean/std of 40 MFCC + delta + delta-delta), StandardScaler fit on dev then L2-normalised",
        "recomputed": {
            "folk15_sil_exact_all_clips": round(sil15, 6),
            "dev_dcos_fig_within": round(w, 6),
            "dev_dcos_fig_between": round(b, 6),
            "dev_dcos_fig": round(dcos, 6),
            "dev_dcos_audit_fix_ref": ref,
            "abs_diff_vs_audit_fix": float(abs(ref - dcos)),
        },
        "authoritative": {
            "dev": audit["exact_v2"]["dev"]["mfcc"],
            "test": audit["exact_v2"]["test"]["mfcc"],
            "dev_clip_knn": json.loads((PROJ / "results" / "e3" / "dev_clip_knn.json").read_text(encoding="utf-8")).get("mfcc"),
            "folk15_perclass_overall": json.loads((PROJ / "results" / "e3" / "folk15_perclass.json").read_text(encoding="utf-8")).get("mfcc", {}).get("overall"),
            "saraga": json.loads((PROJ / "results" / "e3" / "saraga_controls.json").read_text(encoding="utf-8")).get("encoders", {}).get("mfcc"),
        },
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
