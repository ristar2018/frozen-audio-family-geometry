# -*- coding: utf-8 -*-
"""
e3_retrieval_fix.py
===================
Purpose: fix the top-k selection defect in the retrieval Recall@k of e2_supplement_analyze.part4_downstream
and give authoritative instrument/family R@1/5/10 (dev gallery -> test query).

The defect (e2_supplement_analyze.py:466):
    top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
    ...  hit = np.any(yg[top[q, :k]] == yq[q])
argpartition(kth=kmax-1) only guarantees that the first kmax positions hold the top-kmax set (correct set, unordered);
slicing top[q, :k] (k<kmax) afterwards then takes k arbitrary members of that top-10 set, so:
  - R@10 is correct;
  - R@5 is biased low (any 5 of the top 10);
  - R@1 is badly biased low (any 1 of the top 10 rather than the maximum).
This script takes the exact top k with np.argsort(-S) and reports:
  (1) authoritative R@k (both levels);
  (2) a digit-by-digit check against r1_ins in e3_bootstrap_ci (argmax protocol, same setup);
  (3) the recomputed values under the old e2 protocol (to quantify the bias).
Output: results/e3/retrieval_fix.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "code"))

from e2_supplement_analyze import prepare_features  # noqa: E402

KS = (1, 5, 10)
ORDER = ["beats", "mert", "ast", "audiomae", "mfcc"]


def topk_exact(S, k):
    """Exact top-k: returns (n_q, k) indices in descending similarity."""
    return np.argsort(-S, axis=1)[:, :k]


def recall(S, gal_lab, q_lab, ks):
    top = topk_exact(S, max(ks))
    out = {}
    for k in ks:
        hits = np.array([np.any(gal_lab[top[i, :k]] == q_lab[i]) for i in range(len(q_lab))])
        out[f"R@{k}"] = float(hits.mean())
    return out


def recall_buggy(S, gal_lab, q_lab, ks):
    """Recompute the old e2 protocol (argpartition then slice the first k) to quantify the bias."""
    kmax = max(ks)
    top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
    out = {}
    for k in ks:
        hits = np.array([np.any(gal_lab[top[i, :k]] == q_lab[i]) for i in range(len(q_lab))])
        out[f"R@{k}"] = float(hits.mean())
    return out


def main():
    t0 = time.time()
    feats = prepare_features()
    avail = [w for w in ORDER if w in feats and feats[w]["test"] is not None]

    ref_ci = json.loads((PROJ / "results" / "e3" / "bootstrap_ci.json").read_text(encoding="utf-8"))
    ref_e2 = json.loads((PROJ / "results" / "e2_supplement" / "e2_supplement_results.json").read_text(encoding="utf-8"))

    res = {"note": "dev gallery (9573) -> test query (2393), inner product after L2 equals cosine; top-k taken exactly with argsort",
           "n_gallery": None, "n_query": None, "encoders": {}}

    for w in avail:
        Xg, dg = feats[w]["dev"]
        Xq, dq = feats[w]["test"]
        res["n_gallery"], res["n_query"] = int(len(Xg)), int(len(Xq))
        yg_f, yq_f = dg["family"].to_numpy(), dq["family"].to_numpy()
        yg_c, yq_c = dg["instrument_class"].to_numpy(), dq["instrument_class"].to_numpy()
        S = Xq @ Xg.T

        r_ins = recall(S, yg_c, yq_c, KS)
        r_fam = recall(S, yg_f, yq_f, KS)
        r_ins_bug = recall_buggy(S, yg_c, yq_c, KS)
        r1_argmax = float((yg_c[np.argmax(S, axis=1)] == yq_c).mean())

        ref_r1 = ref_ci["encoders"][w]["ci"]["r1_ins"]["point"]
        ref_e2_r1 = None
        try:
            ref_e2_r1 = ref_e2["part4_downstream"][w]["retrieval"]["R@1_instrument"]
        except Exception:
            pass

        res["encoders"][w] = {
            "instrument": r_ins, "family": r_fam,
            "instrument_buggy_argpartition": r_ins_bug,
            "instrument_R@1_argmax": r1_argmax,
            "crosscheck_bootstrap_ci_r1_ins": ref_r1,
            "crosscheck_e2_part4_R@1_instrument": ref_e2_r1,
            "delta_vs_bootstrap_ci": r_ins["R@1"] - ref_r1,
        }
        print(f"[{w:9s}] ins R@1/5/10 = {r_ins['R@1']:.4f}/{r_ins['R@5']:.4f}/{r_ins['R@10']:.4f}"
              f" | fam = {r_fam['R@1']:.4f}/{r_fam['R@5']:.4f}/{r_fam['R@10']:.4f}")
        print(f"            old e2 (argpartition) ins R@1/5/10 = "
              f"{r_ins_bug['R@1']:.4f}/{r_ins_bug['R@5']:.4f}/{r_ins_bug['R@10']:.4f}"
              f" | bootstrap_ci r1_ins = {ref_r1:.4f} | e2 json = "
              f"{ref_e2_r1 if ref_e2_r1 is None else round(ref_e2_r1, 4)}")

    res["elapsed_sec"] = round(time.time() - t0, 1)
    out = PROJ / "results" / "e3" / "retrieval_fix.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
