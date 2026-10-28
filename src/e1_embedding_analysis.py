# -*- coding: utf-8 -*-
"""
FAPN E1 main script (method Sections 3.4 and 3.5 of the paper)

Tests H1a: whether the native embeddings phi of four frozen encoders (BEATs iter3 / MERT v1-330M / AST / AudioMAE)
encode the wind-bowed-plucked family structure by themselves.

Protocol (strict):
  * confirmatory evidence: FolkMusic 80% development split only, frozen phi, class-level (class-prototype) restricted permutation
    that keeps the per-family class counts 6/2/7, N=2000, p=(1+k)/2001, alpha=0.01, conjunction over the four encoders.
  * descriptive: fragment-level phi silhouette by family / within- and between-family cosine distance.
  * exploratory (not part of the decision): NSynth acoustic western-family replication, label-kernel CKA,
    pairwise encoder embedding CKA.
  * the 20% test split is isolated by recording group; only the split manifest is written and it never enters any E1 analysis.

Usage (inside the dedicated fapn environment):
  python e1_embedding_analysis.py --stage all
  python e1_embedding_analysis.py --encoder beats --stage extract
  python e1_embedding_analysis.py --stage analyze        # recompute the analysis from cached embeddings only
"""
import argparse
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    PROJECT_ROOT, RESULTS_ROOT,
    load_folkmusic_manifest, split_folkmusic_dev_test,
    load_nsynth_acoustic, nsynth_family_entity_counts, DEV_TEST_SEED,
)
from encoders import create_encoder
from analysis import (
    run_e1_folkmusic, run_e1_nsynth_exploratory,
    linear_cka, onehot_family_kernel, RANDOM_SEED,
)

RESULTS_DIR = RESULTS_ROOT / "e1"
EMB_DIR = RESULTS_DIR / "embeddings"
SPLIT_DIR = RESULTS_DIR / "splits"
ENCODER_ORDER = ["beats", "mert", "ast", "audiomae"]


def parse_args():
    p = argparse.ArgumentParser(description="FAPN E1 embedding-space analysis")
    p.add_argument("--encoder", default="all", choices=ENCODER_ORDER + ["all"])
    p.add_argument("--stage", default="all", choices=["extract", "analyze", "all"])
    p.add_argument("--model_path", default=None,
                   help="override the default HF repository (point at a local directory if BEATs is gated)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="H1a class-level permutation seed (analysis.RANDOM_SEED, fixed before the run)")
    p.add_argument("--split_seed", type=int, default=DEV_TEST_SEED)
    p.add_argument("--nsynth_per_entity", type=int, default=40,
                   help="NSynth exploratory sampling: max fragments per instrument entity (covers every entity)")
    p.add_argument("--nsynth_min_entities", type=int, default=5,
                   help="drop families with fewer instrument entities than this (bass=1/organ=4 carry no information under a class-level permutation)")
    p.add_argument("--output_dir", default=str(RESULTS_DIR))
    return p.parse_args()


# ============================================================
# Data preparation
# ============================================================
def prepare_data(args):
    """Return FolkMusic dev/test and the NSynth exploratory subset; write the split manifest to disk."""
    out = Path(args.output_dir)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    fm = load_folkmusic_manifest(verify_files=False)
    fm_dev, fm_test = split_folkmusic_dev_test(fm, test_size=0.2, seed=args.split_seed)

    # persist the split manifest (evidence that the test set is isolated; reused by E2-E7)
    fm_dev[["recording_id", "fragment_id", "instrument_class", "family_mapped"]].to_csv(
        out / "splits" / "folkmusic_dev_fragments.csv", index=False, encoding="utf-8-sig")
    fm_test[["recording_id", "fragment_id", "instrument_class", "family_mapped"]].to_csv(
        out / "splits" / "folkmusic_test_fragments.csv", index=False, encoding="utf-8-sig")

    # pre-check that every file exists
    def _missing(df):
        miss = [p for p in df["file_path"].tolist() if not os.path.exists(p)]
        return miss

    miss_fm = _missing(fm_dev)

    ns = load_nsynth_acoustic(
        exclude_vocal=True,
        min_entities_per_family=args.nsynth_min_entities,
        splits=("train",),
    )
    # per-entity capped sampling (stratified, covering every entity), exploratory
    if args.nsynth_per_entity and args.nsynth_per_entity > 0:
        rng = np.random.default_rng(args.split_seed)
        keep_idx = []
        for _, sub in ns.groupby("entity_id"):
            idx = sub.index.to_numpy()
            if len(idx) > args.nsynth_per_entity:
                idx = rng.choice(idx, args.nsynth_per_entity, replace=False)
            keep_idx.append(idx)
        keep_idx = np.concatenate(keep_idx)
        ns = ns.loc[np.sort(keep_idx)].reset_index(drop=True)
    miss_ns = _missing(ns)

    print(f"[data] FolkMusic dev={len(fm_dev)} test={len(fm_test)} "
          f"(test recordings verified non-overlapping); missing files dev={len(miss_fm)}")
    print(f"[data] NSynth exploratory subset n={len(ns)}, "
          f"families={ns['instrument_family'].nunique()}, entities={ns['entity_id'].nunique()}, "
          f"missing={len(miss_ns)}")

    meta = {
        "folkmusic_dev_n": int(len(fm_dev)),
        "folkmusic_test_n": int(len(fm_test)),
        "folkmusic_dev_missing_files": int(len(miss_fm)),
        "nsynth_n": int(len(ns)),
        "nsynth_families": int(ns["instrument_family"].nunique()),
        "nsynth_entities": int(ns["entity_id"].nunique()),
        "nsynth_missing_files": int(len(miss_ns)),
        "nsynth_per_entity_cap": int(args.nsynth_per_entity),
        "split_seed": int(args.split_seed),
        "permutation_seed": int(args.seed),
    }
    return fm_dev, fm_test, ns, meta


# ============================================================
# Embedding extraction and caching
# ============================================================
def _cache_path(out, encoder, tag):
    return Path(out) / "embeddings" / f"{encoder}_{tag}.npz"


def extract_and_cache(encoder_name, fm_dev, ns, args):
    out = Path(args.output_dir)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    enc = create_encoder(encoder_name, model_path=args.model_path)

    # --- FolkMusic dev ---
    fm_paths = fm_dev["file_path"].tolist()
    print(f"[extract] {encoder_name} FolkMusic dev: {len(fm_paths)} files")
    r = enc.encode_dataset(fm_paths, batch_size=args.batch_size)
    ok = np.array(r["ok_idx"])
    sub = fm_dev.iloc[ok].reset_index(drop=True)
    np.savez_compressed(
        _cache_path(out, encoder_name, "folkmusic_dev"),
        embeddings=r["embeddings"],
        fragment_id=sub["fragment_id"].to_numpy(),
        instrument_class=sub["instrument_class"].to_numpy(),
        family=sub["family_mapped"].to_numpy(),
    )
    _write_failed(out, encoder_name, "folkmusic_dev", r["failed"])
    print(f"  cached {r['embeddings'].shape}; failed={len(r['failed'])}")

    # --- NSynth exploratory subset ---
    ns_paths = ns["file_path"].tolist()
    print(f"[extract] {encoder_name} NSynth exploratory: {len(ns_paths)} files")
    r2 = enc.encode_dataset(ns_paths, batch_size=args.batch_size)
    ok2 = np.array(r2["ok_idx"])
    sub2 = ns.iloc[ok2].reset_index(drop=True)
    np.savez_compressed(
        _cache_path(out, encoder_name, "nsynth"),
        embeddings=r2["embeddings"],
        entity_id=sub2["entity_id"].to_numpy(),
        family=sub2["instrument_family"].to_numpy(),
        note_id=sub2["note_id"].to_numpy(),
    )
    _write_failed(out, encoder_name, "nsynth", r2["failed"])
    print(f"  cached {r2['embeddings'].shape}; failed={len(r2['failed'])}")

    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_failed(out, encoder, tag, failed):
    if not failed:
        return
    fp = Path(out) / "embeddings" / f"failed_{encoder}_{tag}.csv"
    pd.DataFrame(failed, columns=["row_index", "file_path", "error"]).to_csv(
        fp, index=False, encoding="utf-8-sig")


def load_cache(out, encoder, tag):
    d = np.load(_cache_path(out, encoder, tag), allow_pickle=True)
    return d


# ============================================================
# Analysis
# ============================================================
def analyze_encoder(encoder_name, args):
    out = Path(args.output_dir)
    d = load_cache(out, encoder_name, "folkmusic_dev")
    fm_res = run_e1_folkmusic(
        d["embeddings"], d["instrument_class"], d["family"],
        n_permutations=args.n_permutations, seed=args.seed,
    )
    # label-kernel CKA (exploratory)
    cka_label_fm = linear_cka(d["embeddings"], onehot_family_kernel(d["family"]))

    # NSynth is exploratory: a missing cache does not block the confirmatory H1a analysis
    ns_path = _cache_path(out, encoder_name, "nsynth")
    if ns_path.exists():
        dn = load_cache(out, encoder_name, "nsynth")
        ns_res = run_e1_nsynth_exploratory(
            dn["embeddings"], dn["entity_id"], dn["family"], seed=args.seed,
        )
        cka_label_ns = float(linear_cka(dn["embeddings"],
                                        onehot_family_kernel(dn["family"])))
        ns_block = {
            "permutation_test": _jsonable(ns_res["exploratory"]),
            "descriptive": _jsonable(ns_res["descriptive"]),
            "cka_embedding_vs_family_label": cka_label_ns,
        }
    else:
        ns_block = {"status": "nsynth cache missing; exploratory analysis skipped"}

    result = {
        "encoder": encoder_name,
        "folkmusic": {
            "confirmatory_H1a": _jsonable(fm_res["confirmatory"]),
            "descriptive": _jsonable(fm_res["descriptive"]),
            "cka_embedding_vs_family_label": float(cka_label_fm),
        },
        "nsynth_exploratory": ns_block,
    }
    with open(out / f"results_{encoder_name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result, d


def pairwise_encoder_cka(out, encoders):
    """Align on the shared fragment_id and compute pairwise encoder embedding-kernel linear CKA (exploratory)."""
    aligned = {}
    for e in encoders:
        d = load_cache(out, e, "folkmusic_dev")
        df = pd.DataFrame({"fragment_id": d["fragment_id"], "idx": np.arange(len(d["fragment_id"]))})
        aligned[e] = (d, df)
    common = None
    for e, (_, df) in aligned.items():
        s = set(df["fragment_id"].tolist())
        common = s if common is None else common & s
    common = sorted(common)
    mats = {}
    for e, (d, df) in aligned.items():
        m = dict(zip(df["fragment_id"], df["idx"]))
        idx = np.array([m[fid] for fid in common])
        mats[e] = d["embeddings"][idx]
    C = pd.DataFrame(np.eye(len(encoders)), index=encoders, columns=encoders)
    for i in range(len(encoders)):
        for j in range(i + 1, len(encoders)):
            v = linear_cka(mats[encoders[i]], mats[encoders[j]])
            C.loc[encoders[i], encoders[j]] = v
            C.loc[encoders[j], encoders[i]] = v
    return C, len(common)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    encoders = ENCODER_ORDER if args.encoder == "all" else [args.encoder]

    fm_dev = fm_test = ns = meta = None
    if args.stage in ("extract", "all"):
        fm_dev, fm_test, ns, meta = prepare_data(args)
        for e in encoders:
            print(f"\n===== EXTRACT {e} =====")
            extract_and_cache(e, fm_dev, ns, args)

    if args.stage in ("analyze", "all"):
        if meta is None:
            # analyze-only: rebuild the size metadata from the split manifest
            d0 = pd.read_csv(out / "splits" / "folkmusic_dev_fragments.csv")
            t0 = pd.read_csv(out / "splits" / "folkmusic_test_fragments.csv")
            meta = {"folkmusic_dev_n": len(d0), "folkmusic_test_n": len(t0),
                    "split_seed": args.split_seed, "permutation_seed": args.seed}

        # analyse only the encoders whose cache is complete
        ready = [e for e in encoders
                 if _cache_path(out, e, "folkmusic_dev").exists()]
        missing = [e for e in encoders if e not in ready]
        if missing:
            print(f"[warn] cached embeddings missing, skipping: {missing} (run --stage extract first)")

        per_encoder, h1a = {}, {}
        for e in ready:
            print(f"\n===== ANALYZE {e} =====")
            res, _ = analyze_encoder(e, args)
            per_encoder[e] = res
            conf = res["folkmusic"]["confirmatory_H1a"]
            h1a[e] = {"p_value": conf["p_value"], "sil_real": conf["sil_real"],
                      "sil_perm_q99": conf["sil_perm_q99"], "passed": conf["passed"]}
            print(f"  H1a p={conf['p_value']:.4f} sil_real={conf['sil_real']:.4f} "
                  f"q99={conf['sil_perm_q99']:.4f} -> {'PASS' if conf['passed'] else 'FAIL'}")

        # pairwise encoder CKA (FolkMusic, shared fragments)
        cka_pair = None
        n_common = 0
        if len(ready) >= 2:
            Cmat, n_common = pairwise_encoder_cka(out, ready)
            cka_pair = Cmat.round(4).to_dict()
            print(f"\n[CKA] pairwise encoders (FolkMusic, n={n_common}):\n{Cmat.round(4)}")

        # H1a conjunction: all four encoders must pass (final decision only once all are ready)
        all_four = all(e in h1a for e in ENCODER_ORDER)
        n_pass = sum(1 for v in h1a.values() if v["passed"])
        h1a_conjunction = bool(all_four and n_pass == 4)

        summary = {
            "experiment": "E1",
            "protocol_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "meta": meta,
            "confirmatory_rule": {
                "space": "frozen phi", "scope": "FolkMusic 80% dev only",
                "permutation": "class-level (class prototype), family-surjective, 6/2/7",
                "n_permutations": int(args.n_permutations),
                "p_formula": "(1 + #{sil_perm>=sil_real})/(N+1)",
                "alpha": 0.01, "conjunction": "all 4 encoders pass",
            },
            "H1a_per_encoder": h1a,
            "H1a_encoders_present": ready,
            "H1a_conjunction_verified": h1a_conjunction,
            "cka_pairwise_folkmusic": cka_pair,
            "cka_n_common_fragments": int(n_common),
            "cka_note": "CKA is exploratory evidence and does not enter the confirmatory H1a decision",
        }
        with open(out / "e1_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print("H1a decision (FolkMusic frozen phi only, class-level permutation)")
        for e in ENCODER_ORDER:
            if e in h1a:
                v = h1a[e]
                print(f"  {e:9s} p={v['p_value']:.4f} sil={v['sil_real']:.4f} "
                      f"{'PASS' if v['passed'] else 'FAIL'}")
        if all_four:
            print(f"\nConjunction: {'H1a holds (4/4 pass)' if h1a_conjunction else f'H1a does not hold ({n_pass}/4)'}")
        else:
            print(f"\n{len(ready)}/4 encoders ready; the final conjunction decision is not available yet.")
        print("=" * 60)


if __name__ == "__main__":
    main()
