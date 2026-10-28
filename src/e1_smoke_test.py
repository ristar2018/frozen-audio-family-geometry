# -*- coding: utf-8 -*-
"""
E1 smoke test: verify on a few real audio files that every encoder loads and emits a 768-dim L2-normalised vector.
Start with AST alone (known to be downloadable) to validate the pipeline, then add the others one by one.

Usage (fapn environment; set the HF mirror first if needed):
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  python e1_smoke_test.py                 # tests ast only by default
  python e1_smoke_test.py --encoders ast mert audiomae
  # beats needs download_beats.ps1 first (fetches the unilm code and the .pt into models/beats/), then:
  python e1_smoke_test.py --encoders beats
  python e1_smoke_test.py --encoders beats --model_path <PROJECT_ROOT>/models/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_folkmusic_manifest, load_nsynth_acoustic
from encoders import create_encoder


def pick_files(per_instrument=2):
    fm = load_folkmusic_manifest(verify_files=True)
    files = []
    for inst, sub in fm.groupby("instrument_class"):
        files.extend(sub["file_path"].head(per_instrument).tolist())
    return files[:30]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["ast"])
    p.add_argument("--model_path", default=None)
    args = p.parse_args()

    files = pick_files()
    print(f"smoke files: {len(files)}")
    assert files, "no usable FolkMusic audio found"

    results = {}
    for name in args.encoders:
        print(f"\n===== smoke: {name} =====")
        try:
            enc = create_encoder(name, model_path=args.model_path)
            r = enc.encode_dataset(files[:8], batch_size=4, log_every=1)
            e = r["embeddings"]
            norms = np.linalg.norm(e, axis=1)
            ok = (
                e.shape[1] == 768
                and np.allclose(norms, 1.0, atol=1e-3)
                and not np.isnan(e).any()
            )
            results[name] = {
                "ok": bool(ok), "shape": list(e.shape),
                "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                "failed_files": len(r["failed"]),
            }
            print(f"  shape={e.shape} norm=[{norms.min():.4f},{norms.max():.4f}] "
                  f"failed={len(r['failed'])} -> {'PASS' if ok else 'CHECK'}")
            del enc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as ex:
            results[name] = {"ok": False, "error": repr(ex)}
            print(f"  FAIL: {ex}")

    print("\n===== smoke summary =====")
    for k, v in results.items():
        print(f"  {k:9s} {v}")


if __name__ == "__main__":
    main()
