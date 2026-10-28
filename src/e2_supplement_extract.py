# -*- coding: utf-8 -*-
"""
FAPN E2 supplementary experiments, stage A: feature extraction (needs GPU)

Outputs (results/e2_supplement/):
  embeddings/{enc}_folkmusic_test.npz   the four encoders on the 20% held-out TEST split
      fields identical to the dev cache: embeddings(N,768) / fragment_id / instrument_class / family
  mfcc/mfcc_folkmusic_dev.npz           MFCC baseline (240 dims, raw statistics, not standardised)
  mfcc/mfcc_folkmusic_test.npz          fields: features / fragment_id / instrument_class / family
  mfcc/mfcc_saraga.npy                  MFCC baseline over the 108 Saraga concerts (108,240)

Protocol held identical to E1 so that dev and test stay comparable:
  * the split reuses data_loader.split_folkmusic_dev_test (seed=DEV_TEST_SEED, recording-grouped and stratified);
    at start-up the reproduced dev fragment_id set is asserted equal to the stored E1 dev embedding set (hard check).
  * encoders go through the same encoders.create_encoder: 16 kHz (MERT 24 kHz) mono, first 10 s, L2-768.
  * the MFCC baseline reuses encoders.load_audio_mono (same resampling, mono and long-audio ffmpeg guard),
    16 kHz, first 10 s; 40-dim MFCC + delta + delta-delta frame mean and std = 240 dims.
    Standardisation and L2 are deferred to the analysis script and fit on dev only, so test cannot leak.

Usage (fapn environment):
  python e2_supplement_extract.py                      # everything: 4 encoders on test + three MFCC sets
  python e2_supplement_extract.py --stage enc_test     # only the encoder test embeddings
  python e2_supplement_extract.py --stage mfcc         # only MFCC (CPU only, slow)
  python e2_supplement_extract.py --encoders beats ast # selected encoders
  python e2_supplement_extract.py --skip_saraga_mfcc
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))

from data_loader import (  # noqa: E402
    load_folkmusic_manifest, split_folkmusic_dev_test, DEV_TEST_SEED,
)
from encoders import create_encoder, load_audio_mono  # noqa: E402

E1_EMB = PROJ / "results" / "e1" / "embeddings"
OUT = PROJ / "results" / "e2_supplement"
EMB_OUT = OUT / "embeddings"
MFCC_OUT = OUT / "mfcc"
SARAGA_MANIFEST = PROJ / "data" / "openethnic" / "saraga" / "manifest.csv"
SARAGA_HINDU = PROJ / "data" / "openethnic" / "saraga" / "hindustani"

ENCODER_ORDER = ["beats", "mert", "ast", "audiomae"]
MFCC_SR = 16000
MFCC_MAX_SEC = 10.0
N_MFCC = 40


# ============================================================
# Data split (reproduce E1 under a hard consistency check)
# ============================================================
def build_splits():
    fm = load_folkmusic_manifest(verify_files=False)
    fm_dev, fm_test = split_folkmusic_dev_test(fm, test_size=0.2, seed=DEV_TEST_SEED)

    # hard check: the reproduced dev must align fragment by fragment with the stored E1 dev embeddings
    ref = np.load(E1_EMB / "beats_folkmusic_dev.npz", allow_pickle=True)
    ref_ids = set(np.atleast_1d(ref["fragment_id"]).tolist())
    dev_ids = set(fm_dev["fragment_id"].tolist())
    if ref_ids != dev_ids:
        only_ref = len(ref_ids - dev_ids)
        only_new = len(dev_ids - ref_ids)
        raise AssertionError(
            f"dev split differs from E1: only in E1={only_ref}, only reproduced={only_new}; "
            f"check whether split_seed or data_loader were modified.")
    print(f"[split] consistency check passed: dev={len(fm_dev)} test={len(fm_test)} "
          f"(identical to the {len(ref_ids)} E1 dev embedding fragments)")
    return fm_dev.reset_index(drop=True), fm_test.reset_index(drop=True)


# ============================================================
# Stage 1: test embeddings for the four encoders
# ============================================================
def extract_encoder_test(fm_test, encoders, batch_size=None):
    EMB_OUT.mkdir(parents=True, exist_ok=True)
    paths = fm_test["file_path"].tolist()
    for name in encoders:
        dst = EMB_OUT / f"{name}_folkmusic_test.npz"
        if dst.exists():
            print(f"[enc-test] {name}: already present, skipping ({dst.name})")
            continue
        print(f"\n[enc-test] === {name} : {len(paths)} test files ===")
        t0 = time.time()
        enc = create_encoder(name)
        r = enc.encode_dataset(paths, batch_size=batch_size, log_every=200)
        ok = np.array(r["ok_idx"])
        sub = fm_test.iloc[ok].reset_index(drop=True)
        np.savez_compressed(
            dst,
            embeddings=r["embeddings"],
            fragment_id=sub["fragment_id"].to_numpy(),
            instrument_class=sub["instrument_class"].to_numpy(),
            family=sub["family_mapped"].to_numpy(),
        )
        if r["failed"]:
            pd.DataFrame(r["failed"],
                         columns=["row_index", "file_path", "error"]).to_csv(
                EMB_OUT / f"failed_{name}_folkmusic_test.csv",
                index=False, encoding="utf-8-sig")
        print(f"[enc-test] {name} cached {r['embeddings'].shape}  "
              f"failed={len(r['failed'])}  {time.time()-t0:.0f}s")
        del enc
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================
# Stage 2: MFCC baseline (240 dims, CPU only)
# ============================================================
def mfcc_vector(path):
    import librosa
    w = load_audio_mono(path, MFCC_SR, max_sec=MFCC_MAX_SEC)  # float32 mono 16k
    if w.size < MFCC_SR * 0.1:  # anything under 0.1 s counts as a broken file
        raise ValueError("audio too short")
    m = librosa.feature.mfcc(y=w, sr=MFCC_SR, n_mfcc=N_MFCC,
                             n_fft=1024, hop_length=160)
    d1 = librosa.feature.delta(m)
    d2 = librosa.feature.delta(m, order=2)
    v = np.concatenate([
        m.mean(axis=1), m.std(axis=1),
        d1.mean(axis=1), d1.std(axis=1),
        d2.mean(axis=1), d2.std(axis=1),
    ])
    return v.astype(np.float32)  # 6 * 40 = 240


def extract_mfcc_table(df, tag, log_every=500):
    """Extract MFCC file by file over the DataFrame, dropping and recording failures; returns the table aligned with the successful rows."""
    n = len(df)
    feats, fids, cls, fam = [], [], [], []
    failed = []
    t0 = time.time()
    paths = df["file_path"].tolist()
    for i in range(n):
        try:
            feats.append(mfcc_vector(paths[i]))
            fids.append(df["fragment_id"].iloc[i])
            cls.append(df["instrument_class"].iloc[i])
            fam.append(df["family_mapped"].iloc[i])
        except Exception as ex:
            failed.append((i, paths[i], repr(ex)))
        if (i + 1) % log_every == 0:
            print(f"    [mfcc-{tag}] {i+1}/{n}  failed={len(failed)}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    if not feats:
        raise RuntimeError(f"MFCC {tag}: no successful features")
    X = np.stack(feats, axis=0)
    out = {
        "features": X,
        "fragment_id": np.array(fids),
        "instrument_class": np.array(cls),
        "family": np.array(fam),
    }
    print(f"  [mfcc-{tag}] {X.shape}  failed={len(failed)}  {time.time()-t0:.0f}s")
    return out, failed


def extract_mfcc(fm_dev, fm_test, do_saraga=True):
    import librosa  # noqa: F401  import early so a missing package fails with a clear error
    MFCC_OUT.mkdir(parents=True, exist_ok=True)

    for df, tag in [(fm_dev, "folkmusic_dev"), (fm_test, "folkmusic_test")]:
        dst = MFCC_OUT / f"mfcc_{tag}.npz"
        if dst.exists():
            print(f"[mfcc] {dst.name} already present, skipping")
            continue
        print(f"\n[mfcc] === {tag}: {len(df)} files (CPU) ===")
        out, failed = extract_mfcc_table(df, tag)
        np.savez_compressed(dst, **out)
        if failed:
            pd.DataFrame(failed, columns=["row_index", "file_path", "error"]).to_csv(
                MFCC_OUT / f"failed_mfcc_{tag}.csv", index=False, encoding="utf-8-sig")

    if do_saraga:
        dst = MFCC_OUT / "mfcc_saraga.npy"
        if dst.exists():
            print(f"[mfcc] {dst.name} already present, skipping")
        else:
            print("\n[mfcc] === Saraga Hindustani: 108 performances (CPU) ===")
            man = pd.read_csv(SARAGA_MANIFEST, encoding="utf-8-sig")
            paths = [str(SARAGA_HINDU / p) for p in man["audio_relpath"].tolist()]
            feats, failed = [], []
            for i, p in enumerate(paths):
                try:
                    feats.append(mfcc_vector(p))
                except Exception as ex:
                    # strictly aligned with the encoder cache: Saraga needs all 108 concerts, so abort on any failure
                    raise RuntimeError(f"Saraga MFCC failed idx={i} {p}: {ex}") from ex
            X = np.stack(feats, axis=0)
            assert X.shape == (len(man), 240), f"Saraga MFCC shape {X.shape}"
            np.save(dst, X.astype(np.float32))
            print(f"  [mfcc-saraga] {X.shape} saved")


def parse_args():
    p = argparse.ArgumentParser(description="FAPN E2 supplementary experiments, stage A: feature extraction")
    p.add_argument("--stage", default="all", choices=["all", "enc_test", "mfcc"])
    p.add_argument("--encoders", nargs="+", default=ENCODER_ORDER,
                   choices=ENCODER_ORDER)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--skip_saraga_mfcc", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fm_dev, fm_test = build_splits()

    if args.stage in ("all", "enc_test"):
        extract_encoder_test(fm_test, args.encoders, batch_size=args.batch_size)
    if args.stage in ("all", "mfcc"):
        extract_mfcc(fm_dev, fm_test, do_saraga=not args.skip_saraga_mfcc)

    print("\nStage A done. Next (CPU only): python e2_supplement_analyze.py")


if __name__ == "__main__":
    main()
