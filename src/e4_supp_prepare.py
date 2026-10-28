# -*- coding: utf-8 -*-
"""
Preparation and feature extraction for the supplementary control corpora (routes B/C).

Four corpora (all CC BY or CC BY-NC, publicly downloadable):
  1. medley  : Medley-solos-DB (3 s solo excerpts, 8 instrument classes; the official split takes training/validation from MedleyDB
               and test from solosDB, so the official split is itself cross-collection)
  2. brid    : BRID Brazilian percussion (solo tracks, 10 percussion instruments x 3 performers)
  3. beijing : Beijing opera percussion (4 percussion instruments)
  4. mridangam: Mridangam Stroke 1.5 (one instrument x 6 tonics x 10 strokes)

Feature protocol identical to the paper: the four encoders go through encoders.load_audio_mono (default max_sec=20 s),
MFCC uses 16 kHz / max_sec=10 s / n_mfcc=40 + delta + delta-delta frame mean and std (240 dims).

Output:
  data/supp/{corpus}/extracted/...            audio
  results/e4/supp/{corpus}_manifest.csv       manifest (with labels and split)
  results/e4/supp/{feat}_{corpus}.npy         features + _ok.npy
"""
import json
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))

SUPP = PROJ / "data" / "supp"
OUT = PROJ / "results" / "e4" / "supp"
ENCODERS = ["beats", "mert", "ast", "audiomae"]
MFCC_SR = 16000
MFCC_MAX_SEC = 10.0
N_MFCC = 40
SEED = 20260501

MEDLEY_PER_CLASS = 400
MRIDANGAM_PER_TONIC = 200

# corpus -> the actual directory name under data/supp (the download directory name; do not change)
DATA_DIR = {"medley": "medley-solos-db", "brid": "brid",
            "beijing": "beijing-opera-percussion", "mridangam": "mridangam-stroke"}

# Medley-solos-DB official classes (the Zenodo record 3464194 description agrees with the mirdata source)
MEDLEY_INSTR = {0: "clarinet", 1: "distorted electric guitar", 2: "female singer",
                3: "flute", 4: "piano", 5: "tenor saxophone", 6: "trumpet", 7: "violin"}
# Hornbostel-Sachs excitation mapped onto the three families of the paper; piano (keyboard) and voice are outside the three families and only serve as class-level reference
MEDLEY_FAMILY = {"clarinet": "wind", "flute": "wind", "tenor saxophone": "wind",
                 "trumpet": "wind", "violin": "bowed",
                 "distorted electric guitar": "plucked"}


# ============================================================
# manifest
# ============================================================
def manifest_medley():
    csv = SUPP / "medley-solos-db" / "Medley-solos-DB_metadata.csv"
    df = pd.read_csv(csv)
    # authoritative check: the instrument string shipped in the csv must match the official id mapping row by row
    assert (df["instrument"] == df["instrument_id"].map(MEDLEY_INSTR)).all(), \
        "instrument string disagrees with the official id mapping"
    rng = np.random.default_rng(SEED)
    keep = []
    for inst, g in df.groupby("instrument_id"):
        g = g.copy()
        # rotate the sampling by song_id so fragments of one track do not fill the quota
        songs = g["song_id"].drop_duplicates().to_numpy()
        rng.shuffle(songs)
        order = {s: i for i, s in enumerate(songs)}
        g["_ord"] = g["song_id"].map(order)
        g = g.sort_values(["_ord"])
        take = g.groupby("song_id", sort=False).head(1)
        if len(take) < MEDLEY_PER_CLASS:
            rest = g.drop(take.index).sample(frac=1, random_state=SEED)
            take = pd.concat([take, rest.head(MEDLEY_PER_CLASS - len(take))])
        keep.append(take.head(MEDLEY_PER_CLASS))
    m = pd.concat(keep).reset_index(drop=True)
    m["file_name"] = ("Medley-solos-DB_" + m["subset"] + "-" + m["instrument_id"].astype(str)
                      + "_" + m["uuid4"] + ".wav")
    m["rel_path"] = m["file_name"]
    m["class_label"] = m["instrument"]
    m["family"] = m["instrument"].map(MEDLEY_FAMILY).fillna("other")
    m["group"] = m["song_id"].astype(str)      # fragments of one track are near-duplicates of each other
    m["split"] = m["subset"]
    m = m[["file_name", "rel_path", "class_label", "family", "group", "split",
           "instrument_id", "song_id", "uuid4"]]
    return m, "medley-solos-db/Medley-solos-DB.tar.gz", "medley"


def manifest_brid():
    z = zipfile.ZipFile(SUPP / "brid" / "audio.zip")
    rows = []
    for n in z.namelist():
        if not (n.startswith("Data/Solo Tracks") and n.endswith(".wav")):
            continue
        parts = n.split("/")
        rows.append({"rel_path": n, "player": parts[2], "class_label": parts[3],
                     "group": parts[2], "split": "all",
                     "file_name": parts[-1]})
    m = pd.DataFrame(rows)
    m["family"] = "percussion"
    m = m[["file_name", "rel_path", "class_label", "family", "group", "split", "player"]]
    return m, "brid/audio.zip", "brid"


def manifest_beijing():
    z = zipfile.ZipFile(SUPP / "beijing-opera-percussion"
                        / "beijing_opera_percussion_instrument_1.0.zip")
    rows = []
    for n in z.namelist():
        if not n.endswith(".wav"):
            continue
        base = n.split("/")[-1]
        inst = re.sub(r"^\d+__\w+?__", "", base).rsplit("-", 1)[0]
        rows.append({"rel_path": n, "file_name": base, "class_label": inst,
                     "group": "single-session", "split": "all"})
    m = pd.DataFrame(rows)
    m["family"] = "percussion"
    m = m[["file_name", "rel_path", "class_label", "family", "group", "split"]]
    return m, "beijing-opera-percussion/beijing_opera_percussion_instrument_1.0.zip", "beijing"


def manifest_mridangam():
    z = zipfile.ZipFile(SUPP / "mridangam-stroke" / "mridangam_stroke_1.5.zip")
    rows = []
    for n in z.namelist():
        if not n.endswith(".wav") or n.startswith("__MACOSX"):
            continue
        parts = n.split("/")
        tonic, base = parts[1], parts[-1]
        stroke = base.split("__")[-1].split("-")[0]
        rows.append({"rel_path": n, "file_name": base, "class_label": stroke,
                     "tonic": tonic, "group": tonic, "split": "all"})
    m = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    keep = []
    for tonic, g in m.groupby("tonic"):
        g = g.copy().sample(frac=1, random_state=SEED)
        # rotate by stroke so the stroke distribution stays balanced inside each tonic
        g["_o"] = g.groupby("class_label").cumcount()
        g = g.sort_values(["_o"])
        keep.append(g.head(MRIDANGAM_PER_TONIC))
    m = pd.concat(keep).reset_index(drop=True)
    m["family"] = "percussion"
    m = m[["file_name", "rel_path", "class_label", "family", "group", "split", "tonic"]]
    return m, "mridangam-stroke/mridangam_stroke_1.5.zip", "mridangam"


# ============================================================
# extract the audio
# ============================================================
def extract_medley(m, tarpath):
    dst = SUPP / DATA_DIR["medley"] / "extracted"
    dst.mkdir(parents=True, exist_ok=True)
    need = set(m["rel_path"])
    missing = {n for n in need if not (dst / n).exists()}
    print(f"[unzip] medley needs {len(missing)}/{len(need)} extracted")
    if not missing:
        return
    t0 = time.time()
    done = 0
    with tarfile.open(tarpath, "r:gz") as t:
        for mem in t:
            base = mem.name.split("/")[-1]
            if base in missing and mem.isfile() and not base.startswith("._"):
                with t.extractfile(mem) as src, open(dst / base, "wb") as fh:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        fh.write(b)
                done += 1
                missing.discard(base)
                if done % 400 == 0:
                    print(f"   {done}  {time.time()-t0:.0f}s")
            if not missing:
                break
    print(f"[unzip] medley done {done}, missing {len(missing)}  {time.time()-t0:.0f}s")


def extract_zip(m, zippath, corpus):
    dst = SUPP / DATA_DIR[corpus] / "extracted"
    dst.mkdir(parents=True, exist_ok=True)
    need = list(m["rel_path"])
    missing = [n for n in need if not (dst / n.split("/")[-1]).exists()]
    print(f"[unzip] {corpus} needs {len(missing)}/{len(need)} extracted")
    if not missing:
        return
    z = zipfile.ZipFile(zippath)
    t0 = time.time()
    for i, n in enumerate(missing, 1):
        target = dst / n.split("/")[-1]
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(n) as src, open(target, "wb") as fh:
            while True:
                b = src.read(1 << 20)
                if not b:
                    break
                fh.write(b)
        if i % 300 == 0:
            print(f"   {i}/{len(missing)}  {time.time()-t0:.0f}s")
    print(f"[unzip] {corpus} done {time.time()-t0:.0f}s")


# ============================================================
# features
# ============================================================
def mfcc_vector(path):
    from encoders import load_audio_mono
    import librosa
    y = load_audio_mono(path, MFCC_SR, max_sec=MFCC_MAX_SEC)
    M = librosa.feature.mfcc(y=y, sr=MFCC_SR, n_mfcc=N_MFCC, n_fft=1024, hop_length=160)
    d1 = librosa.feature.delta(M)
    d2 = librosa.feature.delta(M, order=2)
    return np.concatenate([M.mean(1), M.std(1), d1.mean(1), d1.std(1),
                           d2.mean(1), d2.std(1)]).astype(np.float32)


def extract_features(corpus, paths):
    OUT.mkdir(parents=True, exist_ok=True)
    from encoders import create_encoder
    for name in ENCODERS:
        f, mk = OUT / f"{name}_{corpus}.npy", OUT / f"{name}_{corpus}_ok.npy"
        if f.exists() and mk.exists():
            print(f"[enc] {corpus}/{name}: cached {np.load(f).shape}")
            continue
        print(f"\n[enc] {corpus}/{name}: {len(paths)} files")
        t0 = time.time()
        enc = create_encoder(name)
        r = enc.encode_dataset(paths, batch_size=None, log_every=500)
        ok = np.zeros(len(paths), dtype=bool)
        ok[np.asarray(r["ok_idx"], dtype=int)] = True
        np.save(f, r["embeddings"])
        np.save(mk, ok)
        print(f"[enc] {corpus}/{name} {r['embeddings'].shape} ok={ok.sum()}/{len(paths)} "
              f"failed={len(r['failed'])} {time.time()-t0:.0f}s")
        del enc
    f, mk = OUT / f"mfcc_{corpus}.npy", OUT / f"mfcc_{corpus}_ok.npy"
    if f.exists() and mk.exists():
        print(f"[mfcc] {corpus}: cached {np.load(f).shape}")
        return
    t0 = time.time()
    rows, ok = [], np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        try:
            rows.append(mfcc_vector(p))
            ok[i] = True
        except Exception as ex:
            print(f"   [warn] mfcc fail {p}: {ex!r}")
            rows.append(np.full(6 * N_MFCC, np.nan, dtype=np.float32))
        if (i + 1) % 500 == 0:
            print(f"   mfcc {i+1}/{len(paths)}  {time.time()-t0:.0f}s")
    np.save(f, np.stack(rows))
    np.save(mk, ok)
    print(f"[mfcc] {corpus} {np.stack(rows).shape} ok={ok.sum()}/{len(ok)} {time.time()-t0:.0f}s")


def main():
    jobs = [("medley", manifest_medley, extract_medley),
            ("brid", manifest_brid, extract_zip),
            ("beijing", manifest_beijing, extract_zip),
            ("mridangam", manifest_mridangam, extract_zip)]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for corpus, mk, ex in jobs:
        mpath = OUT / f"{corpus}_manifest.csv"
        if mpath.exists():
            m = pd.read_csv(mpath, encoding="utf-8-sig")
            zpath = None
        else:
            # when the manifest is frozen (the normal case) no archive is needed and all audio is read from extracted/;
            # the archive is only required when the manifest is also missing and has to be rebuilt from scratch.
            try:
                m, rel_zip, _ = mk()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{corpus}: the archive is not local and the manifest {mpath} does not exist; "
                    "re-download with src/fetch_supp_datasets.py") from e
            m.to_csv(mpath, index=False, encoding="utf-8-sig")
            zpath = SUPP / rel_zip
            if corpus == "medley":
                ex(m, zpath)
            else:
                ex(m, zpath, corpus)
        dst = SUPP / DATA_DIR[corpus] / "extracted"
        bases = [p.split("/")[-1] for p in m["rel_path"]]
        assert len(set(bases)) == len(bases), f"{corpus}: duplicate extracted file names, paths would be misaligned"
        paths = [str(dst / b) for b in bases]
        missing = [p for p in paths if not Path(p).exists()]
        assert not missing, f"{corpus}: {len(missing)} audio files were not extracted, e.g. {missing[:2]}"
        extract_features(corpus, paths)
        summary[corpus] = {
            "n_clips": int(len(m)),
            "classes": int(m.class_label.nunique()),
            "class_counts": m.class_label.value_counts().to_dict(),
            "families": m.family.value_counts().to_dict(),
            "groups": int(m.group.nunique()),
            "splits": m.split.value_counts().to_dict(),
        }
        print(f"[summary] {corpus}: {json.dumps(summary[corpus], ensure_ascii=False)}")
    (OUT / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {OUT / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
