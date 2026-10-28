# -*- coding: utf-8 -*-
"""
fetch_supp_datasets.py -- download script for the FAPN supplementary datasets

Background: the existing Saraga probe (106 Hindustani ensemble recordings) can only serve as a negative control, for two reasons:
  (a) the recordings are ensemble mixes without stems; (b) there are no instrument-family labels.
This script collects publicly downloadable supplementary and alternative corpora that can upgrade "cross-cultural family geometry" from a negative control into a testable proposition.

Usage:
  python fetch_supp_datasets.py --list              # list every candidate (licence/size/command)
  python fetch_supp_datasets.py --check             # HEAD probe only, no download
  python fetch_supp_datasets.py --dataset saraga-carnatic
  python fetch_supp_datasets.py --all-open          # download everything open and directly downloadable
  python fetch_supp_datasets.py --dataset folkmusic --no-md5

Dependencies: aria2c (preferred, multi-connection) or curl.exe; if neither is present it falls back to urllib.
Download directory: <project root>/data/supp/<key>/

Licence note: saraga-* is CC BY-NC-SA 4.0 (non-commercial); mridangam-stroke is CC BY-NC 4.0.
          For any commercial use, switch to medley-solos-db / brid / beijing-opera-percussion / folkmusic (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = ROOT / "data" / "supp"

# ---------------------------------------------------------------- candidates
# access: open = directly downloadable; gated = needs an application or authorisation; scrape = must be scraped
DATASETS = {
    "folkmusic": dict(
        title="FolkMusic / China traditional music instrument dataset",
        url="https://zenodo.org/records/8012071/files/FolkMusic.zip?download=1",
        size_gb=5.57, md5="2cc1146dbf48dd014a0d698dd743b71d", license="CC BY 4.0",
        access="open",
        labels="15 Chinese traditional instrument classes (bowed/plucked/wind), 11,966 excerpts of 3 s, single-instrument performance",
        note="NOTE: the main corpus of this paper is itself a public dataset; the data-availability statement should point at this DOI and cite the authors.",
        cite="Li, Z., Zhou, H., Xing, S., & Wang, B. (2023). China traditional music instrument dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8012071",
    ),
    "saraga-carnatic": dict(
        title="Saraga 1.5 Carnatic (with multi-track stems)",
        url="https://zenodo.org/records/4301737/files/saraga1.5_carnatic.zip?download=1",
        size_gb=14.38, md5="e4fcd380b4f6d025964cd16aee00273d", license="CC BY-NC-SA 4.0",
        access="open",
        labels="249 concerts, 168 of them multi-track: voice / violin / mridangam / ghatam stems (approximately solo)",
        note="NOTE: first choice; the same source as the Saraga used in the paper, but with stems, so it can test whether the Hindustani negative result comes from the ensemble mix. "
             "The stems still carry bleed (quantified in the Sanidha paper), which must be stated in the method.",
        cite="Bozkurt, B., Srinivasamurthy, A., Gulati, S., & Serra, X. (2018). Saraga: research datasets of Indian Art Music (Version 1.5) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4301737",
    ),
    "saraga-hindustani": dict(
        title="Saraga 1.5 Hindustani",
        url="https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip?download=1",
        size_gb=3.83, md5="ea9ed2885ea37a1b10e42f60cf299702", license="CC BY-NC-SA 4.0",
        access="open",
        labels="108 Hindustani ensemble recordings (the paper uses 106 of them)",
        note="The original source of the Saraga probe in the paper; listed here to reproduce the data-availability statement.",
        cite="Bozkurt, B., Srinivasamurthy, A., Gulati, S., & Serra, X. (2018). Saraga: research datasets of Indian Art Music (Version 1.5) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4301737",
    ),
    "medley-solos-db": dict(
        title="Medley-solos-DB (cross-collection solo instruments)",
        url="https://zenodo.org/records/3464194/files/Medley-solos-DB.tar.gz?download=1",
        size_gb=7.38, md5="f5facf398793ef5c1f80c013afdf3e5f", license="CC BY 4.0",
        access="open",
        labels="3 s excerpts of 8 solo instrument classes (clarinet/flute/sax/trumpet=wind, violin=bowed, piano=keys, guitar=plucked, voice), "
               "training and test drawn from different collections",
        note="NOTE: second choice; all real solo recordings and cross-collection, so it turns whether family geometry needs an ensemble mix from a counterexample into a positive test.",
        cite="Lostanlen, V., Cella, C. E., Bittner, R., & Essid, S. (2019). Medley-solos-DB: a cross-collection dataset for musical instrument recognition [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3464194",
        extra_files=[
            # official clip-level metadata table (v1.2, 1.3 MB), REQUIRED by e4_supp_prepare.py
            dict(url="https://zenodo.org/records/3464194/files/Medley-solos-DB_metadata.csv?download=1",
                 rename="Medley-solos-DB_metadata.csv",
                 md5="fda6a589c56785f2195c9227809c521a"),
        ],
    ),
    "brid": dict(
        title="Brazilian Rhythmic Instruments Dataset (BRID)",
        url="https://zenodo.org/records/14051323/files/audio.zip?download=1",
        size_gb=0.88, md5="3514b53d66515181f95619adb71a59b4", license="CC BY 4.0",
        access="open",
        labels="10 Brazilian percussion classes, 367 tracks (274 solo + 93 ensemble)",
        note="Same-family control from a third tradition: all percussion (one family), so if the model still separates instruments the geometry is resolvable inside a family, "
             "which is a reverse test of the family-granularity claim.",
        cite="Maia, L. S., de Tomaz J\u00fanior, P. D., Fuentes, M., Rocamora, M., Biscainho, L. W. P., da Costa, M. V. M., & Cohen, S. (2018). Brazilian Rhythmic Instruments Dataset (BRID) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14051323",
    ),
    "beijing-opera-percussion": dict(
        title="Beijing Opera Percussion Instrument Dataset",
        url="https://zenodo.org/records/1285212/files/beijing_opera_percussion_instrument_1.0.zip?download=1",
        size_gb=0.03, md5="51dc70d02a06b9d96befadf320a6996f", license="CC BY 4.0",
        access="open",
        labels="4 Beijing opera percussion classes (bangu/naobo/daluo/xiaoluo), 236 isolated hits",
        note="Same-family control (all percussion), very small, suitable as a quick within-family invariance check.",
        cite="Tian, M., Srinivasamurthy, A., Sandler, M., & Serra, X. (2014). Beijing Opera Percussion Instrument Dataset (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1285212",
    ),
    "mridangam-stroke": dict(
        title="Mridangam Stroke Dataset",
        url="https://zenodo.org/records/4068196/files/mridangam_stroke_1.5.zip?download=1",
        size_gb=0.12, md5="39af55b2476b94c7946bec24331ec01a", license="CC BY-NC 4.0",
        access="open",
        labels="6,977 mridangam stroke samples, 10 stroke classes x 6 tonics",
        note="Same-family control + tonic invariance: different tonics within one family should not produce family structure.",
        cite="Anantapadmanabhan, A., Bellur, A., & Murthy, H. A. (2020). Mridangam Stroke Dataset (Version 1.5) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4068196",
    ),
    "chmusic": dict(
        title="ChMusic (Chinese traditional instruments, solo)",
        url=None, gdrive_id="1rfbXpkYEUGw5h_CZJtC7eayYemeFMzij",
        size_gb=None, md5=None, license="declared free by the authors (Google Drive / Baidu Netdisk)",
        access="open",
        labels="11 Chinese instrument classes, 55 solo pieces (5 per instrument), one instrument per piece",
        note="Solo and spanning the three families (bowed/plucked/wind), but only 55 pieces, so the sample is small; used as a supplementary check of whether family structure survives once the ensemble confound is removed.",
        cite="ChMusic: A Traditional Chinese Music Dataset for Evaluation of Instrument Recognition. arXiv:2108.08470",
    ),
    "sanidha": dict(
        title="Sanidha (Carnatic studio stems)",
        url=None, size_gb=None, md5=None, license="CC BY 4.0 (application required)",
        access="gated",
        labels="5 Carnatic concerts, about 8 hours, studio-isolated stems: vocal / violin / mridangam / ghatam / tanpura",
        note="NOTE: the highest-quality stem corpus, but downloading requires a Georgia Tech campus network or an e-mail application (ccml.gtcmt.gatech.edu/data/Sanidha). "
             "The script cannot download it automatically; an application e-mail has to be sent first.",
        cite="Sanidha: A Studio Quality Multi-Modal Dataset for Carnatic Music. https://ccml.gtcmt.gatech.edu/data/Sanidha",
    ),
    "tabla-solo": dict(
        title="Tabla Solo dataset",
        url="https://zenodo.org/records/1267024", size_gb=None, md5=None, license="restricted",
        access="gated",
        labels="38 solo tabla recordings (about 17 minutes)",
        note="Same-family alternative; the Zenodo record is restricted and needs author authorisation.",
        cite="Tabla Solo dataset. Zenodo. https://doi.org/10.5281/zenodo.1267024",
    ),
    "carnatic-violin": dict(
        title="Carnatic Violin Dataset",
        url="https://zenodo.org/records/3940330", size_gb=None, md5=None, license="restricted",
        access="gated",
        labels="Carnatic violin solos + pitch annotations",
        note="A single instrument (bowed), so it can only serve as a single-family control; the Zenodo record is restricted.",
        cite="Subramani, K., & Rao, P. (2020). Carnatic Violin Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3940330",
    ),
    "ccmusic": dict(
        title="CCMusic (10 sub-corpora including guzheng and erhu techniques)",
        url="https://ccmusic-database.github.io/en/download.html", size_gb=None, md5=None,
        license="free but requires an application form",
        access="gated",
        labels="Chinese instrument material, melody and accompaniment stem recordings; includes GZ_IsoTech (2,824 guzheng excerpts) and ErhuPT (erhu techniques)",
        note="An application form has to be sent to ccmusic.database@hotmail.com; the script cannot download it automatically.",
        cite="Liu, Z., & Li, Z. (2021). Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5676893",
    ),
    "lyra": dict(
        title="Lyra (Greek traditional music, metadata + YouTube links)",
        url=None, size_gb=None, md5=None, license="metadata open; audio must be scraped",
        access="scrape",
        labels="1,570 Greek traditional and folk pieces with instrumentation metadata and YouTube timestamp links",
        note="A Mediterranean tradition covering bowed/plucked/wind/percussion, so the family distribution is ideal; but the audio must be scraped with yt-dlp, "
             "which is subject to copyright and link rot, so reproducibility is weaker than for the Zenodo direct links.",
        cite="Lyra: A Dataset of Greek Traditional Music. Zenodo/GitHub (metadata) + YouTube (audio)",
    ),
}

OPEN_DIRECT = [k for k, v in DATASETS.items() if v["access"] == "open" and v.get("url")]


# ---------------------------------------------------------------- helpers
def human(n: int | float | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def has_aria2() -> bool:
    return shutil.which("aria2c") is not None


def download(url: str, dst_dir: Path, name: str | None = None) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    fname = name or url.split("?")[0].rstrip("/").split("/")[-1]
    out = dst_dir / fname
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] already present {out}")
        return out
    if has_aria2():
        print(f"[aria2c] -> {out}")
        cmd = ["aria2c", "-x", "8", "-s", "8", "-k", "1M", "--file-allocation=none",
               "-c", "-d", str(dst_dir), "-o", fname, url]
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0 and not out.exists():
            print(f"[warn] aria2c returned {r.returncode}")
    else:
        print(f"[urllib] -> {out} (aria2c not detected, a single connection is slower)")
        urllib.request.urlretrieve(url, out)
    return out


def probe(url: str) -> str:
    if shutil.which("curl"):
        r = subprocess.run(["curl.exe", "-sIL", "--max-time", "30", url],
                           capture_output=True, text=True, check=False)
        lines = [x.strip() for x in r.stdout.splitlines()
                 if x.lower().startswith(("http/", "content-length", "location:"))]
        return " | ".join(lines[-4:]) or "no response"
    return "curl unavailable"


# ---------------------------------------------------------------- subcommands
def cmd_list() -> None:
    print(f"\n{'key':<26}{'access':<9}{'size':<10}{'license':<20}title")
    print("-" * 118)
    for k, v in DATASETS.items():
        print(f"{k:<26}{v['access']:<9}{(str(v['size_gb']) + ' GB') if v['size_gb'] else '?':<10}"
              f"{v['license']:<20}{v['title']}")
    print("\nDirectly downloadable (open with a direct link):", ", ".join(OPEN_DIRECT))
    print("\nCommon commands:")
    print("  python code\\fetch_supp_datasets.py --check")
    print("  python code\\fetch_supp_datasets.py --dataset saraga-carnatic")
    print("  python code\\fetch_supp_datasets.py --all-open")
    print("\nEquivalent aria2c direct-link commands (pick one):")
    for k in OPEN_DIRECT:
        v = DATASETS[k]
        print(f"  aria2c -x8 -s8 -c -d data\\supp\\{k} -o {v['url'].split('?')[0].split('/')[-1]} \"{v['url']}\"")
    print("\nChMusic (Google Drive, needs gdown or a browser):")
    print("  pip install gdown && gdown 1rfbXpkYEUGw5h_CZJtC7eayYemeFMzij -O data\\supp\\chmusic\\ChMusic.zip")


def cmd_check() -> None:
    for k, v in DATASETS.items():
        if v.get("url") and v["access"] == "open":
            print(f"[{k}] {probe(v['url'])}")
        else:
            print(f"[{k}] not a direct link ({v['access']}), skipping the probe")


def cmd_fetch(key: str, verify_md5: bool = True) -> None:
    v = DATASETS[key]
    if not v.get("url"):
        print(f"[{key}] no direct link, access={v['access']}. Note: {v['note']}")
        if v.get("gdrive_id"):
            print(f"  gdown {v['gdrive_id']} -O data\\supp\\{key}\\ChMusic.zip")
        return
    out = download(v["url"], DEST_ROOT / key)
    for ex in v.get("extra_files", []):
        ex_out = download(ex["url"], DEST_ROOT / key, name=ex.get("rename"))
        if verify_md5 and ex.get("md5"):
            g = md5_of(ex_out)
            print(f"[md5:{ex_out.name}] {g} " + ("OK" if g == ex["md5"] else "MISMATCH expected " + ex["md5"]))
    if verify_md5 and v.get("md5"):
        got = md5_of(out)
        ok = got == v["md5"]
        print(f"[md5] {got} {'OK' if ok else 'MISMATCH expected ' + v['md5']}")
    print(f"[{key}] license={v['license']}\n  cite: {v['cite']}\n  note: {v['note']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FAPN supplementary dataset download")
    ap.add_argument("--list", action="store_true", help="list every candidate and command")
    ap.add_argument("--check", action="store_true", help="HEAD probe availability")
    ap.add_argument("--dataset", help="download the given key")
    ap.add_argument("--all-open", action="store_true", help="download every open direct-link dataset")
    ap.add_argument("--no-md5", action="store_true", help="skip the md5 check")
    a = ap.parse_args()

    if a.list or not any([a.check, a.dataset, a.all_open]):
        cmd_list()
    if a.check:
        cmd_check()
    if a.dataset:
        if a.dataset not in DATASETS:
            print(f"unknown key: {a.dataset}\navailable: {', '.join(DATASETS)}")
            sys.exit(1)
        cmd_fetch(a.dataset, verify_md5=not a.no_md5)
    if a.all_open:
        total = sum(DATASETS[k]["size_gb"] or 0 for k in OPEN_DIRECT)
        print(f"[all-open] {len(OPEN_DIRECT)} datasets, about {total:.1f} GB in total")
        for k in OPEN_DIRECT:
            cmd_fetch(k, verify_md5=not a.no_md5)


if __name__ == "__main__":
    main()
