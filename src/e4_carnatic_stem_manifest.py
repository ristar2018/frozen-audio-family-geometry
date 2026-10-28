# -*- coding: utf-8 -*-
"""
Route A (supplementary experiment) step one: the Saraga 1.5 Carnatic multi-track stem manifest.

Purpose: build the manifest for the "ensemble mix vs solo stem" paired experiment (no audio decoding; only the directory and json of the archive or of the extracted copy).
Output: results/e4/carnatic_stems_manifest.csv + results/e4/carnatic_stems_summary.json

Design notes (the parts shared with the existing Saraga probe of the paper):
- statistics of the existing Saraga probe: same-rag cross-performer similarity, rag effect / performer effect (bootstrap CI),
  rag silhouette, rag retrieval R@1. The Carnatic stems give per-item violin/vocal/percussion stems,
  so a paired design becomes possible: the ensemble mix and the solo stems of the same item, asking whether the mix masks the statistics.
- honest boundary: the Carnatic stems only cover violin (bowed) + vocal + mridangam/ghatam (percussion),
  with no wind or plucked melodic stem, so the 3-family (wind/bowed/plucked) test of the paper cannot be reproduced.
"""
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "supp" / "saraga-carnatic" / "saraga1.5_carnatic.zip"
EXTRACTED = ROOT / "data" / "supp" / "saraga-carnatic" / "extracted"
OUT = ROOT / "results" / "e4"
OUT.mkdir(parents=True, exist_ok=True)

STEM_RE = re.compile(r"\.multitrack-([a-z\-]+)\.mp3$")
ITEM_RE = re.compile(r"^saraga1\.5_carnatic/([^/]+)/([^/]+)/")


def open_source():
    """Return (list of member names, function that reads bytes).

    Prefer the archive; fall back to the extracted copy when the archive is not local (extracted/ mirrors the archive layout)
    and both paths produce an identical manifest. To re-download the archive see src/fetch_supp_datasets.py.
    """
    if ZIP.exists():
        z = zipfile.ZipFile(ZIP)
        names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
        return names, z.read
    if not EXTRACTED.is_dir():
        raise FileNotFoundError(
            f"neither the archive nor an extracted copy exists: {ZIP} / {EXTRACTED}; "
            "re-download and extract with src/fetch_supp_datasets.py")
    names = [p.relative_to(EXTRACTED).as_posix()
             for p in sorted(EXTRACTED.rglob("*")) if p.is_file()]
    return names, lambda n: (EXTRACTED / n).read_bytes()


def _label_names(entry):
    """The raaga/taala/form fields may hold a str or a dict (unstable key names), so parse defensively."""
    if entry is None:
        return []
    out = []
    for e in entry:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, dict):
            for k in ("title", "name", "common_name", "value"):
                if k in e and isinstance(e[k], str):
                    out.append(e[k])
                    break
            else:
                out.append(json.dumps(e, ensure_ascii=False))
    return out


def main():
    names, read_bytes = open_source()
    items = defaultdict(list)
    for n in names:
        m = ITEM_RE.match(n)
        if m:
            items[(m.group(1), m.group(2))].append(n)

    rows = []
    for (concert, item), files in sorted(items.items()):
        files = sorted(files)
        stems = {}
        for f in files:
            m = STEM_RE.search(f)
            if m:
                stems[m.group(1)] = f
        mix = [f for f in files if f.endswith(".mp3") and ".multitrack-" not in f]
        js = [f for f in files if f.endswith(".json")]
        meta = json.loads(read_bytes(js[0]).decode("utf-8")) if js else {}
        artists = meta.get("artists") or []
        lead = ""
        if artists and isinstance(artists[0], dict):
            lead = (artists[0].get("artist") or {}).get("name", "")
        rows.append({
            "concert": concert,
            "item": item,
            "mbid": meta.get("mbid", ""),
            "duration_s": round((meta.get("length") or 0) / 1000.0, 1),
            "lead_artist": lead,
            "raaga": "|".join(_label_names(meta.get("raaga"))),
            "taala": "|".join(_label_names(meta.get("taala"))),
            "form": "|".join(_label_names(meta.get("form"))),
            "n_stems": len(stems),
            "stems": "|".join(sorted(stems)),
            "has_violin": int("violin" in stems),
            "has_vocal": int("vocal" in stems),
            "has_percussion": int(any(s.startswith("mridangam") or s == "ghatam" for s in stems)),
            "has_mix": int(bool(mix)),
            "mix_path": mix[0] if mix else "",
            "violin_path": stems.get("violin", ""),
            "vocal_path": stems.get("vocal", ""),
            "ghatam_path": stems.get("ghatam", ""),
            "mridangam_l_path": stems.get("mridangam-left", ""),
            "mridangam_r_path": stems.get("mridangam-right", ""),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "carnatic_stems_manifest.csv", index=False, encoding="utf-8-sig")

    paired = df[(df.has_violin == 1) & (df.has_mix == 1)]
    raga_ok = paired[paired.raaga != ""]
    multi = raga_ok.groupby("raaga")["lead_artist"].nunique()
    summary = {
        "zip": str(ZIP),
        "items_total": int(len(df)),
        "items_with_stems": int((df.n_stems > 0).sum()),
        "items_with_violin_and_mix": int(len(paired)),
        "items_paired_with_raga": int(len(raga_ok)),
        "distinct_raaga_paired": int(raga_ok.raaga.nunique()),
        "distinct_artists_paired": int(raga_ok.lead_artist.nunique()),
        "raaga_with_multiple_artists": int((multi >= 2).sum()),
        "paired_audio_hours_mix": round(paired.duration_s.sum() / 3600, 2),
        "paired_audio_hours_violin": round(paired.duration_s.sum() / 3600, 2),
        "paired_audio_hours_vocal": round(
            df[(df.has_vocal == 1) & (df.has_mix == 1)].duration_s.sum() / 3600, 2),
        "stem_set_counts": {str(k): int(v) for k, v in
                            Counter(df[df.n_stems > 0].stems).items()},
        "note": ("the Carnatic stems only contain violin (bowed)/vocal/mridangam/ghatam; "
                 "there is no wind or plucked melodic stem, so the 3-family test of the paper cannot be reproduced."),
    }
    (OUT / "carnatic_stems_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n--- top raaga among paired items ---")
    print(raga_ok.raaga.value_counts().head(10).to_string())
    print("\n[top raaga × distinct artists]")
    for r, g in raga_ok.groupby("raaga"):
        if g.lead_artist.nunique() >= 2:
            print(f"  {r}: {len(g)} items, {g.lead_artist.nunique()} artists")
    print(f"\n[saved] {OUT / 'carnatic_stems_manifest.csv'}")
    print(f"[saved] {OUT / 'carnatic_stems_summary.json'}")


if __name__ == "__main__":
    main()
