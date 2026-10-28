# -*- coding: utf-8 -*-
"""
Generate Saraga 1.5 Hindustani manifest.

Reads every Raag-level directory under
  data/openethnic/saraga/hindustani/<performance>/<raag>/
  - <raag>.mp3.mp3   -> audio path
  - <raag>.json      -> structured metadata (raag / artists / release / length)
and writes:
  data/openethnic/saraga/manifest.csv   (UTF-8 with BOM)

Fields:
  recording_id, performance, raag_common, raag_iaST, lead_artist, album_artists,
  length_ms, duration_s, sr (placeholder, back-filled once probed), audio_path, json_path
"""
from __future__ import annotations
import json
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data" / "openethnic" / "saraga" / "hindustani"
OUT  = ROOT.parent / "manifest.csv"

rows = []
for performance_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    perf = performance_dir.name
    for raag_dir in sorted(d for d in performance_dir.iterdir() if d.is_dir()):
        j = next(raag_dir.glob("*.json"), None)
        mp3_candidates = list(raag_dir.glob("*.mp3.mp3")) or list(raag_dir.glob("*.mp3"))
        if j is None or not mp3_candidates:
            continue
        try:
            meta = json.loads(j.read_text(encoding="utf-8"))
        except Exception as ex:
            print(f"[skip] {j}: {ex}")
            continue

        raags = meta.get("raags") or [{}]
        r0 = raags[0]
        raag_common = (r0.get("common_name") or "").strip() or raag_dir.name
        raag_iaST    = (r0.get("name") or "").strip()

        artists = meta.get("artists") or []
        lead = next((a["artist"]["name"] for a in artists if a.get("lead")),
                    (artists[0]["artist"]["name"] if artists else ""))
        album_artists = "; ".join(a["name"] for a in (meta.get("album_artists") or []) if a.get("name"))

        length_ms = int(meta.get("length") or 0)
        audio_path = mp3_candidates[0]
        # normalise to a path relative to ROOT so the manifest stays portable
        rel_audio = str(audio_path.relative_to(ROOT)).replace("\\", "/")

        rows.append({
            "recording_id":  f"{perf}__{raag_dir.name}",
            "performance":   perf,
            "raag_common":   raag_common,
            "raag_iaST":     raag_iaST,
            "lead_artist":   lead,
            "album_artists": album_artists,
            "length_ms":     length_ms,
            "duration_s":    round(length_ms / 1000.0, 2),
            "audio_relpath": rel_audio,
            "json_path":     str(j.relative_to(ROOT)).replace("\\", "/"),
        })

# write the CSV: UTF-8 with BOM
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"wrote {len(rows)} rows -> {OUT}")
print(f"unique performances: {len({r['performance'] for r in rows})}")
print(f"unique lead artists: {len({r['lead_artist'] for r in rows})}")
print(f"unique raag_common:  {len({r['raag_common'] for r in rows})}")

# per-artist statistics
from collections import Counter
print("\n--- lead artist counts ---")
for a, n in Counter(r["lead_artist"] for r in rows).most_common():
    print(f"  {n:3d}  {a}")
