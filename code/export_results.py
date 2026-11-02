"""
Export all run results to portable CSV files for the public repository
=====================================================================
The experiment scripts store one pickle per (dataset, seed) job in <folder>/parts/.
Pickles are Python-specific and unsafe to load from untrusted sources, so the archived
copy uses compressed CSV instead.

Usage (from the folder that contains results, results_v2, results_v3, ...):
    python export_results.py                     # every folder that has a parts/ subfolder
    python export_results.py --dest ../results   # where to write (default: ../results)
    python export_results.py --only results_v4b results_v5b   # only these folders

For each results folder it writes  <dest>/<folder>/<table>.csv.gz  (one file per record type,
e.g. downstream, detect, mech, overlap, rows, precision) and copies the summary_*.md and *.csv
tables already produced by the analysis scripts. Run logs and timing are kept in log.csv.gz.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=".")
ap.add_argument("--dest", default="../results")
ap.add_argument("--only", nargs="*", default=None, help="export only these folder names")
args = ap.parse_args()

src, dest = Path(args.src), Path(args.dest)
folders = sorted(p.parent for p in src.glob("*/parts") if p.is_dir())
folders = [f for f in folders if "smoke" not in f.name]
if args.only:
    folders = [f for f in folders if f.name in set(args.only)]
if not folders:
    raise SystemExit(f"No */parts folders found under {src.resolve()}")

for folder in folders:
    parts = sorted((folder / "parts").glob("*.pkl"))
    tables, logs = {}, []
    for f in parts:
        r = pd.read_pickle(f)
        for k, v in r.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                tables.setdefault(k, []).extend(v)
            elif k == "log" and isinstance(v, list):
                logs += [dict(job=f.stem, message=m) for m in v]
        logs.append(dict(job=f.stem, message=f"seconds={r.get('seconds', '')}"))
    out = dest / folder.name
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{k}.csv.gz", index=False)
    pd.DataFrame(logs).to_csv(out / "log.csv.gz", index=False)
    copied = 0
    for pattern in ("summary*.md", "*.csv", "failures.log", "*.png"):
        for f in folder.glob(pattern):
            shutil.copy2(f, out / f.name)
            copied += 1
    print(f"{folder.name}: {len(parts)} jobs -> {', '.join(f'{k} ({len(v)} rows)' for k, v in tables.items())}; "
          f"{copied} summary files copied")
print(f"\nDone. Exported to {dest.resolve()}")
