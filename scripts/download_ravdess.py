"""Download and extract the RAVDESS speech-only corpus.

The dataset is hosted on Zenodo (Livingstone & Russo, 2018) at
    https://doi.org/10.5281/zenodo.1188976

Usage
-----
    python scripts/download_ravdess.py --dest ./RAVDESS

The script downloads ``Audio_Speech_Actors_01-24.zip`` (~200 MB), verifies
its integrity, extracts it into ``--dest`` and prints the resulting folder
tree. After this step, export ``RAVDESS_ROOT`` and run any experiment::

    export RAVDESS_ROOT="$(pwd)/RAVDESS"       # Linux / macOS
    $env:RAVDESS_ROOT = "$(Get-Location)\RAVDESS"   # Windows PowerShell
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"


def _download(url: str, out: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, out.open("wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1 << 20  # 1 MiB
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            read += len(data)
            if total:
                pct = 100 * read / total
                print(f"  {read / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct:5.1f}%)", end="\r")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("./RAVDESS"),
                        help="Destination directory (default: ./RAVDESS)")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Do not delete the archive after extraction")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    zip_path = args.dest / "Audio_Speech_Actors_01-24.zip"

    if not zip_path.exists():
        _download(URL, zip_path)
    else:
        print(f"Archive already present: {zip_path}")

    print(f"Extracting to {args.dest} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.dest)

    if not args.keep_zip:
        zip_path.unlink()

    actors = sorted(p for p in args.dest.iterdir() if p.is_dir() and p.name.startswith("Actor_"))
    wavs = sum(len(list(a.glob("*.wav"))) for a in actors)
    print(f"Done. {len(actors)} actors, {wavs} wav files under {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
