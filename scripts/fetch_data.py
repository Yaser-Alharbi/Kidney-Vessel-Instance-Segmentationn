"""Download the HuBMAP competition data into `paths.raw`.

Needs Kaggle credentials and the competition rules accepted on the
website (the API returns 403 otherwise, even when authenticated).

kaggle >= 2.x no longer reads the legacy ~/.kaggle/kaggle.json. It wants
one of: `kaggle auth login` (OAuth), the KAGGLE_API_TOKEN environment
variable, or a raw token in ~/.kaggle/access_token.

    python -m scripts.fetch_data
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from src.utils.paths import Config, load_config

COMPETITION = "hubmap-hacking-the-human-vasculature"
REQUIRED = ("polygons.jsonl", "tile_meta.csv", "train")

# Tile counts per source_wsi recorded by the v1 run, from
# artifacts/inspection_summary.json. The rerun is only comparable if the
# download matches.
EXPECTED_WSI_COUNTS = {
    1: 507, 2: 445, 3: 410, 4: 271,
    6: 600, 7: 600, 8: 600, 9: 600, 10: 600,
    11: 600, 12: 600, 13: 600, 14: 600,
}


def _has_credentials() -> bool:
    """True if the kaggle CLI has some usable credential source."""
    kdir = Path.home() / ".kaggle"
    return bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (kdir / "access_token").exists()
        or (kdir / "kaggle.json").exists()
    )


def download(cfg: Config) -> Path:
    """Fetch and unzip the competition archive into `paths.raw`."""
    raw = cfg.paths.raw
    raw.mkdir(parents=True, exist_ok=True)

    if not _has_credentials():
        raise FileNotFoundError(
            "No Kaggle credentials found. Run `kaggle auth login`, or set "
            "KAGGLE_API_TOKEN, or save a token to ~/.kaggle/access_token."
        )

    if (raw / "polygons.jsonl").exists() and (raw / "tile_meta.csv").exists():
        print(f"[fetch] raw data already present at {raw}; skipping download")
        return raw

    # kaggle >= 2.x dropped --unzip and takes the competition positionally.
    cmd = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        COMPETITION, "-p", str(raw),
    ]
    print(f"[fetch] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed (exit {proc.returncode}). If this is a 403, "
            f"accept the rules at kaggle.com/c/{COMPETITION}/rules first."
        )

    archives = sorted(raw.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"no .zip downloaded into {raw}")
    for zpath in archives:
        print(f"[fetch] unzip {zpath.name}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(raw)
        zpath.unlink()
        print(f"[fetch] removed {zpath.name}")
    return raw


def verify(cfg: Config) -> None:
    """Check the required files exist and the tile census matches v1."""
    raw = cfg.paths.raw
    missing = [n for n in REQUIRED if not (raw / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {raw}: {missing}")

    import pandas as pd

    df = pd.read_csv(raw / "tile_meta.csv")
    counts = df["source_wsi"].value_counts().sort_index().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}

    print(f"[fetch] total tiles        : {len(df)}")
    print(f"[fetch] source_wsi counts  : {counts}")
    print(f"[fetch] dataset counts     : "
          f"{df['dataset'].value_counts().sort_index().to_dict()}")

    if counts != EXPECTED_WSI_COUNTS:
        diff = {
            k: (EXPECTED_WSI_COUNTS.get(k), counts.get(k))
            for k in set(EXPECTED_WSI_COUNTS) | set(counts)
            if EXPECTED_WSI_COUNTS.get(k) != counts.get(k)
        }
        raise RuntimeError(
            "tile_meta.csv census does not match the v1 run "
            f"(expected, got): {diff}. The rerun would not be comparable."
        )
    print("[fetch] census matches the v1 run")


def main() -> None:
    """Download then verify the raw competition data."""
    cfg = load_config()
    download(cfg)
    verify(cfg)


if __name__ == "__main__":
    main()
