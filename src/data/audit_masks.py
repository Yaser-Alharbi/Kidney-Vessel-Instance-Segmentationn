"""Mask coverage audit.

Ranks every mask by vessel pixel fraction and writes:
    data/processed/mask_coverage.csv
    results/mask_audit_top.png
    results/mask_audit_bottom.png

Used to eyeball outliers (tiles with very high or very low coverage)
for labelling errors. Run on demand:

    python -m src.data.audit_masks
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.paths import Config, ensure_dirs, load_config


def _save_grid(
    cfg: Config,
    tile_ids: List[str],
    coverages: List[float],
    out_path: Path,
    title: str,
) -> None:
    n = min(len(tile_ids), 8)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols, 2.2 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax_img = axes[r, 2 * c]
        ax_msk = axes[r, 2 * c + 1]
        ax_img.axis("off")
        ax_msk.axis("off")
        if i >= n:
            continue
        tid = tile_ids[i]
        cov = coverages[i]
        img_p = cfg.paths.raw / "train" / f"{tid}.tif"
        msk_p = cfg.paths.masks / f"{tid}.png"
        if img_p.exists():
            ax_img.imshow(np.array(Image.open(img_p).convert("RGB")))
        if msk_p.exists():
            ax_msk.imshow(
                cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE), cmap="gray", vmin=0, vmax=1
            )
        ax_img.set_title(f"{tid}\ncov={cov:.3f}", fontsize=8)
        ax_msk.set_title("mask", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def audit_coverage(cfg: Config, top_k: int = 8) -> pd.DataFrame:
    ensure_dirs(cfg)
    rows = []
    for p in sorted(cfg.paths.masks.glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        rows.append({"tile_id": p.stem, "coverage": float(m.mean())})
    if not rows:
        print(f"[audit] no masks found under {cfg.paths.masks}")
        return pd.DataFrame(columns=["tile_id", "coverage"])

    df = pd.DataFrame(rows).sort_values("coverage", ascending=False).reset_index(drop=True)
    csv_path = cfg.paths.processed / "mask_coverage.csv"
    df.to_csv(csv_path, index=False)
    print(f"[audit] coverage csv saved : {csv_path}")
    print(f"[audit] tiles audited      : {len(df)}")
    print(f"[audit] coverage stats     : "
          f"mean={df['coverage'].mean():.4f}  "
          f"median={df['coverage'].median():.4f}  "
          f"max={df['coverage'].max():.4f}")
    print(f"\n[audit] top-{top_k} coverage:")
    print(df.head(top_k).to_string(index=False))
    nz = df[df["coverage"] > 0]
    print(f"\n[audit] bottom-{top_k} non-zero coverage:")
    print(nz.tail(top_k).to_string(index=False))
    print(f"\n[audit] zero-coverage tiles: {int((df['coverage'] == 0).sum())}")

    top = df.head(top_k)
    bottom = nz.tail(top_k)
    _save_grid(
        cfg,
        top["tile_id"].tolist(),
        top["coverage"].tolist(),
        cfg.paths.results / "mask_audit_top.png",
        f"Top-{top_k} highest-coverage tiles",
    )
    _save_grid(
        cfg,
        bottom["tile_id"].tolist(),
        bottom["coverage"].tolist(),
        cfg.paths.results / "mask_audit_bottom.png",
        f"Bottom-{top_k} non-zero coverage tiles",
    )
    print(f"[audit] preview grids saved to {cfg.paths.results}/mask_audit_*.png")
    return df


if __name__ == "__main__":
    audit_coverage(load_config())
