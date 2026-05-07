"""Render the four-arm learning curves figure from cached Phase 3 logs.

Read-only: this script never retrains and never edits training code.
Inputs : artifacts/phase3_summary.json (produced by src/training/run_experiments.py)
Outputs: figures/learning_curves.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Arm key -> legend label. Order matters: it controls plot legend order
# and the order of the per-arm stdout summary lines.
ARMS: dict[str, str] = {
    "rgb_aug": "Arm 1: RGB jitter",
    "hed_only": "Arm 2: HED jitter",
    "macenko_only": "Arm 3: Macenko",
    "full_stain_aware": "Arm 4: HED + Macenko",
}

# Run-tag prefix matches _PHASE3_RUN_TAG_BY_ARM in main.py:60-65.
RUN_TAG_PREFIX = "unet_resnet34_"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    summary_path = project_root / "artifacts" / "phase3_summary.json"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "learning_curves.pdf"

    with open(summary_path, "r") as f:
        phase3 = json.load(f)
    runs = phase3["runs"]

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(11, 4))

    # Cache per-arm summary stats so we can print after saving the figure.
    per_arm_stats: list[tuple[str, int, int, float]] = []

    for arm_key, arm_label in ARMS.items():
        run = runs[f"{RUN_TAG_PREFIX}{arm_key}"]
        train_loss = run["train_loss"]
        val_dice = run["val_dice"]

        epochs_train = range(1, len(train_loss) + 1)
        epochs_val = range(1, len(val_dice) + 1)

        ax_train.plot(epochs_train, train_loss, label=arm_label)
        (val_line,) = ax_val.plot(epochs_val, val_dice, label=arm_label)

        # mark argmax(val_dice) with an open circle in the same color
        # as the line for that arm.
        best_idx = max(range(len(val_dice)), key=lambda i: val_dice[i])
        ax_val.plot(
            best_idx + 1,
            val_dice[best_idx],
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=val_line.get_color(),
            markeredgewidth=1.5,
            linestyle="None",
        )

        per_arm_stats.append(
            (arm_key, len(val_dice), best_idx + 1, float(val_dice[best_idx]))
        )

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train loss (Dice + 0.5 BCE)")
    ax_train.set_title("Training loss")
    ax_train.grid(alpha=0.3)
    ax_train.legend(fontsize=9, loc="upper right")

    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation Dice")
    ax_val.set_title("Validation Dice")
    ax_val.grid(alpha=0.3)
    ax_val.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    for arm_key, total_epochs, best_epoch, best_val_dice in per_arm_stats:
        print(
            f"{arm_key}: total_epochs={total_epochs}, "
            f"best_epoch={best_epoch}, best_val_dice={best_val_dice:.4f}"
        )


if __name__ == "__main__":
    main()
