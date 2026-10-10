# HuBMAP Phase 3 Stain Augmentation Ablation

[HuBMAP — Hacking the Human Vasculature](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature) blood-vessel segmentation challenge.

**Live Demo**: [vessel-stain-lab.streamlit.app](https://vessel-stain-lab.streamlit.app/) 

## Hypotheses

Two hypotheses on HuBMAP Dataset 1 (n=2 WSIs):

1. HED-channel colour jitter at training improves Dice over RGB-only augmentation.
2. Macenko stain normalization with an H&E-derived reference does not transfer to PAS and degrades performance.

The combined arm tests whether augmentation and normalization combine additively as Tellez et al. (2019) report for H&E.

## Method

U-Net with ResNet-34 ImageNet encoder, four augmentation arms × three seeds (1, 2, 42):

| Arm                | Train-time aug | Eval-time stain norm |
| ------------------ | -------------- | -------------------- |
| `rgb_aug`          | RGB jitter     | –                    |
| `hed_only`         | HED jitter     | –                    |
| `macenko_only`     | Macenko        | Macenko              |
| `full_stain_aware` | HED jitter     | Macenko              |

Reported with bootstrap 95% CIs and paired one-sided Wilcoxon tests.

## Quickstart

```bash
conda env create -f environment.yml
conda activate amls2-final
python main.py
```

`main.py` is the single entry point. It runs deterministically across seeds 1, 2, and 42 (`torch.use_deterministic_algorithms(True, warn_only=True)`) and writes all figures, JSON summaries, and reports to disk without user input.

**Without raw data (autograder path).** If `data/raw/polygons.jsonl` is absent, `main.py` replays from cached `artifacts/` and rebuilds `results/` plus the markdown report. This is the GitHub Classroom autograder path on a fresh clone.

**With raw data.** Unzip the Kaggle data so:

```text
data/raw/
├── polygons.jsonl
├── tile_meta.csv
├── train/
└── test/
```

then `python main.py`. The full sweep (12 runs) completes in approximately 27 minutes on Apple Silicon (MPS) per `wall_time_seconds` in `phase3_summary.json`. Falls back to CPU.

## Live demo

**Vessel Stain Lab**: [vessel-stain-lab.streamlit.app](https://vessel-stain-lab.streamlit.app/) (Streamlit Community Cloud; if it has been idle, click "wake up" and give it a moment).

An interactive viewer for the v2 run (20 seeds per arm, two test WSIs). It has a tile explorer with every arm's prediction, a stain stress test with live inference, cross-seed results, and a methods view. A sidebar toggle switches between technical and plain-language views. Statistics come from the committed `artifacts_v2/phase3_summary.json`; images and live inference use the seed-42 models.

Run it locally (needs `data/raw/`, `data/processed/masks/` and `results_v2/checkpoints/`):

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

`python -m demo.build_deploy --out <folder outside the repo>` stages the deployment. `app/` holds the curated tiles, seed-42 prediction masks and slim CPU requirements, and goes to its own GitHub repo for Streamlit Community Cloud. `weights/` holds the seed-42 checkpoints and goes to a Hugging Face model repo, which the app downloads from on first start. It does not push.

## Repository structure

```text
.
├── main.py                 # single entry point
├── environment.yml         # conda env: amls2-final
├── requirements.txt
├── configs/default.yaml    # seeds, image size, epochs, paths, arms
├── src/
│   ├── data/               # masks, splits, datasets, augmentations
│   ├── models/             # build_model() factory
│   ├── training/           # train loop, evaluator, losses, runner
│   └── utils/              # config, paths, deterministic seeding
├── scripts/
│   ├── learning_curves.py
│   └── umap_activations.py
├── demo/                   # Streamlit demo + deploy builder (not used by main.py)
├── data/                   # raw/ + processed/ (gitignored)
├── results/                # per-run figures, JSONs (gitignored)
├── artifacts/              # cached outputs (committed for replay)
├── figures/                # report-ready PDFs
└── docs/ASSIGNMENT.md
```

## Reproducibility

- Single root entry point: `python main.py`.
- Conda env at root: `environment.yml`.
- Fixed seeds (1, 2, 42) + `torch.use_deterministic_algorithms(True, warn_only=True)` in `src/utils/seed.py`.
- Plain Python only; no notebooks, no Makefile.
- Plots written via matplotlib `Agg`.
- Device auto-selection (MPS → CPU).
- Cached `artifacts/` for autograder replay.

## Outputs

- `results/phase3_comparison.png` — four-arm Dice comparison with CIs.
- `results/unet_resnet34_<arm>_seed<seed>_{loss,val_dice,predictions}.png` — per-run curves and qualitative panels.
- `data/processed/phase3_summary.json` — bootstrap CIs, Wilcoxon tests, config snapshot.
- `~/Desktop/hubmap_phase3_results.md` — generated report (configurable).

## Autograding

See [`TUTORIAL_AUTOGRADING.md`](TUTORIAL_AUTOGRADING.md) for local autograder runs via fork. The graded workflow installs `environment.yml`, runs `pylint`, and executes `python main.py`.
