# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A graded coursework repo (UCL ELEC0135 / AMLS II) built around one experiment: a four-arm stain-augmentation ablation on the HuBMAP "Hacking the Human Vasculature" blood-vessel segmentation data. U-Net + ResNet-34, four augmentation arms x three seeds. `README.md` states the hypotheses; `TUTORIAL_AUTOGRADING.md` explains the GitHub Classroom autograder.

Constraints imposed by the grader (do not break these):

- `main.py` must stay the single entry point, with `main()` taking no arguments, requiring no user input, and writing every metric/plot/log to disk.
- Plain Python only. No notebooks, no Makefile. Plots go through matplotlib `Agg`.
- CI runs `pylint -ry $(git ls-files '*.py')` and posts the report, so every module/class/function needs a docstring and the code must stay import-clean.

## Commands

```powershell
conda env create -f environment.yml   # env name: amls2-final
conda activate amls2-final
python main.py                        # full pipeline (the only supported entry point)
```

Individual stages, for debugging only (all require `data/raw/`):

```powershell
python -m src.data.inspect_data     # raw-data census -> inspection_summary.json
python -m src.data.build_masks      # polygons.jsonl -> data/processed/masks/*.png
python -m src.data.audit_masks      # mask coverage ranking -> mask_coverage.csv
python -m src.data.splits           # WSI-grouped split -> data/processed/splits.json
python -m scripts.learning_curves   # figures/learning_curves.pdf from cached summary
python -m scripts.umap_activations  # figures/umap_activations.pdf (needs checkpoints)
```

Live demo (`demo/`, never imported by `main.py` or `src/`). It reads `configs/rerun_v2.yaml` and the committed `artifacts_v2/phase3_summary.json`:

```powershell
streamlit run demo/app.py                         # needs data/raw, masks, results_v2/checkpoints
python -m demo.build_deploy --out <dir outside repo> # stage app/ (GitHub + Streamlit Cloud) and weights/ (HF model repo); does not push
```

Lint the way CI does:

```powershell
pylint -ry $(git ls-files '*.py')
```

There is no test suite. Verification is: `python main.py` completes and pylint is clean.

## Two execution modes

`main()` branches on whether `data/raw/polygons.jsonl` exists:

- **Full run** — inspect raw data, build masks, audit coverage, build splits, `run_phase3`, write figures + report, then `_save_artifacts` mirrors outputs into `artifacts/`.
- **Replay** (`_replay_from_artifacts`) — no raw data, so cached `artifacts/*.json` and `artifacts/*.png` are copied into `results/` and the markdown report is regenerated from `artifacts/phase3_summary.json`. This is the autograder path on a fresh clone, and the reason `artifacts/` is committed while `results/`, `data/` and `*.pth` are gitignored.

Consequence: a fresh clone cannot retrain or recompute anything (no data, no checkpoints). Changes to report formatting are verifiable via replay; changes to training are not.

## Architecture

Config-first. `src/utils/paths.py::load_config` reads `configs/default.yaml` into a frozen `Config` and resolves every path relative to the repo root (`~` expanded, absolutes kept). `cfg.raw` carries the whole YAML dict, so hyperparameters not promoted to `Config` fields (`epochs`, `lr`, `weight_decay`, `early_stop_patience`, `n_pred_examples`, `augs`) are read as `cfg.raw.get(...)` at the point of use. `cfg.paths.external_results_md` defaults to `~/Desktop/hubmap_phase3_results.md` — the report lands **outside** the repo.

Data flow: `tile_meta.csv` + `polygons.jsonl` -> `build_masks` (cv2.fillPoly, blood_vessel only, idempotent) -> `build_splits` -> `HuBMAPDataset` -> arm-specific transform -> `train_one_run`.

Splits are WSI-grouped and asserted non-overlapping: train and val are one dataset-1 WSI each (expert labels), test is the first held-out dataset-2 WSI (**auto-generated noisy labels** — test Dice is for relative comparison only, never absolute accuracy). A `train_noisy` pool of dataset-2 tiles from the train WSIs is also emitted; nothing currently trains on it.

`src/training/run_experiments.py::run_phase3` is the orchestrator: seeds `(42, 1, 2)` outer, arms inner, then stats, figure and summary JSON.

### The four arms

Arms differ at **both** train and eval time, which is the point of the ablation:

| Arm | Train | Eval |
| --- | --- | --- |
| `rgb_aug` | flips + rot90 + brightness/contrast | plain |
| `hed_only` | flips + rot90 + `HEDJitter` | plain |
| `macenko_only` | flips + rot90 + `MacenkoNormalize` | `MacenkoNormalize` |
| `full_stain_aware` | flips + rot90 + `HEDJitter` | `MacenkoNormalize` |

`HEDJitter` and `MacenkoNormalize` are hand-written Albumentations `ImageOnlyTransform`s in `src/data/transforms.py` (Ruifrok HED matrix; fixed Macenko reference stain matrix + 99th-pct concentrations). `MacenkoNormalize` returns the input unchanged on degenerate tiles rather than raising. Eval datasets are built with `cache=True` so Macenko is not recomputed each epoch — which is also why eval loaders use `num_workers=0`.

### Run tags, caching and resume

Run tag is `unet_resnet34_{arm}_seed{seed}`. Three layers of caching, checked in this order per (seed, arm):

1. Entry present in `phase3_summary.json` **and** `results/{tag}_per_tile.json` carries both Dice and IoU arrays -> skip.
2. Entry missing IoU -> backfill from the on-disk per-tile JSON.
3. Seed 42 with `results/checkpoints/{tag}.pth` present -> `_recompute_metrics_from_ckpt` re-runs forward passes for Dice + IoU without retraining.
4. Otherwise train from scratch.

`_migrate_legacy_seed42_keys` renames pre-seed-suffix tags (`unet_resnet34_rgb_aug` -> `..._seed42`) in place and is run before any training, so old summaries keep working. `run_phase3` also enforces a 3-hour wall-clock budget; anything not reached is recorded in `summary["pending"]` instead of failing, and the summary is re-saved after every run so a crash mid-sweep is resumable.

### Stats and reporting

`src/training/evaluate.py` provides per-tile Dice/IoU, `bootstrap_dice` (2000 resamples, fixed seed 42), and one-sided paired Wilcoxon. Two levels of test live in the summary:

- per-tile paired Wilcoxon on seed 42 (`stats.{val,test}.wilcoxon`) — powers the report tables,
- cross-seed paired Wilcoxon over three per-seed means (`stats.{val,test}.wilcoxon_cross_seed`); with n=3 the smallest achievable two-sided p is 0.25, noted in the code.

`main.py` renders the markdown report straight from that nested `stats` dict (`_format_phase3_section`, `_phase3_conclusion`, including an additivity check on the combined arm). The report reads seed-42 runs via `_PHASE3_RUN_TAG_BY_ARM`; the other seeds appear only in the cross-seed block.

## Gotchas

- **Adding or renaming an arm touches six places**: a builder in `src/data/transforms.py`, `_AUG_BUILDERS` in *both* `src/training/train.py` and `src/training/run_experiments.py`, `_ARMS`/`_ARM_COLORS`/`_ARM_LABELS` in `run_experiments.py`, `augs` in `configs/default.yaml`, `_PHASE3_ARMS` + `_PHASE3_RUN_TAG_BY_ARM` + the Wilcoxon key lists in `main.py`, and `ARMS` in `scripts/learning_curves.py`. The demo adds two more: `EVAL_BUILDERS` in `demo/inference.py` and `ARM_STYLE`/`ARM_BLURB`/`ARM_PIPELINE` in `demo/views.py` (the app raises if an arm has no label).
- `hed_shift` in `src/data/transforms.py` mirrors `HEDJitter.apply` with fixed alpha/beta (D channel untouched). Keep them in sync.
- `ARTIFACT_FIGURE_FILES` in `main.py` still lists legacy figure names without the seed suffix (`unet_resnet34_rgb_aug_loss.png`), while `train_one_run` writes `{run_tag}_loss.png` (i.e. `..._seed42_loss.png`). Newly trained figures are therefore not picked up by `_save_artifacts`; the committed `artifacts/` PNGs are the legacy-named ones. Fix the list if you regenerate figures.
- `run_phase3`'s seed list is hardcoded in `_DEFAULT_SEEDS`, not read from `cfg.seed` — `cfg.seed` only seeds the top-level `set_seed` and the split RNG.
- Device resolution is duplicated: `main._device_string` (mps -> cpu) and `train._resolve_device` (also honours cuda). Keep them consistent.
- `src/data/transforms.py` keeps back-compat aliases (`get_stain_aware_transforms`, `stain_aware_hed_only`) so older checkpoints/configs resolve; don't delete them.
- The UMAP step in `main()` is wrapped in a bare `except` and reported as non-fatal — a silent failure there will not fail the run.
- `README.md` references `docs/ASSIGNMENT.md`, which is not in the repo.
