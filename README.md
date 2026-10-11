# Blood Vessel Segmentation in Kidney Histology

[![Lint](https://github.com/Yaser-Alharbi/kidney-blood-vessel-segmentation/actions/workflows/lint.yml/badge.svg)](https://github.com/Yaser-Alharbi/kidney-blood-vessel-segmentation/actions/workflows/lint.yml)

**Teaching AI to find blood vessels in kidney tissue, even when the stain colour changes from lab to lab.**

- **Story site** (methods, every number explained): https://yaser-alharbi.github.io/kidney-blood-vessel-segmentation/
- **Live lab** (run the models in your browser): https://kidney-blood-vessel-segmentation.streamlit.app/
- **Model weights**: https://huggingface.co/Yaser-Alharbi/vessel-stain-lab-weights

Pathology labs stain tissue slightly differently, and a segmentation model trained on one lab's colours can lose accuracy on another's. This project trains the same U-Net (ResNet-34 encoder, ImageNet weights) four ways, changing only how stain colour is handled, and repeats each recipe with 20 random seeds (80 runs). Data: the [HuBMAP *Hacking the Human Vasculature*](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature) kidney tiles, with the split made by whole-slide image so no slide appears in more than one split.

## Result

Test set: 681 tiles from 2 held-out slides. Mean Dice across 20 runs per recipe, with a paired one-sided Wilcoxon test against the baseline across those runs.

| Recipe | Train time | Test time | Mean Dice ± std | vs baseline | Wins | p |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (`rgb_aug`) | flips, rot90, brightness/contrast | none | 0.179 ± 0.024 | | | |
| Stain jitter (`hed_only`) | flips, rot90, HED stain jitter | none | **0.204 ± 0.022** | +13.8% | 16 / 20 | 0.0021 |
| Stain normalisation (`macenko_only`) | flips, rot90, Macenko | Macenko | 0.141 ± 0.022 | -21.6% | 1 / 20 | 1.00 |
| Jitter + normalisation (`full_stain_aware`) | flips, rot90, HED stain jitter | Macenko | 0.089 ± 0.008 | -50.2% | 0 / 20 | 1.00 |

- **Stain jitter helps** (supported on test and on the expert-labelled validation slide).
- **Macenko normalisation with an H&E reference hurts on PAS-stained tissue.**
- **The two do not combine additively**, unlike the H&E result of Tellez et al. (2019).

Test labels are auto-generated and noisy, so these scores compare recipes with each other; they are not absolute accuracy. The [story site](https://yaser-alharbi.github.io/kidney-blood-vessel-segmentation/) explains every number, the statistics and the limitations.

## Run it

```bash
conda env create -f environment.yml
conda activate amls2-final
python -m scripts.fetch_data          # Kaggle data into data/raw/ (needs Kaggle credentials)
```

| Task | Command |
| --- | --- |
| Train and evaluate the 20-seed study (CUDA) | `python -m scripts.run_v2` (`--pilot` for seed 42 only) |
| Cross-seed test report | `python -m scripts.report_v2` |
| Live lab, locally | `streamlit run demo/app.py` |
| Rebuild the story site into `docs/` | `python -m demo.build_site` (`--skip-images` re-renders text only) |
| Refresh the lab's deployment files | `python -m demo.build_deploy` |

All statistics come from the committed `artifacts_v2/phase3_summary.json`. The lab and the site use the seed-42 models (`results_v2/checkpoints/` locally, the Hugging Face repo when deployed).

`python main.py` is the grader entry point.

## Repository structure

```text
.
├── src/                    # data (masks, splits, transforms), model, training, evaluation
├── scripts/                # 20-seed study runner, report, data download, figures
├── configs/rerun_v2.yaml   # the 20-seed study configuration
├── artifacts_v2/           # committed results summary (every number on the site)
├── demo/
│   ├── app.py              # live lab (Streamlit)
│   ├── build_site.py       # story site generator -> docs/
│   ├── build_deploy.py     # writes demo_assets/ and demo/requirements.txt
│   └── site/               # site template, CSS and JS
├── demo_assets/            # curated tiles and predictions for the deployed lab
├── docs/                   # generated story site (GitHub Pages)
├── main.py                 # grader entry point
└── environment.yml
```
