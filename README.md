# MCI / Dementia EEG Benchmarking Pipeline

Benchmarks a set of deep learning architectures on EEG-based classification
tasks for dementia and mild cognitive impairment (MCI) detection, across
four datasets. For each dataset/task/model combination the pipeline runs a
hyperparameter search, k-fold cross-validated training, and saves accuracy
metrics plus gradient-based (saliency) interpretability plots.

> **Scope:** this repository contains the benchmarking pipeline only. A
> separate clustering-based analysis pipeline previously lived alongside it
> and has been removed — if you find references to it in old commits or
> docs, that work no longer lives here.

## Datasets & tasks

| `--dataset` | Task(s) |
|---|---|
| `GENEEG` | MCI vs HC |
| `MCIvsHC` | MCI vs HC |
| `ADvsFTDvsHC` | AD vs HC, FTD vs HC, FTD vs AD |
| `CAUEEG` | Dementia vs Normal, MCI vs Normal, MCI vs Dementia |

A dataset's tasks all run automatically in a single invocation — you don't
select a task, only a dataset.

## Models

Passed via `--model`:

- `EEG_Conformer`
- `EEG_Deformer`
- `MSVTNet`
- `Oh_CNN`
- `SzHNN`
- `DeepConvNet`
- `EEGNet`
- `ShallowConveNet`
- `SCCNet`
- `MBSzEEGNet`

Use `--model all` to run every model above in sequence, or any single name
to run just that one. See `models/models_base/models_list.py` for each
model's class and constructor arguments.

## Setup

```bash
conda env create -f environment.yml
conda activate MCI
```

## Data layout

`datasets/raw/` and `datasets/preprocessed/` are tracked as empty folders —
their contents are gitignored and not distributed with this repo. Populate
`datasets/raw/` with the following structure before running:

| Dataset | Path | Required file |
|---|---|---|
| GENEEG | `datasets/raw/GENEEG/all_data/` | `metadata.xlsx` |
| MCIvsHC | `datasets/raw/MCIvsHC/data_extended/` | `states_2.xlsx` |
| ADvsFTDvsHC | `datasets/raw/ADvsFTDvsHC/data/` | `participants.tsv` |
| CAUEEG | `datasets/raw/CAUEEG/signal/edf/` (signals) + `datasets/raw/CAUEEG/dementia-no-overlap.json` (annotations) | — |

Plus the raw EEG recordings themselves under each dataset's directory. See
`utils/dataset_registry.py` for the exact paths and channel layouts used to
load each one. If `need_preprocessing: true` in the config, `datasets/preprocessed/`
is (re)populated automatically from the raw data on the next run.

## Configuration

Settings that stay constant across most runs live in
[`configs/benchmark.yaml`](configs/benchmark.yaml): sampling rate, frequency
bands for interpretability plots, k-fold/repeat counts, the pilot
hyperparameter search grid, and whether preprocessing should run first.
Pass a different file with `--config path/to/other.yaml` to override it.

Only *which* dataset and model to run are CLI arguments — everything else
is config.

## Running it

```bash
python main.py --dataset CAUEEG --model EEGNet
python main.py --dataset CAUEEG --model all
python main.py --help
```

### Caching behavior

Runs are resumable and cached by design. For each
(dataset, task, model, iteration):

- if a metrics JSON already exists under `results/metrics/<dataset>/logs/`,
  training is skipped and the cached result is reused;
- if a best-hyperparameters JSON already exists under
  `results/best params logs/<dataset>/`, the pilot search is skipped and
  full k-fold training runs directly with those parameters.

To force a re-run, delete the corresponding cached file(s) for that
dataset/task/model/iteration.

## Where results are saved

All under `results/<dataset>/` unless noted:

| Folder | Contents |
|---|---|
| `best params logs/` | Best hyperparameters found by the pilot search, per task/model/iteration |
| `k-fold dist logs/` | Per-fold subject/class distribution logs |
| `metrics/` | Per-iteration metrics JSONs (`logs/` subfolder) and the final `<dataset>_metrics_overall_(...).xlsx` summary averaged across iterations |
| `predictions/` | Reserved for per-subject labels/predictions (`save_labels_and_predictions()` in `main_func.py`) — not currently invoked by the training loop, so this stays empty on a normal run today |
| `topomaps/` | Gradient-based PSD and topomap interpretability plots per iteration |
| `training and inference time logs/` | Timing logs |
| `tsne plots/` | t-SNE embedding visualizations |
| `weights/` | Trained model weights |
| `seeds_<dataset>_<task>.txt` (repo root of `results/`) | Fixed random seeds used for that dataset/task, generated once and reused on every subsequent run for reproducibility |

Note: these folders are tracked in git as empty structure only (`.gitkeep`)
— their actual contents are gitignored and stay local to whoever ran the
benchmark.

## Repository layout

```
main.py                        CLI entry point (parses --dataset/--model/--config, runs the benchmark loop)
main_func.py                   Dataset build/label-mapping wrappers, seeding, metrics helpers
configs/benchmark.yaml         Non-identity settings (sfreq, k-folds, hyperparameter grid, ...)
models/
  models_base/                 One file per model architecture + models_list.py registry
  models_train/                Training loop (train_10_K_fold, train_10_K_fold_pilot)
utils/
  dataset_registry.py          Per-dataset paths, channels, tasks, label maps, metadata loaders
  EEG_preprocess.py             Raw EEG preprocessing/loading
  interpretation.py             Saliency/PSD/topomap plotting
  metrics.py                    Segment- and subject-wise metric computation
datasets/                      Raw and preprocessed EEG data (gitignored content)
results/                       All run outputs (gitignored content, see table above)
```

## License

[MIT](LICENSE)