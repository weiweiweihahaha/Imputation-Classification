# Better Imputation ≠ Better Classification

Source code and supplementary materials for “ Better Imputation ≠ Better Classification: What Really Affects the Impact of Imputation on Classification Performance?”

## Repository Structure

```text
.
|-- code/
|   |-- workflow.py                    # unified command-line entry point
|   |-- generate_missing_data.py       # MCAR/MAR/MNAR generation
|   |-- Imputation.py                  # imputation-method dispatcher
|   |-- classification.py              # classification entry point
|   |-- analysis/                      # structural and decision metrics
|   |-- config/                        # datasets, methods, and defaults
|   |-- core/                          # data I/O, classifiers, runtime, paths
|   |-- pipeline/                      # split, training, and evaluation pipeline
|   |-- scripts/                       # batch and paper experiment scripts
|   `-- hsai/                          # HSR implementation and experiment driver
|-- data/
|   |-- Ground_truth_data/             # real-world complete datasets
|   `-- Real_missing_truth/            # real-world datasets with real missingness
|-- imputation_algorithms/             # imputation implementations
|-- run_holoclean.sh                   # HoloClean runner
|-- requirements.txt                   # core CPU and analysis dependencies
|-- requirements-optional.txt          # optional method-specific dependencies
`-- base.yml                           # Linux Conda environment snapshot
```

## Environment Setup

Python 3.10 recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt           # core
python -m pip install -r requirements-optional.txt  # optional methods
```

Or recreate the full Linux Conda environment:

```bash
conda env create -f base.yml -n imputation-classification
conda activate imputation-classification
```

R-backed methods, IIM/DLM, GCID, and HoloClean additionally require R, Java,
MATLAB, and PostgreSQL respectively.

For HoloClean, set the source directory and PostgreSQL credentials, then run:

```bash
HOLOCLEAN_SOURCE_DIR=/path/to/HoloClean/holoclean \
DB_USER=holocleanuser DB_PASSWORD=your_password \
./run_holoclean.sh
```

## Data

Included datasets under `data/Ground_truth_data/`: glass, hcv, iris, mobile, obesity, rice, seed, wine. Each CSV needs a header row with the class label in the first column and features after.

**Add a complete dataset** — place it at `data/Ground_truth_data/<dataset>.csv`:

```csv
class_id,feature_1,feature_2,feature_3
0,1.2,3.4,5.6
1,2.1,4.3,6.5
```

**Real-world data with real missingness** — create `data/Real_missing_truth/<dataset>/` containing `missing.csv` and a cell-aligned `truth.csv`.

## Quick Run

The following example creates one 20% MCAR version of Iris, imputes it with
Mean and KNNI, and evaluates a logistic-regression classifier:

```bash
python -m code.workflow generate \
  --dataset iris --dataset_category no_real_missing \
  --iter 1 --seed 42 --missing_percent 0.2 \
  --max_missing_lenth 4 --missing_mechanism MCAR

python -m code.workflow core \
  --datasets iris --dataset_category no_real_missing \
  --iters 1 --seeds 42 --missing_percents 0.2 --mechanisms MCAR \
  --classifiers LogisticRegression --imputers Mean KNNI \
  --tuning_cv_folds 3 --tuning_n_jobs 1 --parallel_jobs 1
```

Run `python -m code.workflow --help` for all stages and options.

## Supported Methods

**Classifiers:** DecisionTree, LogisticRegression, RandomForest, NaiveBayes, KNN, Adaboost, SVM, MLP, LightGBM

**Imputers (46):** Mean, Mice, Mode, KNNI, CMI, CI, XGBI, GMM, MIDA, SOM, Holoclean, MISSF, MIWAE, Datawig, RRSI, LOESS, HDI, MLPI, VAEI, CDI, IIM, DLM, GINN, NOMI, KNNE, GCID, IFC, MVNI, SMC-FCS, MIGEC, Sice, TWDA, MIPCA, NMF, POP, RIBG, CVBKNN, BPCA, LSA, PMM, CCF, ESD, MIST, CLRMC, ECLRMC, NIIA

## Output Files

Generated artifacts are written to git-ignored directories:

```text
data/Missing_data/   generated missing datasets
result_data/         imputed datasets and timing sidecars
result/              evaluation, analysis, figures, and logs
```

Per experiment cell, the main outputs are:

- `run_summary.csv` — one row per dataset/imputer/classifier configuration.
- `sample_level.csv` — predictions, margins, hard-sample indicators, and flip events.
- `structure_features.csv` — feature-level structural distortion.
- `manifest.json` — run configuration and reproducibility metadata.
- `stage_cache/` — intermediate split, prediction, and evaluation files.

## Reproducing the Paper Experiments

Experiments 1–4 consume existing `run_summary.csv` and `sample_level.csv`:

```bash
python -m code.scripts.run_paper_experiments \
  --summary result --sample_level result \
  --experiments 1 2 3 4 --output_dir result/paper_experiments
```

Experiment 5 consumes the paired HSR result table:

```bash
python -m code.scripts.run_paper_experiments \
  --experiments 5 --hsr-paired <path/to/hsr_paired.csv> \
  --output_dir result/paper_experiments
```

