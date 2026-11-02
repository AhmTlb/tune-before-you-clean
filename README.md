# Tune Before You Clean

Code, pre-specified protocols and per-run results for the paper
**"Tune Before You Clean: A Confirmatory Benchmark of When Outlier and Label-Noise Removal Helps Tabular Classification."**

> Archived version: DOI `10.5281/zenodo.23896892` (added after the first Zenodo release)

## What is here

| Folder | Contents |
|---|---|
| `code/` | All experiment and analysis scripts (Python) and the Jupyter notebooks used to run them |
| `protocols/` | The written protocol of each stage (hypotheses and decision criteria fixed before the run) and the implementation-validation report |
| `results/` | Per-run results as compressed CSV (one row per dataset × seed × condition × treatment × model) and the summary tables produced by the analysis scripts |
| `figures/` | Figures of the paper (PDF and 300 dpi PNG) |

The raw datasets are **not** redistributed: all of them are downloaded from OpenML (benchmark suite 99, OpenML-CC18) by the scripts.

## Stages, scripts and protocols

| Stage | Question | Run | Analyse | Protocol |
|---|---|---|---|---|
| 1 | Overlap of eight scorers; headroom of a per-dataset recommender | `overlap_experiment.py` | `oracle_analysis.py` | (exploratory) |
| 2 | Detection of injected corruption | `experiment_v2.py` | `analyze_v2.py` | `PROTOCOL.md` |
| 3 | Realistic corruption, diagnose-then-treat, discovery/confirmation | `experiment_v3.py` | `analyze_v3.py`, `analyze_v3_confirm.py`, `analyze_v3_extra.py` | `PROTOCOL_v3.md`, `PROTOCOL_v3_confirm.md` |
| 4 | Tuning versus cleaning | `experiment_v4.py` | `analyze_v4.py` | `PROTOCOL_v4.md` |
| 5 | Oracle and right-sized removal; implementation validation | `experiment_v5.py`, `validate_implementations.py` | `analyze_v5.py` | `PROTOCOL_v5.md` |
| 6 | Classical ensemble filter | `experiment_v6.py` | `analyze_v6.py` | `PROTOCOL_v6.md` |
| – | Follow-up analyses requested in internal review | – | `analyze_review.py` | – |
| – | Decision-rule check (Section 4.9): Stages 4–5 rerun with prior-corrected balanced accuracy and ROC AUC | `experiment_v4.py`, `experiment_v5.py` | `compare_ba_check.py`; `analyze_v4.py` / `analyze_v5.py` with metric `ba_prior` | `PROTOCOL_ba_check.md` |
| – | Figures | – | `make_figures.py` | – |

## Installation

Python 3.10 or later. The protocols are written in Arabic (the authors' working language); every hypothesis and decision rule they contain is restated in English in the paper.

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the packages; `requirements-lock.txt` records the exact versions used for the paper.

## Reproducing the results

All commands are run from `code/`. Each run is resumable: finished (dataset, seed) jobs are stored in `<out>/parts/` and skipped on restart. `--n-jobs` sets the number of parallel workers.

```bash
# quick offline check (a few minutes)
python experiment_v5.py --source sklearn --max-datasets 2 --seeds 1 --n-jobs 2 --preset smoke --out v5_smoke

# Stage 1
python overlap_experiment.py --source openml --max-datasets 20 --seeds 3 --out results
python oracle_analysis.py results
# Stage 2
python experiment_v2.py --source openml --max-datasets 20 --seeds 3 --out results_v2
python analyze_v2.py results_v2
# Stage 3 (discovery = first 20 datasets, then all 40)
python experiment_v3.py --source openml --max-datasets 20 --seeds 3 --out results_v3
python experiment_v3.py --source openml --max-datasets 40 --seeds 5 --max-rows 2000 --out results_v3_full
python analyze_v3.py results_v3_full
python analyze_v3_confirm.py results_v3_full --discovery-from results_v3
python analyze_v3_extra.py results_v3_full
# Stage 4
python experiment_v4.py --source openml --max-datasets 40 --seeds 3 --out results_v4
python analyze_v4.py results_v4
# Stage 5
python validate_implementations.py
python experiment_v5.py --source openml --max-datasets 40 --seeds 3 --out results_v5
python experiment_v5.py --source openml --max-datasets 40 --seeds 3 --preset asym --out results_v5_asym
python analyze_v5.py results_v5
# Stage 6
python experiment_v6.py --source openml --max-datasets 40 --seeds 3 --out results_v6
python analyze_v6.py results_v6
# Review analyses and figures
python analyze_review.py --v3 results_v3 --v4 results_v4 --v5 results_v5 --v5asym results_v5_asym
python analyze_review.py --v3 results_v3_full --m5-exclude-from results_v3 --only M5 --out summary_review_M5_holdout.md
python make_figures.py
# Decision-rule check (Section 4.9)
python experiment_v4.py --source openml --max-datasets 40 --seeds 3 --out results_v4b
python experiment_v5.py --source openml --max-datasets 40 --seeds 3 --out results_v5b
python compare_ba_check.py results_v4b results_v5b
python analyze_v4.py results_v4b ba_prior
python analyze_v5.py results_v5b ba_prior
```

Row caps used in the paper: 3,000 rows in Stages 1–2 and in the 20-dataset run `results_v3` (the scripts' default for those stages); 2,000 rows in `results_v3_full` and in Stages 4–6. Pass `--max-rows` explicitly if you change a default.

To analyse the archived CSV results without re-running the experiments, load them with pandas, e.g. `pd.read_csv("results/results_v5/rows.csv.gz")`.

## Hardware and run time

The paper's runs used one laptop (AMD Ryzen AI 9 HX 370, 12 cores, 32 GB RAM) with 10 parallel workers.

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0. The OpenML datasets keep their own licences.

## Citation

See `CITATION.cff`, or cite the Zenodo DOI above.
