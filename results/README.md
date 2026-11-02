Results
Fill this folder by running, from `code/`:
    python export_results.py --dest ../results --only results results_v2 results_v3 results_v3_full results_v4 results_v5 results_v5_asym results_v6 results_v4b results_v5b

One sub-folder per run folder (results, results_v2, results_v3, results_v3_full, results_v4,
results_v5, results_v5_asym, results_v6, and the decision-rule check results_v4b, results_v5b). Each contains one compressed CSV per record type
(for example `downstream.csv.gz`, `rows.csv.gz`, `precision.csv.gz`) and the summary tables
(`summary_*.md`, `*.csv`) produced by the analysis scripts. Gains in the paper are in
balanced-accuracy points (x100).
