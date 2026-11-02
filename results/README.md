# Results

Per-run results of the six stages and of the decision-rule check, exported from the experiment scripts with

```
python export_results.py --dest ../results --only results results_v2 results_v3 results_v3_full results_v4 results_v5 results_v5_asym results_v6 results_v4b results_v5b
```

| Folder | Stage |
|---|---|
| `results` | Stage 1, exploratory scorer overlap |
| `results_v2` | Stage 2, controlled corruption |
| `results_v3`, `results_v3_full` | Stage 3, discovery (20 datasets) and full run (40 datasets) |
| `results_v4` | Stage 4, tuning versus cleaning |
| `results_v5`, `results_v5_asym` | Stage 5, oracle and right-sized removal |
| `results_v6` | Stage 6, ensemble filter |
| `results_v4b`, `results_v5b` | Decision-rule check (Section 4.9) |

Each folder contains one compressed CSV per record type (for example `downstream.csv.gz`, `rows.csv.gz`, `precision.csv.gz`) and the summary tables (`summary_*.md`, `*.csv`) produced by the analysis scripts. Scores are stored as fractions; the paper reports points (x100).
