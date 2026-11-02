# Experiment v3: per-model and confirmatory analysis

Datasets: 20 | seeds: 3 | models: HistGB, KNN, LogReg, RandomForest

Gains = balanced-accuracy points vs. no treatment; averaged per dataset, then across datasets. Harm rate = % of cases with gain < -1 point.

## Part 1. Strategies per model (exploratory; DTT thresholds leave-one-dataset-out)

### HistGB

| strategy                     |   clean |   label |   feature |   mixed |    ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|-------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     |  0     |            0   |
| always classical (IF 5%)     |  -0.293 |  -0.53  |     0.058 |   0.043 | -0.191 |           23.8 |
| always target-aware (OOB 5%) |  -1.23  |   0.438 |    -0.781 |   0.501 | -0.146 |           28   |
| always both (DTT sizes)      |  -0.601 |   0.719 |    -0.607 |   0.404 |  0.06  |           32.1 |
| always reweight (OOB)        |   0.038 |   0.449 |    -0.019 |   0.437 |  0.24  |           11.1 |
| diagnose-then-treat (LODO)   |  -0.498 |   0.631 |    -0.527 |   0.363 |  0.059 |           30   |

### KNN

| strategy                     |   clean |   label |   feature |   mixed |     ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|--------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     |   0     |            0   |
| always classical (IF 5%)     |  -0.272 |  -0.219 |    -0.08  |  -0.048 |  -0.142 |           15   |
| always target-aware (OOB 5%) |  -0.371 |   0.629 |    -0.121 |   0.446 |   0.232 |           11.8 |
| always both (DTT sizes)      |  -0.214 |   0.717 |    -0.22  |   0.225 |   0.203 |           23.8 |
| always reweight (OOB)        | nan     | nan     |   nan     | nan     | nan     |          nan   |
| diagnose-then-treat (LODO)   |  -0.273 |   0.688 |    -0.213 |   0.221 |   0.188 |           23   |

### LogReg

| strategy                     |   clean |   label |   feature |   mixed |   ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     | 0     |            0   |
| always classical (IF 5%)     |  -0.311 |  -0.275 |     1.391 |   1.892 | 0.721 |           14.1 |
| always target-aware (OOB 5%) |  -0.167 |   0.433 |     0.66  |   0.787 | 0.525 |            8.9 |
| always both (DTT sizes)      |  -0.428 |   0.949 |     1.767 |   4.094 | 1.693 |           12.1 |
| always reweight (OOB)        |  -0.019 |   0.604 |     0.487 |   0.735 | 0.529 |            4.7 |
| diagnose-then-treat (LODO)   |  -0.467 |   1.009 |     1.635 |   3.919 | 1.631 |           11.7 |

### RandomForest

| strategy                     |   clean |   label |   feature |   mixed |    ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|-------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     |  0     |            0   |
| always classical (IF 5%)     |  -0.317 |  -0.223 |     0.079 |  -0.015 | -0.084 |           18   |
| always target-aware (OOB 5%) |  -1.02  |   0.469 |    -0.64  |   0.139 | -0.13  |           22.4 |
| always both (DTT sizes)      |  -0.728 |   0.142 |    -0.609 |  -0.18  | -0.269 |           33.2 |
| always reweight (OOB)        |   0.061 |   0.097 |    -0.065 |   0.044 |  0.025 |           11.5 |
| diagnose-then-treat (LODO)   |  -0.602 |   0.135 |    -0.516 |  -0.196 | -0.229 |           31.1 |
