# Experiment v3: per-model and confirmatory analysis

Datasets: 40 | seeds: 5 | models: HistGB, KNN, LogReg, RandomForest

Gains = balanced-accuracy points vs. no treatment; averaged per dataset, then across datasets. Harm rate = % of cases with gain < -1 point.

## Part 1. Strategies per model (exploratory; DTT thresholds leave-one-dataset-out)

### HistGB

| strategy                     |   clean |   label |   feature |   mixed |    ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|-------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     |  0     |            0   |
| always classical (IF 5%)     |  -0.531 |  -0.504 |    -0.067 |   0.005 | -0.255 |           29.6 |
| always target-aware (OOB 5%) |  -1.666 |   0.155 |    -1.352 |   0.619 | -0.474 |           35.5 |
| always both (DTT sizes)      |  -2.311 |   0.021 |    -1.865 |  -0.598 | -0.989 |           41.3 |
| always reweight (OOB)        |  -0.038 |   0.277 |    -0.069 |   0.347 |  0.135 |           17.5 |
| diagnose-then-treat (LODO)   |  -0.805 |  -0.483 |    -0.347 |  -0.516 | -0.469 |           28.8 |

### KNN

| strategy                     |   clean |   label |   feature |   mixed |     ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|--------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     |   0     |            0   |
| always classical (IF 5%)     |  -0.571 |  -0.411 |    -0.026 |  -0.07  |  -0.224 |           22.6 |
| always target-aware (OOB 5%) |  -0.853 |   0.36  |    -0.664 |   0.244 |  -0.144 |           22.3 |
| always both (DTT sizes)      |  -1.522 |  -0.158 |    -1.306 |  -0.887 |  -0.832 |           37   |
| always reweight (OOB)        | nan     | nan     |   nan     | nan     | nan     |          nan   |
| diagnose-then-treat (LODO)   |  -0.395 |  -0.362 |    -0.484 |  -0.472 |  -0.429 |           25.5 |

### LogReg

| strategy                     |   clean |   label |   feature |   mixed |   ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     | 0     |            0   |
| always classical (IF 5%)     |  -0.095 |   0.147 |     1.822 |   2.156 | 1.1   |           15   |
| always target-aware (OOB 5%) |  -0.338 |   0.397 |     0.483 |   1.119 | 0.493 |           14.2 |
| always both (DTT sizes)      |  -0.807 |   1.482 |     1.878 |   4.747 | 2.012 |           18.5 |
| always reweight (OOB)        |  -0.391 |   0.408 |     0.176 |   0.802 | 0.323 |           12.6 |
| diagnose-then-treat (LODO)   |   0.201 |   0.568 |     2.5   |   3.159 | 1.708 |           14.2 |

### RandomForest

| strategy                     |   clean |   label |   feature |   mixed |    ALL |   harm % (ALL) |
|:-----------------------------|--------:|--------:|----------:|--------:|-------:|---------------:|
| never treat                  |   0     |   0     |     0     |   0     |  0     |            0   |
| always classical (IF 5%)     |  -0.116 |  -0.191 |     0.098 |   0.016 | -0.042 |           23.5 |
| always target-aware (OOB 5%) |  -1.272 |  -0.023 |    -1.116 |  -0.073 | -0.543 |           34.5 |
| always both (DTT sizes)      |  -1.498 |  -0.372 |    -1.512 |  -1.105 | -1.022 |           43.4 |
| always reweight (OOB)        |   0.079 |   0.065 |     0.001 |   0.042 |  0.039 |           17   |
| diagnose-then-treat (LODO)   |   0.073 |   0.162 |    -0.252 |  -0.375 | -0.094 |           23.3 |

## Part 2. CONFIRMATORY test on held-out datasets (pre-registered)

Discovery datasets (used to choose everything): **20**. Confirmation datasets (never used for any choice): **20**.

DTT thresholds learned on discovery: theta_label = 0.051, theta_feature = 0.002.

Model-aware policy chosen on discovery:

|              | strategy                     |
|:-------------|:-----------------------------|
| HistGB       | always reweight (OOB)        |
| KNN          | always target-aware (OOB 5%) |
| LogReg       | always both (DTT sizes)      |
| RandomForest | never treat                  |

Best global strategy chosen on discovery: **diagnose-then-treat**.

### Results on CONFIRMATION datasets

| strategy                     |   mean_gain |    harm |   p_vs_never |
|:-----------------------------|------------:|--------:|-------------:|
| never treat                  |      0      |  0      |     nan      |
| always classical (IF 5%)     |      0.2101 | 25.8636 |       0.0296 |
| always target-aware (OOB 5%) |     -0.4812 | 34.3636 |       0.0172 |
| always both (DTT sizes)      |     -0.838  | 44.0682 |       0.2305 |
| always reweight (OOB)        |      0.0299 | 20.9697 |       0.498  |
| diagnose-then-treat          |     -0.7094 | 41.1591 |       0.3118 |
| MODEL-AWARE POLICY           |      0.4434 | 20.0455 |       0.0973 |
| best global strategy         |     -0.7094 | 41.1591 |       0.3118 |

### Pre-registered decisions

- H9a  model-aware > never (one-sided Wilcoxon): p = 0.04865 -> SUPPORTED
- H9b  model-aware vs best global (two-sided): diff = 1.153, p = 0.0003948 -> SUPPORTED
- H9c  harm rate of model-aware (20.0%) <= lowest harm of any fixed treating strategy (21.0%) -> SUPPORTED
- H9d  model-aware on clean confirmation data >= -0.2: -0.623 -> NOT supported
