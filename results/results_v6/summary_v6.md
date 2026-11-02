# Experiment v6: classical ensemble filter vs. Data-OOB

Datasets: 40 | seeds: 3

## A. Detectors: share of rows removed, precision, recall

|                                       |   share_removed |   precision |   recall |
|:--------------------------------------|----------------:|------------:|---------:|
| ('clean', 'ef:consensus')             |           0.072 |     nan     |  nan     |
| ('clean', 'ef:majority')              |           0.147 |     nan     |  nan     |
| ('clean', 'fix:EF@0.1')               |           0.1   |     nan     |  nan     |
| ('clean', 'fix:OOB@0.1')              |           0.1   |     nan     |  nan     |
| ('feature_gross@0.1', 'ef:consensus') |           0.084 |       0.286 |    0.181 |
| ('feature_gross@0.1', 'ef:majority')  |           0.186 |       0.265 |    0.391 |
| ('feature_gross@0.1', 'fix:EF@0.1')   |           0.1   |       0.297 |    0.297 |
| ('feature_gross@0.1', 'fix:OOB@0.1')  |           0.1   |       0.223 |    0.223 |
| ('feature_gross@0.1', 'rs:EF')        |           0.1   |       0.297 |    0.297 |
| ('feature_gross@0.1', 'rs:OOB')       |           0.1   |       0.223 |    0.223 |
| ('label_asym@0.2', 'ef:consensus')    |           0.11  |       0.389 |    0.249 |
| ('label_asym@0.2', 'ef:majority')     |           0.217 |       0.329 |    0.428 |
| ('label_asym@0.2', 'fix:EF@0.1')      |           0.1   |       0.404 |    0.254 |
| ('label_asym@0.2', 'fix:OOB@0.1')     |           0.1   |       0.368 |    0.233 |
| ('label_asym@0.2', 'rs:EF')           |           0.161 |       0.37  |    0.37  |
| ('label_asym@0.2', 'rs:OOB')          |           0.161 |       0.349 |    0.349 |
| ('label_sym@0.1', 'ef:consensus')     |           0.13  |       0.592 |    0.638 |
| ('label_sym@0.1', 'ef:majority')      |           0.227 |       0.461 |    0.846 |
| ('label_sym@0.1', 'fix:EF@0.1')       |           0.1   |       0.637 |    0.637 |
| ('label_sym@0.1', 'fix:OOB@0.1')      |           0.1   |       0.664 |    0.664 |
| ('label_sym@0.1', 'rs:EF')            |           0.1   |       0.637 |    0.637 |
| ('label_sym@0.1', 'rs:OOB')           |           0.1   |       0.664 |    0.664 |
| ('label_sym@0.2', 'ef:consensus')     |           0.18  |       0.71  |    0.596 |
| ('label_sym@0.2', 'ef:majority')      |           0.302 |       0.593 |    0.817 |
| ('label_sym@0.2', 'fix:EF@0.1')       |           0.1   |       0.787 |    0.393 |
| ('label_sym@0.2', 'fix:OOB@0.1')      |           0.1   |       0.818 |    0.409 |
| ('label_sym@0.2', 'rs:EF')            |           0.2   |       0.691 |    0.691 |
| ('label_sym@0.2', 'rs:OOB')           |           0.2   |       0.704 |    0.704 |
| ('mixed@0.1', 'ef:consensus')         |           0.138 |       0.651 |    0.397 |
| ('mixed@0.1', 'ef:majority')          |           0.261 |       0.534 |    0.612 |
| ('mixed@0.1', 'fix:EF@0.1')           |           0.1   |       0.708 |    0.354 |
| ('mixed@0.1', 'fix:OOB@0.1')          |           0.1   |       0.703 |    0.351 |
| ('mixed@0.1', 'rs:EF')                |           0.1   |       0.708 |    0.354 |
| ('mixed@0.1', 'rs:OOB')               |           0.1   |       0.703 |    0.351 |

### Asymmetric noise, binary vs multiclass

|                                |   precision |   recall |
|:-------------------------------|------------:|---------:|
| ('binary', 'ef:consensus')     |       0.199 |    0.046 |
| ('binary', 'ef:majority')      |       0.177 |    0.142 |
| ('binary', 'fix:EF@0.1')       |       0.221 |    0.206 |
| ('binary', 'fix:OOB@0.1')      |       0.172 |    0.176 |
| ('binary', 'rs:EF')            |       0.217 |    0.217 |
| ('binary', 'rs:OOB')           |       0.184 |    0.184 |
| ('multiclass', 'ef:consensus') |       0.62  |    0.498 |
| ('multiclass', 'ef:majority')  |       0.515 |    0.777 |
| ('multiclass', 'fix:EF@0.1')   |       0.627 |    0.313 |
| ('multiclass', 'fix:OOB@0.1')  |       0.608 |    0.304 |
| ('multiclass', 'rs:EF')        |       0.557 |    0.557 |
| ('multiclass', 'rs:OOB')       |       0.551 |    0.551 |

## B. Gains vs. no treatment (balanced_accuracy, points)

### LogReg

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |                5.41 |             8.58 |            2.78 |            4.5  |        7.25 |
| ef:majority  |   -0.33 |                1.41 |             1.68 |            1.95 |            2.57 |        2.87 |
| ef:consensus |   -0.05 |                0.96 |             1.67 |            1.97 |            2.48 |        2.52 |
| rs:EF        |  nan    |                1.4  |             1.81 |            2.55 |            2.97 |        2.98 |
| rs:OOB       |  nan    |                1.22 |             0.39 |            2.12 |            2.8  |        1.89 |
| fix:EF@0.1   |   -0.3  |                1.4  |             1.47 |            2.55 |            1.98 |        2.98 |
| fix:OOB@0.1  |   -0.8  |                1.22 |             0.26 |            2.12 |            1.33 |        1.89 |

### KNN

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |               -0.07 |             6.93 |            0.87 |            2.84 |        0.47 |
| ef:majority  |   -1.75 |               -2.7  |            -0.03 |           -1.2  |            0.3  |       -2.04 |
| ef:consensus |   -0.18 |               -0.5  |             0.94 |           -0.02 |            1.31 |       -0.41 |
| rs:EF        |  nan    |               -1.3  |             0.37 |            0.22 |            1.46 |       -0.04 |
| rs:OOB       |  nan    |               -0.93 |            -0.42 |            0.2  |            1.62 |        0.23 |
| fix:EF@0.1   |   -1.41 |               -1.3  |             0.5  |            0.22 |            1.48 |       -0.04 |
| fix:OOB@0.1  |   -1.33 |               -0.93 |            -0.41 |            0.2  |            1.32 |        0.23 |

### RandomForest

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |                0.08 |             9.19 |            0.63 |            1.88 |        1.04 |
| ef:majority  |   -1.96 |               -3.6  |            -0.37 |           -2.17 |           -1.72 |       -3.28 |
| ef:consensus |   -0.56 |               -1.22 |             0.5  |           -0.36 |           -0.08 |       -0.64 |
| rs:EF        |  nan    |               -2.31 |             0.15 |           -0.24 |           -0.11 |       -0.68 |
| rs:OOB       |  nan    |               -2    |            -0.55 |           -0.3  |            0.49 |       -0.11 |
| fix:EF@0.1   |   -1.73 |               -2.31 |             0.36 |           -0.24 |            0.72 |       -0.68 |
| fix:OOB@0.1  |   -2.15 |               -2    |            -0.39 |           -0.3  |            0.44 |       -0.11 |

### HistGB

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |                0.16 |             9.77 |            1.73 |            4.87 |        2.09 |
| ef:majority  |   -2.32 |               -3.34 |            -0.59 |           -1.31 |            0.56 |       -2.06 |
| ef:consensus |   -0.75 |               -1.15 |             0.8  |            0.11 |            1.72 |        0.07 |
| rs:EF        |  nan    |               -2.31 |            -0.07 |            0.31 |            2.12 |       -0.09 |
| rs:OOB       |  nan    |               -2.6  |            -0.92 |            0.3  |            2.07 |        0.76 |
| fix:EF@0.1   |   -2.24 |               -2.31 |            -0.24 |            0.31 |            1.89 |       -0.09 |
| fix:OOB@0.1  |   -2.76 |               -2.6  |            -1.14 |            0.3  |            1.71 |        0.76 |

## B. Gains vs. no treatment (accuracy, points)

### LogReg

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |                2.91 |             6.47 |            1.76 |            3.24 |        4.18 |
| ef:majority  |   -0.47 |                0.95 |             1.76 |            1.07 |            1.7  |        1.75 |
| ef:consensus |   -0.12 |                0.75 |             1.42 |            1.17 |            1.54 |        1.69 |
| rs:EF        |  nan    |                1.01 |             1.81 |            1.21 |            1.75 |        1.77 |
| rs:OOB       |  nan    |                0.87 |             1.06 |            1.36 |            2.15 |        1.18 |
| fix:EF@0.1   |   -0.4  |                1.01 |             1.41 |            1.21 |            0.8  |        1.77 |
| fix:OOB@0.1  |   -0.15 |                0.87 |             0.61 |            1.36 |            0.85 |        1.18 |

### KNN

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |               -0.01 |             5.42 |            1.52 |            4.49 |        1.59 |
| ef:majority  |   -0.5  |               -0.89 |             1.06 |            0.58 |            2.82 |        0.31 |
| ef:consensus |    0.24 |                0.17 |             1.43 |            0.83 |            2.63 |        0.65 |
| rs:EF        |  nan    |                0.01 |             1.29 |            0.81 |            3.12 |        0.94 |
| rs:OOB       |  nan    |                0.19 |             0.78 |            1.01 |            3.41 |        1.29 |
| fix:EF@0.1   |   -0.36 |                0.01 |             1.18 |            0.81 |            2.09 |        0.94 |
| fix:OOB@0.1  |   -0.03 |                0.19 |             0.57 |            1.01 |            2.32 |        1.29 |

### RandomForest

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |               -0.08 |             6.42 |            1.32 |            3.25 |        1.33 |
| ef:majority  |   -1.11 |               -1.99 |             0.38 |           -0.56 |            0.82 |       -1.17 |
| ef:consensus |   -0.21 |               -0.56 |             0.8  |            0.28 |            1.22 |        0.24 |
| rs:EF        |  nan    |               -0.74 |             0.67 |            0.3  |            1.58 |        0.38 |
| rs:OOB       |  nan    |               -0.99 |             0.2  |            0.37 |            1.72 |        0.47 |
| fix:EF@0.1   |   -0.9  |               -0.74 |             0.81 |            0.3  |            1.34 |        0.38 |
| fix:OOB@0.1  |   -1.14 |               -0.99 |             0.23 |            0.37 |            1.03 |        0.47 |

### HistGB

| option       |   clean |   feature_gross@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:-------------|--------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle       |  nan    |                0.06 |             7.27 |            2.48 |            5.94 |        2.56 |
| ef:majority  |   -1.17 |               -1.39 |             1.07 |            0.74 |            3.29 |        0.33 |
| ef:consensus |   -0.27 |               -0.41 |             1.51 |            1.17 |            3.04 |        1.08 |
| rs:EF        |  nan    |               -0.53 |             1.22 |            1.25 |            3.82 |        1.07 |
| rs:OOB       |  nan    |               -1.06 |             0.64 |            1.17 |            3.68 |        1.45 |
| fix:EF@0.1   |   -0.95 |               -0.53 |             1.04 |            1.25 |            2.57 |        1.07 |
| fix:OOB@0.1  |   -1.26 |               -1.06 |             0.37 |            1.17 |            2.38 |        1.45 |

## C. Pre-specified hypotheses (Wilcoxon over datasets, Holm over this table)

| hypothesis                            | family       | condition      |   n |    mean |      p |   p_holm |
|:--------------------------------------|:-------------|:---------------|----:|--------:|-------:|---------:|
| H13a EF (native) harms trees on clean | RandomForest | clean          |  40 | -1.9588 | 0.0005 |   0.0016 |
| H13a EF (native) harms trees on clean | HistGB       | clean          |  40 | -2.3166 | 0      |   0.0001 |
| H13b asym gap oracle - rs:EF > 0      | LogReg       | label_asym@0.2 |  40 |  6.7698 | 0      |   0      |
| H13b asym gap oracle - rs:EF > 0      | KNN          | label_asym@0.2 |  40 |  6.553  | 0      |   0      |
| H13b asym gap oracle - rs:EF > 0      | RandomForest | label_asym@0.2 |  40 |  9.0409 | 0      |   0      |
| H13b asym gap oracle - rs:EF > 0      | HistGB       | label_asym@0.2 |  40 |  9.8327 | 0      |   0      |
| H13c rs:EF vs rs:OOB (two-sided)      | mean of 4    | label_sym@0.2  |  40 | -0.1324 | 0.6179 |   0.6179 |
| H13c rs:EF vs rs:OOB (two-sided)      | mean of 4    | label_asym@0.2 |  40 |  0.943  | 0.0172 |   0.0343 |

- H13a (EF harms RF and HistGB on clean data): **SUPPORTED**
- H13b (asymmetric-noise gap > 1 point and significant for every family with EF): **SUPPORTED**
- H13c (EF vs Data-OOB at the right size): label_sym@0.2: diff = -0.13, p_holm = 0.618; label_asym@0.2: diff = +0.94, p_holm = 0.0343 (non-significance is not evidence of equivalence).

## D. Removals that deleted a whole class (%)

| option       |   % wiped |
|:-------------|----------:|
| ef:consensus |      0    |
| ef:majority  |      0.83 |
| fix:EF@0.1   |      0.83 |
| fix:OOB@0.1  |      1.47 |
| oracle       |      0    |
| rs:EF        |      0    |
| rs:OOB       |      0.63 |
