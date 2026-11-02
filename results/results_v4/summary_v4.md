# Experiment v4: tuning / robust pipelines vs. cleaning

Datasets: 40 | seeds: 3 | conditions: clean, feature_gross@0.1, feature_mild@0.1, label_asym@0.2, label_sym@0.2, mixed@0.1

Gain = balanced-accuracy points vs. the family's DEFAULT model with NO treatment. Datasets weigh equally (seeds averaged first).

## A. Gains per family

### LogReg

| option                      |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:----------------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none              |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF         |   -0.52 |                5.17 |               1.79 |             0.79 |           -0.71 |        4.02 |         2.21 |
| default | remove:OOB        |   -0.8  |                1.22 |               0.26 |             0.26 |            1.33 |        1.89 |         0.99 |
| default | winsorize         |    0.21 |                2.06 |               0.75 |             0.28 |            0.14 |        1.77 |         1    |
| tuned | none                |    0.7  |                0.12 |               0.02 |             1.19 |            0.82 |        0.31 |         0.49 |
| tuned | remove:IF           |   -0.02 |                5.88 |               2.36 |             2.11 |            0.47 |        5.01 |         3.17 |
| tuned | remove:OOB          |   -0.35 |                1.52 |               0.7  |             1.31 |            2.19 |        2.04 |         1.55 |
| tuned | winsorize           |    0.8  |                2.25 |               0.79 |             1.44 |            1.17 |        2.28 |         1.58 |
| quantile | none             |   -0.84 |                3.69 |               0.65 |            -0.95 |           -0.47 |        3.32 |         1.25 |
| quantile | remove:IF        |   -1    |                4.54 |               1.7  |            -0.13 |           -0.7  |        3.11 |         1.7  |
| quantile | remove:OOB       |   -1.78 |                3.54 |               0.59 |            -0.95 |            0.52 |        5.32 |         1.8  |
| quantile | winsorize        |   -1.9  |                1.89 |              -0.97 |            -1.33 |           -1.44 |        1.44 |        -0.08 |
| quantile+tuned | none       |   -0.47 |                3.94 |               0.74 |             0.42 |            0.65 |        3.75 |         1.9  |
| quantile+tuned | remove:IF  |   -0.75 |                4.67 |               1.8  |             1.13 |            0.42 |        4.3  |         2.46 |
| quantile+tuned | remove:OOB |   -1.17 |                3.74 |               0.92 |             0.35 |            1.78 |        5.59 |         2.48 |
| quantile+tuned | winsorize  |   -1.59 |                1.96 |              -0.97 |            -0.28 |           -0.27 |        1.91 |         0.47 |

### KNN

| option                |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:----------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none        |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF   |   -1.29 |                0.02 |              -0.42 |            -0.76 |           -0.95 |       -0.14 |        -0.45 |
| default | remove:OOB  |   -1.33 |               -0.93 |              -1.07 |            -0.41 |            1.32 |        0.23 |        -0.17 |
| default | winsorize   |    0.13 |                0.04 |               0.14 |            -0.04 |            0.13 |       -0.19 |         0.02 |
| tuned | none          |    0.98 |                0.87 |               0.38 |             1.56 |            1.7  |        0.84 |         1.07 |
| tuned | remove:IF     |   -0.45 |                0.99 |               0.11 |             0.73 |            1.05 |        0.86 |         0.75 |
| tuned | remove:OOB    |   -0.65 |               -0.2  |              -0.57 |             0.23 |            2.26 |        1.1  |         0.57 |
| tuned | winsorize     |    1.02 |                0.9  |               0.82 |             1.6  |            2.01 |        0.77 |         1.22 |
| quantile | none       |   -1.17 |                0.37 |               0.08 |            -1.02 |           -0.72 |        0.13 |        -0.23 |
| quantile | remove:IF  |   -2.37 |               -0.95 |              -1.37 |            -1.66 |           -1.77 |       -1.41 |        -1.43 |
| quantile | remove:OOB |   -2.46 |               -0.6  |              -0.94 |            -1.28 |            0.54 |        0.45 |        -0.36 |
| quantile | winsorize  |   -1.85 |               -1.06 |              -1.64 |            -1.39 |           -1.4  |       -1.73 |        -1.44 |

### RandomForest

| option               |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:---------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none       |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF  |   -1.12 |                0.09 |               0.25 |            -0.09 |           -1.27 |       -0.12 |        -0.23 |
| default | remove:OOB |   -2.15 |               -2    |              -1.88 |            -0.39 |            0.44 |       -0.11 |        -0.79 |
| default | winsorize  |    0.01 |               -0.01 |               0.11 |             0.05 |           -0.06 |        0.01 |         0.02 |
| tuned | none         |    0.45 |                0.36 |               0.47 |             1.18 |            1.13 |        0.62 |         0.75 |
| tuned | remove:IF    |   -0.37 |                0.45 |               0.53 |             1.01 |            0.01 |        0.74 |         0.55 |
| tuned | remove:OOB   |   -1.94 |               -1.8  |              -1.61 |            -0.05 |            1.03 |        0.21 |        -0.44 |
| tuned | winsorize    |    0.48 |                0.2  |               0.44 |             1.19 |            0.97 |        0.59 |         0.68 |

### HistGB

| option               |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:---------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none       |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF  |   -1.15 |                0.11 |              -0.07 |            -0.78 |           -1.14 |       -0.37 |        -0.45 |
| default | remove:OOB |   -2.76 |               -2.6  |              -2.36 |            -1.14 |            1.71 |        0.76 |        -0.73 |
| default | winsorize  |   -0.04 |               -0.03 |               0.04 |            -0    |            0.02 |       -0.03 |        -0    |
| tuned | none         |   -0.36 |                0.19 |               0.4  |             0.58 |            2.07 |        1    |         0.85 |
| tuned | remove:IF    |   -0.96 |               -0.33 |               0.04 |            -0.27 |            0.37 |        0.44 |         0.05 |
| tuned | remove:OOB   |   -2.6  |               -2.45 |              -1.9  |            -1.03 |            2.17 |        0.91 |        -0.46 |
| tuned | winsorize    |   -0.39 |                0.13 |               0.48 |             0.62 |            2.08 |        0.86 |         0.84 |

## B. Pre-registered tests

### H11a: does a robust tuned LogReg (quantile + tuned C) without cleaning recover the cleaning gain?

| condition         | cleaning   |   default_cleaning_gain |   robust_tuned_no_cleaning_gain |   recovery |
|:------------------|:-----------|------------------------:|--------------------------------:|-----------:|
| feature_gross@0.1 | remove:IF  |                   5.172 |                           3.939 |      0.762 |
| feature_mild@0.1  | remove:IF  |                   1.794 |                           0.74  |      0.413 |
| label_asym@0.2    | remove:OOB |                   0.261 |                           0.421 |      1.615 |
| label_sym@0.2     | remove:OOB |                   1.333 |                           0.648 |      0.486 |
| mixed@0.1         | remove:IF  |                   4.018 |                           3.754 |      0.934 |
| mixed@0.1         | remove:OOB |                   1.886 |                           3.754 |      1.99  |

Average over dirty conditions: default+cleaning = 2.302, robust tuned without cleaning = 1.900 -> recovery = **83%** -> H11a SUPPORTED (criterion >= 50%).

### H11b and H11c (one-sided Wilcoxon over datasets, Holm over this table)

| hypothesis                                     | condition         | comparison                                             |   n |   mean_diff |      p |   p_holm |
|:-----------------------------------------------|:------------------|:-------------------------------------------------------|----:|------------:|-------:|---------:|
| H11b cleaning on top of robust tuned LogReg    | feature_gross@0.1 | quantile+tuned | remove:IF  vs  quantile+tuned | none  |  40 |      0.7305 | 0.0068 |   0.0271 |
| H11b cleaning on top of robust tuned LogReg    | feature_mild@0.1  | quantile+tuned | remove:IF  vs  quantile+tuned | none  |  40 |      1.059  | 0.0112 |   0.0335 |
| H11b cleaning on top of robust tuned LogReg    | label_asym@0.2    | quantile+tuned | remove:OOB  vs  quantile+tuned | none |  37 |     -0.0126 | 0.2892 |   0.2892 |
| H11b cleaning on top of robust tuned LogReg    | label_sym@0.2     | quantile+tuned | remove:OOB  vs  quantile+tuned | none |  40 |      1.134  | 0.0011 |   0.008  |
| H11b cleaning on top of robust tuned LogReg    | mixed@0.1         | quantile+tuned | remove:IF  vs  quantile+tuned | none  |  40 |      0.5458 | 0.0301 |   0.0603 |
| H11b cleaning on top of robust tuned LogReg    | mixed@0.1         | quantile+tuned | remove:OOB  vs  quantile+tuned | none |  40 |      1.8372 | 0      |   0      |
| H11c removal harms tuned RandomForest on clean | clean             | tuned | remove:IF  vs  tuned | none                    |  40 |     -0.8189 | 0.0021 |   0.0106 |
| H11c removal harms tuned RandomForest on clean | clean             | tuned | remove:OOB  vs  tuned | none                   |  40 |     -2.3854 | 0      |   0      |
| H11c removal harms tuned HistGB on clean       | clean             | tuned | remove:IF  vs  tuned | none                    |  40 |     -0.6005 | 0.0015 |   0.0093 |
| H11c removal harms tuned HistGB on clean       | clean             | tuned | remove:OOB  vs  tuned | none                   |  40 |     -2.2398 | 0      |   0      |

H11b: cleaning adds significantly on top of the robust tuned LogReg in **4 of 6** dirty-condition tests.
H11c: removal significantly harms tuned trees on clean data in **4 of 4** tests.

## C. Deleting whole classes (now evaluated, not skipped)

| treatment   |   % cases deleting a class |
|:------------|---------------------------:|
| none        |                        0   |
| remove:IF   |                        0   |
| remove:OOB  |                        1.7 |
| winsorize   |                        0   |

Mean gain (default models) when the removal did / did not delete a class:

| treatment   |   gain when wiped=False |   gain when wiped=True |
|:------------|------------------------:|-----------------------:|
| remove:IF   |                    0.06 |                 nan    |
| remove:OOB  |                   -0.34 |                  -4.96 |

Tuned variants that fell back to defaults (smallest class too small for inner CV), %:

| family       |   fallback % |
|:-------------|-------------:|
| HistGB       |          0.1 |
| KNN          |          0.1 |
| LogReg       |          0.1 |
| RandomForest |          0.1 |
