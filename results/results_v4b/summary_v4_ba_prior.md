# Experiment v4: tuning / robust pipelines vs. cleaning

Datasets: 40 | seeds: 3 | conditions: clean, feature_gross@0.1, feature_mild@0.1, label_asym@0.2, label_sym@0.2, mixed@0.1

Gain = balanced-accuracy points vs. the family's DEFAULT model with NO treatment. Datasets weigh equally (seeds averaged first).

## A. Gains per family

### LogReg

| option                      |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:----------------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none              |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF         |   -0.66 |                2.97 |               1.19 |            -0.93 |           -1.6  |        2.12 |         0.75 |
| default | remove:OOB        |   -1.22 |                0.03 |              -0.56 |            -2.37 |            1.29 |        1.34 |        -0.05 |
| default | winsorize         |    0.13 |                1    |               0.16 |             0.15 |            0.06 |        0.77 |         0.43 |
| tuned | none                |    0.15 |                0.07 |              -0.03 |             0.45 |            1.31 |        0.38 |         0.44 |
| tuned | remove:IF           |   -0.58 |                3.33 |               1.27 |            -0.12 |           -0.18 |        2.97 |         1.45 |
| tuned | remove:OOB          |   -1.07 |               -0.03 |              -0.74 |            -1.98 |            2.1  |        1.41 |         0.15 |
| tuned | winsorize           |    0.17 |                0.87 |               0.21 |             0.61 |            1.22 |        1.41 |         0.86 |
| quantile | none             |   -0.53 |                2.4  |               0.41 |            -1.34 |           -0.71 |        2.26 |         0.61 |
| quantile | remove:IF        |   -1.13 |                2.76 |               0.82 |            -1.83 |           -2.19 |        1    |         0.11 |
| quantile | remove:OOB       |   -2.19 |                1.46 |              -0.44 |            -3.16 |            0.22 |        3.54 |         0.32 |
| quantile | winsorize        |   -0.94 |                0.95 |              -1.15 |            -1.78 |           -1.64 |        1.06 |        -0.51 |
| quantile+tuned | none       |   -0.32 |                2.49 |               0.41 |            -0.1  |            0.35 |        2.74 |         1.18 |
| quantile+tuned | remove:IF  |   -1.09 |                2.85 |               0.76 |            -0.66 |           -0.51 |        2.29 |         0.95 |
| quantile+tuned | remove:OOB |   -2.12 |                1.19 |              -0.46 |            -1.91 |            1.37 |        3.78 |         0.79 |
| quantile+tuned | winsorize  |   -0.89 |                0.93 |              -1.16 |            -0.87 |           -0.1  |        1.51 |         0.06 |

### KNN

| option                |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:----------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none        |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF   |   -1.16 |               -0.22 |              -0.13 |            -0.89 |           -1.28 |        0    |        -0.5  |
| default | remove:OOB  |   -2.08 |               -1.66 |              -1.48 |            -3.8  |            2.46 |        2.78 |        -0.34 |
| default | winsorize   |    0.07 |                0.12 |               0.21 |             0.22 |            0.05 |        0.21 |         0.16 |
| tuned | none          |    1.26 |                0.38 |               0.63 |             1.72 |            2.43 |        1.8  |         1.39 |
| tuned | remove:IF     |   -0.02 |                0.61 |               0.45 |             0.73 |            1.84 |        1.88 |         1.1  |
| tuned | remove:OOB    |   -1.48 |               -1.17 |              -1.16 |            -3.43 |            3.58 |        3.32 |         0.23 |
| tuned | winsorize     |    1.23 |                0.99 |               0.75 |             1.55 |            2.38 |        1.83 |         1.5  |
| quantile | none       |   -0.82 |                0.74 |               0.98 |            -1.13 |           -0.99 |        0.65 |         0.05 |
| quantile | remove:IF  |   -1.69 |               -0.82 |              -0.6  |            -1.86 |           -2.02 |       -0.9  |        -1.24 |
| quantile | remove:OOB |   -3.2  |               -1.23 |              -0.98 |            -4.68 |            1.68 |        3.48 |        -0.35 |
| quantile | winsorize  |   -1.38 |               -0.75 |              -0.86 |            -1.65 |           -1.56 |       -1.05 |        -1.17 |

### RandomForest

| option               |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:---------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none       |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF  |   -0.68 |               -0.12 |              -0.05 |            -0.75 |           -1.3  |       -0.28 |        -0.5  |
| default | remove:OOB |   -2.79 |               -2.7  |              -2.95 |            -5.35 |            1.61 |        1.34 |        -1.61 |
| default | winsorize  |   -0.1  |                0.03 |               0.04 |            -0.07 |            0.12 |        0.17 |         0.06 |
| tuned | none         |    0.22 |                0.3  |               0.44 |             0.63 |            1.42 |        1.02 |         0.76 |
| tuned | remove:IF    |   -0.56 |                0.24 |               0.05 |            -0.28 |            0.39 |        0.71 |         0.22 |
| tuned | remove:OOB   |   -3.16 |               -2.95 |              -2.95 |            -5.52 |            2.05 |        1.44 |        -1.59 |
| tuned | winsorize    |    0.16 |                0.36 |               0.45 |             0.91 |            1.52 |        0.89 |         0.82 |

### HistGB

| option               |   clean |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.2 |   mixed@0.1 |   dirty mean |
|:---------------------|--------:|--------------------:|-------------------:|-----------------:|----------------:|------------:|-------------:|
| default | none       |    0    |                0    |               0    |             0    |            0    |        0    |         0    |
| default | remove:IF  |   -0.92 |                0.45 |               0.18 |            -0.55 |           -1.13 |       -0.03 |        -0.21 |
| default | remove:OOB |   -3.78 |               -3.4  |              -3.36 |            -4.2  |            2.28 |        1.03 |        -1.53 |
| default | winsorize  |    0.02 |                0.04 |               0.02 |            -0.02 |           -0.07 |       -0    |        -0.01 |
| tuned | none         |   -0.4  |                0.58 |               0.43 |             1.49 |            2.19 |        1.27 |         1.19 |
| tuned | remove:IF    |   -0.86 |               -0.37 |               0.22 |             0.55 |            0.56 |        0.76 |         0.35 |
| tuned | remove:OOB   |   -3.36 |               -2.95 |              -2.87 |            -3.82 |            2.87 |        1.07 |        -1.14 |
| tuned | winsorize    |   -0.44 |                0.57 |               0.46 |             1.82 |            2.23 |        1.24 |         1.26 |

## B. Pre-registered tests

### H11a: does a robust tuned LogReg (quantile + tuned C) without cleaning recover the cleaning gain?

| condition         | cleaning   |   default_cleaning_gain |   robust_tuned_no_cleaning_gain |   recovery |
|:------------------|:-----------|------------------------:|--------------------------------:|-----------:|
| feature_gross@0.1 | remove:IF  |                   2.965 |                           2.488 |      0.839 |
| feature_mild@0.1  | remove:IF  |                   1.186 |                           0.406 |      0.342 |
| label_asym@0.2    | remove:OOB |                  -2.37  |                          -0.101 |    nan     |
| label_sym@0.2     | remove:OOB |                   1.294 |                           0.354 |      0.273 |
| mixed@0.1         | remove:IF  |                   2.119 |                           2.74  |      1.293 |
| mixed@0.1         | remove:OOB |                   1.342 |                           2.74  |      2.042 |

Average over dirty conditions: default+cleaning = 0.961, robust tuned without cleaning = 1.177 -> recovery = **123%** -> H11a SUPPORTED (criterion >= 50%).

### H11b and H11c (one-sided Wilcoxon over datasets, Holm over this table)

| hypothesis                                     | condition         | comparison                                             |   n |   mean_diff |      p |   p_holm |
|:-----------------------------------------------|:------------------|:-------------------------------------------------------|----:|------------:|-------:|---------:|
| H11b cleaning on top of robust tuned LogReg    | feature_gross@0.1 | quantile+tuned | remove:IF  vs  quantile+tuned | none  |  40 |      0.3649 | 0.0472 |   0.189  |
| H11b cleaning on top of robust tuned LogReg    | feature_mild@0.1  | quantile+tuned | remove:IF  vs  quantile+tuned | none  |  40 |      0.3532 | 0.5238 |   1      |
| H11b cleaning on top of robust tuned LogReg    | label_asym@0.2    | quantile+tuned | remove:OOB  vs  quantile+tuned | none |  37 |     -2.0444 | 0.9901 |   1      |
| H11b cleaning on top of robust tuned LogReg    | label_sym@0.2     | quantile+tuned | remove:OOB  vs  quantile+tuned | none |  40 |      1.0188 | 0.0004 |   0.0028 |
| H11b cleaning on top of robust tuned LogReg    | mixed@0.1         | quantile+tuned | remove:IF  vs  quantile+tuned | none  |  40 |     -0.4457 | 0.766  |   1      |
| H11b cleaning on top of robust tuned LogReg    | mixed@0.1         | quantile+tuned | remove:OOB  vs  quantile+tuned | none |  40 |      1.0401 | 0.0018 |   0.0106 |
| H11c removal harms tuned RandomForest on clean | clean             | tuned | remove:IF  vs  tuned | none                    |  40 |     -0.7882 | 0.0001 |   0.0004 |
| H11c removal harms tuned RandomForest on clean | clean             | tuned | remove:OOB  vs  tuned | none                   |  40 |     -3.3857 | 0      |   0      |
| H11c removal harms tuned HistGB on clean       | clean             | tuned | remove:IF  vs  tuned | none                    |  40 |     -0.4671 | 0.0049 |   0.0246 |
| H11c removal harms tuned HistGB on clean       | clean             | tuned | remove:OOB  vs  tuned | none                   |  40 |     -2.9617 | 0      |   0      |

H11b: cleaning adds significantly on top of the robust tuned LogReg in **2 of 6** dirty-condition tests.
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
| remove:IF   |                   -0.24 |                 nan    |
| remove:OOB  |                   -0.74 |                 -24.12 |

Tuned variants that fell back to defaults (smallest class too small for inner CV), %:

| family       |   fallback % |
|:-------------|-------------:|
| HistGB       |          0.1 |
| KNN          |          0.1 |
| LogReg       |          0.1 |
| RandomForest |          0.1 |
