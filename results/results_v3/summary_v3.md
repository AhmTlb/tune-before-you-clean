# Experiment v3: summary

Datasets: 20 | seeds: 3 | conditions: clean, feature_gross@0.05, feature_gross@0.1, feature_mild@0.05, feature_mild@0.1, label_asym@0.1, label_asym@0.2, label_sym@0.1, label_sym@0.2, mixed@0.05, mixed@0.1

Accuracy = balanced accuracy in points (x100); datasets weigh equally.

## A. RQ1 - Overlap of top-5% flagged points (clean)

Random expectation of Jaccard: 0.026

| pair_type           |   jaccard |   spearman |
|:--------------------|----------:|-----------:|
| cross-family        |     0.031 |      0.076 |
| within-classical    |     0.275 |      0.582 |
| within-target-aware |     0.324 |      0.588 |

## B. RQ4a - AUROC for detecting injected corrupted rows

Column = condition [which corrupted rows are the target]. 0.5 = chance.

| method            |   feature_gross@0.05 [feature] |   feature_gross@0.1 [feature] |   feature_mild@0.05 [feature] |   feature_mild@0.1 [feature] |   label_asym@0.1 [label] |   label_asym@0.2 [label] |   label_sym@0.1 [label] |   label_sym@0.2 [label] |   mixed@0.05 [feature] |   mixed@0.05 [label] |   mixed@0.1 [feature] |   mixed@0.1 [label] |
|:------------------|-------------------------------:|------------------------------:|------------------------------:|-----------------------------:|-------------------------:|-------------------------:|------------------------:|------------------------:|-----------------------:|---------------------:|----------------------:|--------------------:|
| IQR               |                          0.97  |                         0.97  |                         0.911 |                        0.914 |                    0.514 |                    0.521 |                   0.504 |                   0.496 |                  0.97  |                0.477 |                 0.971 |               0.448 |
| MAD               |                          0.953 |                         0.955 |                         0.883 |                        0.881 |                    0.531 |                    0.529 |                   0.496 |                   0.493 |                  0.955 |                0.476 |                 0.958 |               0.449 |
| RobustMahalanobis |                          0.994 |                         0.993 |                         0.974 |                        0.976 |                    0.537 |                    0.543 |                   0.501 |                   0.494 |                  0.995 |                0.476 |                 0.993 |               0.448 |
| LOF               |                          0.996 |                         0.996 |                         0.99  |                        0.989 |                    0.51  |                    0.516 |                   0.5   |                   0.496 |                  0.996 |                0.478 |                 0.996 |               0.443 |
| IsolationForest   |                          0.998 |                         0.999 |                         0.983 |                        0.988 |                    0.53  |                    0.538 |                   0.5   |                   0.494 |                  0.999 |                0.482 |                 1     |               0.445 |
| DataOOB           |                          0.742 |                         0.746 |                         0.74  |                        0.747 |                    0.86  |                    0.799 |                   0.931 |                   0.914 |                  0.709 |                0.939 |                 0.655 |               0.925 |
| KNNShapley        |                          0.801 |                         0.802 |                         0.791 |                        0.789 |                    0.826 |                    0.828 |                   0.891 |                   0.893 |                  0.774 |                0.887 |                 0.74  |               0.89  |
| SelfConfidence    |                          0.737 |                         0.742 |                         0.739 |                        0.742 |                    0.86  |                    0.799 |                   0.935 |                   0.919 |                  0.705 |                0.941 |                 0.645 |               0.931 |

## C. RQ3 - Who gets flagged on clean data (top-5%)

| method            |   ('minority_enrichment', 'mean') |   ('minority_enrichment', 'median') |   ('hardness_ratio', 'mean') |   ('hardness_ratio', 'median') |
|:------------------|----------------------------------:|------------------------------------:|-----------------------------:|-------------------------------:|
| IQR               |                              0.72 |                                0.41 |                         0.99 |                           0.86 |
| MAD               |                              0.79 |                                0.65 |                         0.93 |                           0.83 |
| RobustMahalanobis |                              0.85 |                                0.51 |                         0.99 |                           0.77 |
| LOF               |                              1.05 |                                1.02 |                         1.88 |                           1.25 |
| IsolationForest   |                              0.95 |                                0.82 |                         1.05 |                           0.98 |
| DataOOB           |                              1.78 |                                1.1  |                         3.96 |                           3.01 |
| KNNShapley        |                              2.37 |                                1.57 |                         3.56 |                           2.66 |
| SelfConfidence    |                              1.99 |                                1.23 |                         4.28 |                           3.19 |

## D. RQ6 - Can corruption be diagnosed without knowing it?

Spearman(est_label_noise, true label rate) = **0.52**; Spearman(extreme_rate, true feature rate) = **0.36** (over all dataset x seed x condition cases).

On clean data both diagnostics should be low; their clean level is the 'false alarm' baseline (inherent class overlap inflates est_label_noise).

| condition          |   ('est_label_noise', 'mean') |   ('est_label_noise', 'std') |   ('true_label_rate', 'mean') |   ('true_label_rate', 'std') |   ('extreme_rate', 'mean') |   ('extreme_rate', 'std') |   ('true_feature_rate', 'mean') |   ('true_feature_rate', 'std') |
|:-------------------|------------------------------:|-----------------------------:|------------------------------:|-----------------------------:|---------------------------:|--------------------------:|--------------------------------:|-------------------------------:|
| clean              |                         0.107 |                        0.127 |                         0     |                        0     |                      0.081 |                     0.163 |                            0    |                              0 |
| feature_gross@0.05 |                         0.109 |                        0.128 |                         0     |                        0     |                      0.121 |                     0.159 |                            0.05 |                              0 |
| feature_gross@0.1  |                         0.112 |                        0.13  |                         0     |                        0     |                      0.164 |                     0.162 |                            0.1  |                              0 |
| feature_mild@0.05  |                         0.109 |                        0.127 |                         0     |                        0     |                      0.093 |                     0.186 |                            0.05 |                              0 |
| feature_mild@0.1   |                         0.111 |                        0.127 |                         0     |                        0     |                      0.099 |                     0.188 |                            0.1  |                              0 |
| label_asym@0.1     |                         0.187 |                        0.111 |                         0.1   |                        0     |                      0.081 |                     0.163 |                            0    |                              0 |
| label_asym@0.2     |                         0.265 |                        0.101 |                         0.193 |                        0.014 |                      0.081 |                     0.163 |                            0    |                              0 |
| label_sym@0.1      |                         0.206 |                        0.113 |                         0.1   |                        0     |                      0.081 |                     0.163 |                            0    |                              0 |
| label_sym@0.2      |                         0.307 |                        0.093 |                         0.2   |                        0     |                      0.081 |                     0.163 |                            0    |                              0 |
| mixed@0.05         |                         0.16  |                        0.122 |                         0.05  |                        0     |                      0.123 |                     0.16  |                            0.05 |                              0 |
| mixed@0.1          |                         0.217 |                        0.11  |                         0.1   |                        0     |                      0.171 |                     0.171 |                            0.1  |                              0 |

## E. PRIMARY pre-specified tests (matched family, remove 10%, vs no treatment)

Holm correction over these 56 tests only (fixed in PROTOCOL_v3.md before the run).

| condition          | model        | option                     |   n |   mean_gain |   wins |   losses |      p |   p_holm |
|:-------------------|:-------------|:---------------------------|----:|------------:|-------:|---------:|-------:|---------:|
| clean              | HistGB       | remove:DataOOB@0.1         |  20 |     -2.0243 |      4 |       16 | 0.0027 |   0.1139 |
| clean              | KNN          | remove:DataOOB@0.1         |  20 |     -0.5898 |      7 |       13 | 0.1327 |   1      |
| clean              | LogReg       | remove:DataOOB@0.1         |  20 |     -0.201  |      8 |       12 | 0.4009 |   1      |
| clean              | RandomForest | remove:DataOOB@0.1         |  20 |     -1.73   |      4 |       16 | 0.0003 |   0.0123 |
| clean              | HistGB       | remove:IsolationForest@0.1 |  20 |     -0.7901 |      4 |       16 | 0.0073 |   0.2699 |
| clean              | KNN          | remove:IsolationForest@0.1 |  20 |     -0.8353 |      2 |       18 | 0.0001 |   0.0043 |
| clean              | LogReg       | remove:IsolationForest@0.1 |  20 |     -0.6342 |      5 |       15 | 0.0328 |   0.9175 |
| clean              | RandomForest | remove:IsolationForest@0.1 |  20 |     -0.7513 |      4 |       16 | 0.0049 |   0.1895 |
| feature_gross@0.05 | HistGB       | remove:IsolationForest@0.1 |  20 |     -0.2727 |      5 |       15 | 0.0484 |   1      |
| feature_gross@0.05 | KNN          | remove:IsolationForest@0.1 |  20 |     -0.301  |      7 |       13 | 0.1327 |   1      |
| feature_gross@0.05 | LogReg       | remove:IsolationForest@0.1 |  20 |      2.6356 |     19 |        0 | 0.0001 |   0.0067 |
| feature_gross@0.05 | RandomForest | remove:IsolationForest@0.1 |  20 |     -0.3824 |      4 |       15 | 0.0126 |   0.4408 |
| feature_gross@0.1  | HistGB       | remove:IsolationForest@0.1 |  20 |      0.4493 |     14 |        6 | 0.05   |   1      |
| feature_gross@0.1  | KNN          | remove:IsolationForest@0.1 |  20 |     -0.0974 |      7 |       12 | 0.4209 |   1      |
| feature_gross@0.1  | LogReg       | remove:IsolationForest@0.1 |  20 |      4.4669 |     19 |        1 | 0      |   0.0005 |
| feature_gross@0.1  | RandomForest | remove:IsolationForest@0.1 |  20 |      0.1245 |     11 |        9 | 0.4749 |   1      |
| feature_mild@0.05  | HistGB       | remove:IsolationForest@0.1 |  20 |     -0.4574 |      3 |       16 | 0.0329 |   0.9175 |
| feature_mild@0.05  | KNN          | remove:IsolationForest@0.1 |  20 |     -0.4378 |      4 |       16 | 0.0017 |   0.0744 |
| feature_mild@0.05  | LogReg       | remove:IsolationForest@0.1 |  20 |      0.8145 |     14 |        6 | 0.0037 |   0.1498 |
| feature_mild@0.05  | RandomForest | remove:IsolationForest@0.1 |  20 |     -0.1388 |      8 |       12 | 0.498  |   1      |
| feature_mild@0.1   | HistGB       | remove:IsolationForest@0.1 |  20 |      0.3882 |      9 |       10 | 0.6874 |   1      |
| feature_mild@0.1   | KNN          | remove:IsolationForest@0.1 |  20 |     -0.2875 |      6 |       12 | 0.0139 |   0.463  |
| feature_mild@0.1   | LogReg       | remove:IsolationForest@0.1 |  20 |      1.3388 |     17 |        3 | 0.0002 |   0.0082 |
| feature_mild@0.1   | RandomForest | remove:IsolationForest@0.1 |  20 |      0.1169 |     10 |       10 | 0.9563 |   1      |
| label_asym@0.1     | HistGB       | remove:DataOOB@0.1         |  20 |      0.816  |     11 |        9 | 0.1893 |   1      |
| label_asym@0.1     | KNN          | remove:DataOOB@0.1         |  20 |      1.0323 |     16 |        4 | 0.0192 |   0.6155 |
| label_asym@0.1     | LogReg       | remove:DataOOB@0.1         |  20 |      0.8414 |     15 |        5 | 0.0192 |   0.6155 |
| label_asym@0.1     | RandomForest | remove:DataOOB@0.1         |  20 |      0.4211 |     11 |        9 | 0.4524 |   1      |
| label_asym@0.2     | HistGB       | remove:DataOOB@0.1         |  20 |      0.1173 |     12 |        8 | 0.5958 |   1      |
| label_asym@0.2     | KNN          | remove:DataOOB@0.1         |  20 |      0.3222 |     12 |        8 | 0.5706 |   1      |
| label_asym@0.2     | LogReg       | remove:DataOOB@0.1         |  20 |      0.8732 |     12 |        7 | 0.0401 |   1      |
| label_asym@0.2     | RandomForest | remove:DataOOB@0.1         |  20 |      0.2164 |     12 |        8 | 0.6215 |   1      |
| label_sym@0.1      | HistGB       | remove:DataOOB@0.1         |  20 |      0.7632 |     14 |        6 | 0.0136 |   0.463  |
| label_sym@0.1      | KNN          | remove:DataOOB@0.1         |  20 |      0.8006 |     15 |        5 | 0.0064 |   0.244  |
| label_sym@0.1      | LogReg       | remove:DataOOB@0.1         |  20 |      1.7337 |     19 |        1 | 0      |   0.0002 |
| label_sym@0.1      | RandomForest | remove:DataOOB@0.1         |  20 |      0.2143 |     12 |        8 | 0.2943 |   1      |
| label_sym@0.2      | HistGB       | remove:DataOOB@0.1         |  20 |      1.8761 |     18 |        2 | 0      |   0.001  |
| label_sym@0.2      | KNN          | remove:DataOOB@0.1         |  20 |      1.6486 |     19 |        1 | 0.0004 |   0.0179 |
| label_sym@0.2      | LogReg       | remove:DataOOB@0.1         |  20 |      0.5968 |     16 |        4 | 0.0094 |   0.3397 |
| label_sym@0.2      | RandomForest | remove:DataOOB@0.1         |  20 |      1.0995 |     17 |        3 | 0.0012 |   0.0544 |
| mixed@0.05         | HistGB       | remove:DataOOB@0.1         |  20 |      0.3608 |     11 |        9 | 0.4091 |   1      |
| mixed@0.05         | KNN          | remove:DataOOB@0.1         |  20 |      0.2266 |     10 |       10 | 0.5459 |   1      |
| mixed@0.05         | LogReg       | remove:DataOOB@0.1         |  20 |      1.6191 |     20 |        0 | 0      |   0.0001 |
| mixed@0.05         | RandomForest | remove:DataOOB@0.1         |  20 |     -0.4317 |      7 |       13 | 0.2162 |   1      |
| mixed@0.05         | HistGB       | remove:IsolationForest@0.1 |  20 |     -0.3468 |      5 |       15 | 0.0484 |   1      |
| mixed@0.05         | KNN          | remove:IsolationForest@0.1 |  20 |     -0.421  |      7 |       13 | 0.0484 |   1      |
| mixed@0.05         | LogReg       | remove:IsolationForest@0.1 |  20 |      1.9941 |     14 |        6 | 0.0023 |   0.1    |
| mixed@0.05         | RandomForest | remove:IsolationForest@0.1 |  20 |     -0.302  |      6 |       14 | 0.0897 |   1      |
| mixed@0.1          | HistGB       | remove:DataOOB@0.1         |  20 |      1.0628 |     16 |        4 | 0.0266 |   0.7726 |
| mixed@0.1          | KNN          | remove:DataOOB@0.1         |  20 |      0.8643 |     16 |        4 | 0.0037 |   0.1498 |
| mixed@0.1          | LogReg       | remove:DataOOB@0.1         |  20 |      1.0598 |     18 |        2 | 0.0001 |   0.0067 |
| mixed@0.1          | RandomForest | remove:DataOOB@0.1         |  20 |      0.4272 |     15 |        5 | 0.0696 |   1      |
| mixed@0.1          | HistGB       | remove:IsolationForest@0.1 |  20 |     -0.3362 |      6 |       14 | 0.024  |   0.7185 |
| mixed@0.1          | KNN          | remove:IsolationForest@0.1 |  20 |     -0.2161 |      8 |       12 | 0.3683 |   1      |
| mixed@0.1          | LogReg       | remove:IsolationForest@0.1 |  20 |      3.8938 |     17 |        3 | 0.0002 |   0.0101 |
| mixed@0.1          | RandomForest | remove:IsolationForest@0.1 |  20 |     -0.1643 |      9 |       11 | 0.5256 |   1      |

## F. HEADLINE - Diagnose-then-treat vs. fixed strategies

Mean gain vs. no treatment (points), averaged per dataset. DTT thresholds are learned on the OTHER datasets (leave-one-dataset-out), so the DTT row is an honest estimate. The two oracle rows are upper bounds, not methods.

| strategy                     |   clean |   label |   feature |   mixed |   ALL |
|:-----------------------------|--------:|--------:|----------:|--------:|------:|
| never treat                  |   0     |   0     |     0     |   0     | 0     |
| always classical (IF 5%)     |  -0.298 |  -0.312 |     0.362 |   0.468 | 0.076 |
| always target-aware (OOB 5%) |  -0.697 |   0.492 |    -0.221 |   0.468 | 0.12  |
| always both (DTT sizes)      |  -0.493 |   0.632 |     0.083 |   1.136 | 0.422 |
| always reweight (OOB)        |   0.02  |   0.288 |     0.101 |   0.304 | 0.198 |
| DIAGNOSE-THEN-TREAT (LODO)   |  -0.44  |   0.615 |     0.12  |   1.126 | 0.432 |
| type oracle (knows type)     |   0     |   0.851 |     0.333 |   1.136 | 0.637 |
| action oracle (upper bound)  |   0.573 |   1.834 |     1.079 |   2.086 | 1.491 |

Harm rate (% of dataset x seed x condition x model cases with gain < -1 point):

| strategy                     |   clean |   label |   feature |   mixed |   ALL |
|:-----------------------------|--------:|--------:|----------:|--------:|------:|
| never treat                  |     0   |     0   |       0   |     0   |   0   |
| always classical (IF 5%)     |    22.1 |    26   |      11   |    12.3 |  17.7 |
| always target-aware (OOB 5%) |    34.6 |    10.7 |      24.7 |     9.8 |  17.8 |
| always both (DTT sizes)      |    32.1 |    24.8 |      27   |    19.6 |  25.3 |
| always reweight (OOB)        |     5.4 |     7.8 |       7.4 |     4.4 |   6.8 |
| DIAGNOSE-THEN-TREAT (LODO)   |    28.8 |    24.5 |      22.7 |    19.8 |  23.4 |
| type oracle (knows type)     |     0   |    18.1 |      14.5 |    19.6 |  15.4 |
| action oracle (upper bound)  |     0   |     0   |       0   |     0   |   0   |

Wilcoxon over datasets, ALL conditions: DTT vs never p = 0.027; DTT vs best fixed strategy ('always both (DTT sizes)') p = 0.5936.

Thresholds chosen per fold (median): theta_label = 0.048, theta_feature = 0.008.

## G. Best option of each family per condition (exploratory, post hoc)

| family                |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.05 |   feature_mild@0.1 |   label_asym@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.05 |   mixed@0.1 |
|:----------------------|--------:|---------------------:|--------------------:|--------------------:|-------------------:|-----------------:|-----------------:|----------------:|----------------:|-------------:|------------:|
| dtt_both              |   -0.63 |                 0.22 |                0.55 |               -0.39 |              -0.06 |             0.27 |             0.16 |            0.56 |            1.54 |         0.83 |        1.44 |
| dtt_classical         |   -0.27 |                 0.48 |                0.96 |               -0.17 |               0.06 |            -0.34 |            -0.01 |           -0.33 |           -0.38 |         0.31 |        0.78 |
| dtt_target            |   -0.52 |                -0.08 |               -0.16 |               -0.35 |              -0.09 |             0.48 |             0.31 |            0.81 |            1.81 |         0.45 |        0.72 |
| random-removal        |   -0.03 |                -0.04 |               -0.13 |               -0.03 |               0.08 |            -0    |             0.19 |           -0.04 |            0.05 |         0.03 |       -0.05 |
| remove-classical      |    0    |                 0.79 |                1.26 |                0.18 |               0.39 |             0.08 |             0.13 |           -0.14 |           -0.08 |         0.58 |        0.81 |
| remove-target-aware   |   -0.19 |                 0.01 |               -0.02 |               -0.17 |              -0.02 |             0.78 |             3.12 |            0.88 |            1.31 |         0.48 |        0.85 |
| reweight-classical    |    0.03 |                 0.47 |                0.63 |                0.14 |               0.36 |             0.11 |             0.31 |           -0.06 |            0.04 |         0.51 |        0.72 |
| reweight-target-aware |    0.03 |                 0.19 |                0.18 |                0.03 |               0.14 |             0.35 |             1.03 |            0.38 |            0.45 |         0.45 |        0.36 |
| winsorize             |    0.02 |                 0.61 |                0.33 |                0.15 |               0.1  |             0.02 |             0.07 |            0.04 |           -0.02 |         0.46 |        0.35 |

## Notes
failures.log: 20 lines (20 removals skipped: they deleted a whole class).
