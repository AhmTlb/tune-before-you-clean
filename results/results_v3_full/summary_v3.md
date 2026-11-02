# Experiment v3: summary

Datasets: 40 | seeds: 5 | conditions: clean, feature_gross@0.05, feature_gross@0.1, feature_mild@0.05, feature_mild@0.1, label_asym@0.1, label_asym@0.2, label_sym@0.1, label_sym@0.2, mixed@0.05, mixed@0.1

Accuracy = balanced accuracy in points (x100); datasets weigh equally.

## A. RQ1 - Overlap of top-5% flagged points (clean)

Random expectation of Jaccard: 0.026

| pair_type           |   jaccard |   spearman |
|:--------------------|----------:|-----------:|
| cross-family        |     0.027 |      0.119 |
| within-classical    |     0.264 |      0.583 |
| within-target-aware |     0.349 |      0.551 |

## B. RQ4a - AUROC for detecting injected corrupted rows

Column = condition [which corrupted rows are the target]. 0.5 = chance.

| method            |   feature_gross@0.05 [feature] |   feature_gross@0.1 [feature] |   feature_mild@0.05 [feature] |   feature_mild@0.1 [feature] |   label_asym@0.1 [label] |   label_asym@0.2 [label] |   label_sym@0.1 [label] |   label_sym@0.2 [label] |   mixed@0.05 [feature] |   mixed@0.05 [label] |   mixed@0.1 [feature] |   mixed@0.1 [label] |
|:------------------|-------------------------------:|------------------------------:|------------------------------:|-----------------------------:|-------------------------:|-------------------------:|------------------------:|------------------------:|-----------------------:|---------------------:|----------------------:|--------------------:|
| IQR               |                          0.956 |                         0.958 |                         0.885 |                        0.886 |                    0.528 |                    0.529 |                   0.499 |                   0.499 |                  0.955 |                0.48  |                 0.957 |               0.447 |
| MAD               |                          0.92  |                         0.923 |                         0.839 |                        0.839 |                    0.54  |                    0.54  |                   0.498 |                   0.501 |                  0.917 |                0.483 |                 0.923 |               0.451 |
| RobustMahalanobis |                          0.989 |                         0.991 |                         0.965 |                        0.969 |                    0.549 |                    0.551 |                   0.499 |                   0.498 |                  0.989 |                0.483 |                 0.991 |               0.444 |
| LOF               |                          0.992 |                         0.992 |                         0.984 |                        0.984 |                    0.519 |                    0.52  |                   0.498 |                   0.499 |                  0.992 |                0.478 |                 0.992 |               0.443 |
| IsolationForest   |                          0.994 |                         0.996 |                         0.974 |                        0.98  |                    0.552 |                    0.554 |                   0.5   |                   0.5   |                  0.994 |                0.481 |                 0.996 |               0.443 |
| DataOOB           |                          0.729 |                         0.729 |                         0.724 |                        0.722 |                    0.822 |                    0.779 |                   0.92  |                   0.898 |                  0.701 |                0.925 |                 0.649 |               0.913 |
| KNNShapley        |                          0.793 |                         0.794 |                         0.783 |                        0.783 |                    0.741 |                    0.743 |                   0.883 |                   0.884 |                  0.765 |                0.881 |                 0.73  |               0.879 |
| SelfConfidence    |                          0.73  |                         0.732 |                         0.726 |                        0.725 |                    0.822 |                    0.781 |                   0.923 |                   0.902 |                  0.696 |                0.928 |                 0.64  |               0.917 |

## C. RQ3 - Who gets flagged on clean data (top-5%)

| method            |   ('minority_enrichment', 'mean') |   ('minority_enrichment', 'median') |   ('hardness_ratio', 'mean') |   ('hardness_ratio', 'median') |
|:------------------|----------------------------------:|------------------------------------:|-----------------------------:|-------------------------------:|
| IQR               |                              1.19 |                                0.8  |                         1.1  |                           1.01 |
| MAD               |                              1.35 |                                1.03 |                         1.07 |                           0.93 |
| RobustMahalanobis |                              1.45 |                                1.04 |                         1.09 |                           1.02 |
| LOF               |                              0.95 |                                0.95 |                         1.52 |                           1.15 |
| IsolationForest   |                              1.35 |                                1.2  |                         1.2  |                           1.05 |
| DataOOB           |                              3.19 |                                1.92 |                         3.97 |                           3.4  |
| KNNShapley        |                              4.06 |                                2.94 |                         3.8  |                           3.1  |
| SelfConfidence    |                              3.54 |                                1.99 |                         4.34 |                           3.59 |

## D. RQ6 - Can corruption be diagnosed without knowing it?

Spearman(est_label_noise, true label rate) = **0.51**; Spearman(extreme_rate, true feature rate) = **0.21** (over all dataset x seed x condition cases).

On clean data both diagnostics should be low; their clean level is the 'false alarm' baseline (inherent class overlap inflates est_label_noise).

| condition          |   ('est_label_noise', 'mean') |   ('est_label_noise', 'std') |   ('true_label_rate', 'mean') |   ('true_label_rate', 'std') |   ('extreme_rate', 'mean') |   ('extreme_rate', 'std') |   ('true_feature_rate', 'mean') |   ('true_feature_rate', 'std') |
|:-------------------|------------------------------:|-----------------------------:|------------------------------:|-----------------------------:|---------------------------:|--------------------------:|--------------------------------:|-------------------------------:|
| clean              |                         0.13  |                        0.164 |                         0     |                        0     |                      0.154 |                     0.185 |                            0    |                              0 |
| feature_gross@0.05 |                         0.132 |                        0.166 |                         0     |                        0     |                      0.194 |                     0.186 |                            0.05 |                              0 |
| feature_gross@0.1  |                         0.136 |                        0.167 |                         0     |                        0     |                      0.231 |                     0.184 |                            0.1  |                              0 |
| feature_mild@0.05  |                         0.132 |                        0.165 |                         0     |                        0     |                      0.168 |                     0.199 |                            0.05 |                              0 |
| feature_mild@0.1   |                         0.135 |                        0.166 |                         0     |                        0     |                      0.184 |                     0.215 |                            0.1  |                              0 |
| label_asym@0.1     |                         0.2   |                        0.15  |                         0.092 |                        0.02  |                      0.154 |                     0.185 |                            0    |                              0 |
| label_asym@0.2     |                         0.257 |                        0.151 |                         0.161 |                        0.056 |                      0.154 |                     0.185 |                            0    |                              0 |
| label_sym@0.1      |                         0.242 |                        0.15  |                         0.1   |                        0     |                      0.154 |                     0.185 |                            0    |                              0 |
| label_sym@0.2      |                         0.348 |                        0.125 |                         0.2   |                        0     |                      0.154 |                     0.185 |                            0    |                              0 |
| mixed@0.05         |                         0.195 |                        0.158 |                         0.05  |                        0     |                      0.19  |                     0.179 |                            0.05 |                              0 |
| mixed@0.1          |                         0.258 |                        0.149 |                         0.1   |                        0     |                      0.23  |                     0.181 |                            0.1  |                              0 |

## E. PRIMARY pre-specified tests (matched family, remove 10%, vs no treatment)

Holm correction over these 56 tests only (fixed in PROTOCOL_v3.md before the run).

| condition          | model        | option                     |   n |   mean_gain |   wins |   losses |      p |   p_holm |
|:-------------------|:-------------|:---------------------------|----:|------------:|-------:|---------:|-------:|---------:|
| clean              | HistGB       | remove:DataOOB@0.1         |  40 |     -3.0988 |      7 |       33 | 0      |   0      |
| clean              | KNN          | remove:DataOOB@0.1         |  40 |     -1.3958 |     13 |       27 | 0.0005 |   0.0192 |
| clean              | LogReg       | remove:DataOOB@0.1         |  40 |     -0.85   |     13 |       26 | 0.0158 |   0.4731 |
| clean              | RandomForest | remove:DataOOB@0.1         |  40 |     -2.1065 |      7 |       33 | 0      |   0      |
| clean              | HistGB       | remove:IsolationForest@0.1 |  40 |     -0.9968 |     10 |       30 | 0.0012 |   0.0455 |
| clean              | KNN          | remove:IsolationForest@0.1 |  40 |     -1.1586 |     11 |       29 | 0      |   0.0013 |
| clean              | LogReg       | remove:IsolationForest@0.1 |  40 |     -0.435  |     13 |       27 | 0.0662 |   1      |
| clean              | RandomForest | remove:IsolationForest@0.1 |  40 |     -1.0262 |     13 |       27 | 0.0014 |   0.052  |
| feature_gross@0.05 | HistGB       | remove:IsolationForest@0.1 |  40 |     -0.4967 |     11 |       29 | 0.0032 |   0.112  |
| feature_gross@0.05 | KNN          | remove:IsolationForest@0.1 |  40 |     -0.5527 |     12 |       28 | 0.002  |   0.0725 |
| feature_gross@0.05 | LogReg       | remove:IsolationForest@0.1 |  40 |      3.1334 |     32 |        8 | 0      |   0.0001 |
| feature_gross@0.05 | RandomForest | remove:IsolationForest@0.1 |  40 |     -0.2534 |     14 |       26 | 0.0867 |   1      |
| feature_gross@0.1  | HistGB       | remove:IsolationForest@0.1 |  40 |      0.0412 |     16 |       24 | 0.4929 |   1      |
| feature_gross@0.1  | KNN          | remove:IsolationForest@0.1 |  40 |      0.0721 |     21 |       19 | 0.5808 |   1      |
| feature_gross@0.1  | LogReg       | remove:IsolationForest@0.1 |  40 |      5.2101 |     38 |        2 | 0      |   0      |
| feature_gross@0.1  | RandomForest | remove:IsolationForest@0.1 |  40 |      0.0963 |     23 |       17 | 0.39   |   1      |
| feature_mild@0.05  | HistGB       | remove:IsolationForest@0.1 |  40 |     -0.534  |     10 |       30 | 0.0076 |   0.2579 |
| feature_mild@0.05  | KNN          | remove:IsolationForest@0.1 |  40 |     -0.5749 |     10 |       29 | 0.0002 |   0.0077 |
| feature_mild@0.05  | LogReg       | remove:IsolationForest@0.1 |  40 |      1.1551 |     28 |       12 | 0.0024 |   0.0861 |
| feature_mild@0.05  | RandomForest | remove:IsolationForest@0.1 |  40 |     -0.1561 |     17 |       23 | 0.3138 |   1      |
| feature_mild@0.1   | HistGB       | remove:IsolationForest@0.1 |  40 |     -0.1785 |     18 |       21 | 0.5959 |   1      |
| feature_mild@0.1   | KNN          | remove:IsolationForest@0.1 |  40 |     -0.3431 |     16 |       23 | 0.0285 |   0.7967 |
| feature_mild@0.1   | LogReg       | remove:IsolationForest@0.1 |  40 |      1.7557 |     32 |        8 | 0      |   0.0002 |
| feature_mild@0.1   | RandomForest | remove:IsolationForest@0.1 |  40 |      0.0955 |     21 |       18 | 0.468  |   1      |
| label_asym@0.1     | HistGB       | remove:DataOOB@0.1         |  37 |      0.1441 |     20 |       17 | 0.7204 |   1      |
| label_asym@0.1     | KNN          | remove:DataOOB@0.1         |  37 |      0.057  |     20 |       17 | 0.8698 |   1      |
| label_asym@0.1     | LogReg       | remove:DataOOB@0.1         |  37 |      0.7821 |     21 |       15 | 0.1162 |   1      |
| label_asym@0.1     | RandomForest | remove:DataOOB@0.1         |  37 |     -0.1839 |     14 |       22 | 0.4796 |   1      |
| label_asym@0.2     | HistGB       | remove:DataOOB@0.1         |  37 |     -0.5989 |     19 |       18 | 0.4642 |   1      |
| label_asym@0.2     | KNN          | remove:DataOOB@0.1         |  37 |     -0.1566 |     18 |       19 | 0.7884 |   1      |
| label_asym@0.2     | LogReg       | remove:DataOOB@0.1         |  37 |      0.0804 |     19 |       17 | 0.4414 |   1      |
| label_asym@0.2     | RandomForest | remove:DataOOB@0.1         |  37 |     -0.2491 |     16 |       21 | 0.316  |   1      |
| label_sym@0.1      | HistGB       | remove:DataOOB@0.1         |  40 |      0.531  |     28 |       12 | 0.013  |   0.4168 |
| label_sym@0.1      | KNN          | remove:DataOOB@0.1         |  40 |      0.283  |     26 |       14 | 0.0945 |   1      |
| label_sym@0.1      | LogReg       | remove:DataOOB@0.1         |  40 |      2.1854 |     38 |        2 | 0      |   0      |
| label_sym@0.1      | RandomForest | remove:DataOOB@0.1         |  40 |     -0.067  |     23 |       17 | 0.9464 |   1      |
| label_sym@0.2      | HistGB       | remove:DataOOB@0.1         |  40 |      1.7144 |     35 |        5 | 0      |   0.0001 |
| label_sym@0.2      | KNN          | remove:DataOOB@0.1         |  40 |      1.4206 |     33 |        7 | 0      |   0      |
| label_sym@0.2      | LogReg       | remove:DataOOB@0.1         |  40 |      1.3395 |     31 |        9 | 0      |   0.0015 |
| label_sym@0.2      | RandomForest | remove:DataOOB@0.1         |  40 |      0.4899 |     25 |       15 | 0.0381 |   1      |
| mixed@0.05         | HistGB       | remove:DataOOB@0.1         |  40 |     -0.4457 |     19 |       21 | 0.2016 |   1      |
| mixed@0.05         | KNN          | remove:DataOOB@0.1         |  40 |     -0.4648 |     15 |       25 | 0.1387 |   1      |
| mixed@0.05         | LogReg       | remove:DataOOB@0.1         |  40 |      1.9765 |     33 |        7 | 0      |   0      |
| mixed@0.05         | RandomForest | remove:DataOOB@0.1         |  40 |     -1.034  |      9 |       31 | 0.0002 |   0.009  |
| mixed@0.05         | HistGB       | remove:IsolationForest@0.1 |  40 |     -0.4308 |     14 |       26 | 0.0086 |   0.2845 |
| mixed@0.05         | KNN          | remove:IsolationForest@0.1 |  40 |     -0.3545 |     12 |       28 | 0.0269 |   0.7787 |
| mixed@0.05         | LogReg       | remove:IsolationForest@0.1 |  40 |      2.5578 |     30 |       10 | 0      |   0.0013 |
| mixed@0.05         | RandomForest | remove:IsolationForest@0.1 |  40 |     -0.154  |     15 |       25 | 0.1787 |   1      |
| mixed@0.1          | HistGB       | remove:DataOOB@0.1         |  40 |      0.7594 |     27 |       12 | 0.0146 |   0.4526 |
| mixed@0.1          | KNN          | remove:DataOOB@0.1         |  40 |      0.3179 |     28 |       11 | 0.1148 |   1      |
| mixed@0.1          | LogReg       | remove:DataOOB@0.1         |  40 |      1.7101 |     36 |        4 | 0      |   0      |
| mixed@0.1          | RandomForest | remove:DataOOB@0.1         |  40 |     -0.243  |     18 |       21 | 0.4103 |   1      |
| mixed@0.1          | HistGB       | remove:IsolationForest@0.1 |  40 |     -0.2321 |     17 |       23 | 0.1088 |   1      |
| mixed@0.1          | KNN          | remove:IsolationForest@0.1 |  40 |     -0.1488 |     17 |       23 | 0.1357 |   1      |
| mixed@0.1          | LogReg       | remove:IsolationForest@0.1 |  40 |      3.9969 |     33 |        7 | 0      |   0      |
| mixed@0.1          | RandomForest | remove:IsolationForest@0.1 |  40 |     -0.1073 |     20 |       20 | 0.6559 |   1      |

## F. HEADLINE - Diagnose-then-treat vs. fixed strategies

Mean gain vs. no treatment (points), averaged per dataset. DTT thresholds are learned on the OTHER datasets (leave-one-dataset-out), so the DTT row is an honest estimate. The two oracle rows are upper bounds, not methods.

| strategy                     |   clean |   label |   feature |   mixed |    ALL |
|:-----------------------------|--------:|--------:|----------:|--------:|-------:|
| never treat                  |   0     |   0     |     0     |   0     |  0     |
| always classical (IF 5%)     |  -0.328 |  -0.24  |     0.457 |   0.527 |  0.145 |
| always target-aware (OOB 5%) |  -1.032 |   0.222 |    -0.662 |   0.477 | -0.167 |
| always both (DTT sizes)      |  -1.534 |   0.243 |    -0.701 |   0.539 | -0.208 |
| always reweight (OOB)        |  -0.087 |   0.187 |     0.027 |   0.298 |  0.124 |
| DIAGNOSE-THEN-TREAT (LODO)   |  -0.237 |  -0.029 |     0.352 |   0.45  |  0.178 |
| type oracle (knows type)     |   0     |   0.451 |     0.322 |   0.539 |  0.379 |
| action oracle (upper bound)  |   0.928 |   2.179 |     1.457 |   2.382 |  1.84  |

Harm rate (% of dataset x seed x condition x model cases with gain < -1 point):

| strategy                     |   clean |   label |   feature |   mixed |   ALL |
|:-----------------------------|--------:|--------:|----------:|--------:|------:|
| never treat                  |     0   |     0   |       0   |     0   |   0   |
| always classical (IF 5%)     |    27.5 |    30.2 |      15.8 |    19.1 |  22.7 |
| always target-aware (OOB 5%) |    41.6 |    19.8 |      35.2 |    15.6 |  26.6 |
| always both (DTT sizes)      |    46.9 |    29.5 |      39.4 |    31.7 |  35.1 |
| always reweight (OOB)        |    13.2 |    13   |      11.5 |     9   |  11.8 |
| DIAGNOSE-THEN-TREAT (LODO)   |    22.9 |    23.1 |      20.8 |    27.3 |  23   |
| type oracle (knows type)     |     0   |    25.7 |      22.6 |    31.7 |  23.3 |
| action oracle (upper bound)  |     0   |     0   |       0   |     0   |   0   |

Wilcoxon over datasets, ALL conditions: DTT vs never p = 0.075; DTT vs best fixed strategy ('always classical (IF 5%)') p = 0.2265.

Thresholds chosen per fold (median): theta_label = 0.408, theta_feature = 0.013.

## G. Best option of each family per condition (exploratory, post hoc)

| family                |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.05 |   feature_mild@0.1 |   label_asym@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.05 |   mixed@0.1 |
|:----------------------|--------:|---------------------:|--------------------:|--------------------:|-------------------:|-----------------:|-----------------:|----------------:|----------------:|-------------:|------------:|
| dtt_both              |   -1.56 |                -0.73 |               -0.07 |               -1.31 |              -0.78 |            -0.2  |            -0.2  |           -0.1  |            1.33 |        -0.14 |        1.14 |
| dtt_classical         |   -0.3  |                 0.31 |                1.07 |               -0.26 |               0.17 |            -0.08 |             0.21 |           -0.2  |           -0.5  |         0.1  |        0.7  |
| dtt_target            |   -1.06 |                -0.82 |               -0.83 |               -0.9  |              -0.82 |            -0.14 |            -0.31 |            0.36 |            1.68 |        -0.16 |        0.32 |
| random-removal        |   -0.04 |                -0.03 |               -0.11 |               -0.02 |               0.03 |            -0.09 |            -0.02 |            0.03 |           -0.02 |        -0.05 |       -0.04 |
| remove-classical      |   -0.02 |                 0.81 |                1.35 |                0.27 |               0.52 |            -0.02 |             0.03 |            0.01 |           -0.08 |         0.63 |        0.88 |
| remove-target-aware   |   -0.47 |                -0.19 |               -0.14 |               -0.27 |              -0.24 |             0.2  |             2.17 |            0.73 |            1.24 |         0.43 |        0.64 |
| reweight-classical    |    0.02 |                 0.41 |                0.58 |                0.15 |               0.37 |             0.06 |             0.12 |            0.04 |            0.06 |         0.43 |        0.6  |
| reweight-target-aware |   -0.12 |                 0.05 |                0.07 |                0    |               0.02 |             0.15 |             0.43 |            0.48 |            0.44 |         0.32 |        0.47 |
| winsorize             |    0.07 |                 0.68 |                0.54 |                0.24 |               0.29 |             0.05 |             0.07 |            0.09 |            0.07 |         0.55 |        0.44 |

## Notes
failures.log: 444 lines (444 removals skipped: they deleted a whole class).
