# Experiment v2: summary

Datasets: 20 | seeds: 3 | conditions: clean, feature@0.05, feature@0.1, label@0.05, label@0.1, label@0.2

All accuracy figures: balanced accuracy in points (x100), datasets weighted equally.

## A. RQ1 - Overlap of top-5% flagged points (clean data)

Random expectation of Jaccard: 0.026

| pair_type           |   ('jaccard', 'mean') |   ('jaccard', 'std') |   ('spearman', 'mean') |   ('spearman', 'std') |
|:--------------------|----------------------:|---------------------:|-----------------------:|----------------------:|
| cross-family        |                 0.031 |                0.027 |                  0.076 |                 0.157 |
| within-classical    |                 0.275 |                0.11  |                  0.582 |                 0.1   |
| within-target-aware |                 0.324 |                0.102 |                  0.588 |                 0.131 |

## B. RQ4a - Detection of injected corruption

AUROC (0.5 = chance) and precision@k (k = true number of corrupted rows).

| method            |   ('auroc', 'feature@0.05') |   ('auroc', 'feature@0.1') |   ('auroc', 'label@0.05') |   ('auroc', 'label@0.1') |   ('auroc', 'label@0.2') |   ('precision_at_k', 'feature@0.05') |   ('precision_at_k', 'feature@0.1') |   ('precision_at_k', 'label@0.05') |   ('precision_at_k', 'label@0.1') |   ('precision_at_k', 'label@0.2') |
|:------------------|----------------------------:|---------------------------:|--------------------------:|-------------------------:|-------------------------:|-------------------------------------:|------------------------------------:|-----------------------------------:|----------------------------------:|----------------------------------:|
| IQR               |                       0.969 |                      0.971 |                     0.5   |                    0.495 |                    0.503 |                                0.781 |                               0.839 |                              0.045 |                             0.103 |                             0.201 |
| MAD               |                       0.953 |                      0.96  |                     0.497 |                    0.496 |                    0.506 |                                0.777 |                               0.834 |                              0.053 |                             0.101 |                             0.207 |
| RobustMahalanobis |                       0.995 |                      0.994 |                     0.515 |                    0.5   |                    0.502 |                                0.9   |                               0.93  |                              0.056 |                             0.107 |                             0.197 |
| LOF               |                       0.996 |                      0.996 |                     0.504 |                    0.499 |                    0.5   |                                0.921 |                               0.95  |                              0.049 |                             0.097 |                             0.2   |
| IsolationForest   |                       0.999 |                      0.999 |                     0.503 |                    0.498 |                    0.503 |                                0.943 |                               0.983 |                              0.049 |                             0.103 |                             0.196 |
| DataOOB           |                       0.745 |                      0.747 |                     0.938 |                    0.931 |                    0.914 |                                0.175 |                               0.256 |                              0.644 |                             0.7   |                             0.733 |
| KNNShapley        |                       0.802 |                      0.803 |                     0.892 |                    0.893 |                    0.892 |                                0.061 |                               0.138 |                              0.493 |                             0.59  |                             0.698 |
| SelfConfidence    |                       0.742 |                      0.746 |                     0.941 |                    0.937 |                    0.921 |                                0.137 |                               0.228 |                              0.661 |                             0.719 |                             0.751 |

## C. RQ3 - Who gets flagged? (clean data, top-5%)

minority_enrichment = share of smallest class among flagged / its share in T (1 = no bias). hardness_ratio = mean kDN of flagged / mean kDN of all (>1 = flagged points are harder, i.e. near class boundaries).

| method            |   ('minority_enrichment', 'mean') |   ('minority_enrichment', 'median') |   ('hardness_ratio', 'mean') |   ('hardness_ratio', 'median') |   ('kdn_flagged', 'mean') |   ('kdn_flagged', 'median') |
|:------------------|----------------------------------:|------------------------------------:|-----------------------------:|-------------------------------:|--------------------------:|----------------------------:|
| IQR               |                              0.72 |                                0.41 |                         0.99 |                           0.86 |                      0.2  |                        0.2  |
| MAD               |                              0.79 |                                0.65 |                         0.93 |                           0.83 |                      0.2  |                        0.2  |
| RobustMahalanobis |                              0.85 |                                0.51 |                         0.99 |                           0.77 |                      0.21 |                        0.2  |
| LOF               |                              1.05 |                                1.02 |                         1.88 |                           1.25 |                      0.32 |                        0.32 |
| IsolationForest   |                              0.95 |                                0.82 |                         1.05 |                           0.98 |                      0.22 |                        0.22 |
| DataOOB           |                              1.78 |                                1.1  |                         3.96 |                           3.01 |                      0.68 |                        0.75 |
| KNNShapley        |                              2.37 |                                1.57 |                         3.56 |                           2.66 |                      0.62 |                        0.64 |
| SelfConfidence    |                              1.99 |                                1.23 |                         4.28 |                           3.19 |                      0.72 |                        0.79 |

## D. RQ2/RQ4b - Top-3 treatments per condition and model (gain vs. no treatment)

p: Wilcoxon signed-rank across datasets; p_holm: Holm-corrected over ALL 1056 tests (see v2_gains_all.csv for every option).

| condition    | model        | option                        |   n |   mean_gain |   median_gain |   wins |   losses |      p |   p_holm |
|:-------------|:-------------|:------------------------------|----:|------------:|--------------:|-------:|---------:|-------:|---------:|
| clean        | HistGB       | reweight:IQR                  |  20 |      0.1885 |       -0.0442 |      9 |       10 | 0.4209 |   1      |
| clean        | HistGB       | reweight:SelfConfidence       |  20 |      0.15   |        0      |      9 |        9 | 0.6475 |   1      |
| clean        | HistGB       | remove:IQR@0.02               |  20 |      0.1067 |       -0.2188 |      7 |       13 | 0.7841 |   1      |
| clean        | KNN          | winsorize:winsor_p1_99        |  20 |      0.0471 |        0      |      7 |        8 | 0.6701 |   1      |
| clean        | KNN          | remove:MAD@0.02               |  20 |      0.0156 |        0.0217 |     11 |        9 | 0.8695 |   1      |
| clean        | KNN          | remove:IQR@0.02               |  20 |     -0.0393 |       -0.0333 |      8 |       10 | 0.4724 |   1      |
| clean        | LogReg       | remove:DataOOB@0.02           |  20 |      0.0523 |       -0.0248 |      8 |       11 | 0.9039 |   1      |
| clean        | LogReg       | remove:SelfConfidence@0.02    |  20 |      0.0018 |       -0.0259 |      9 |       10 | 0.6874 |   1      |
| clean        | LogReg       | remove:random@0.02            |  20 |     -0.0053 |        0      |      8 |        8 | 0.9588 |   1      |
| clean        | RandomForest | remove:random@0.02            |  20 |      0.1343 |       -0.0044 |     10 |       10 | 0.9563 |   1      |
| clean        | RandomForest | remove:random@0.05            |  20 |      0.1085 |        0.0076 |     10 |        9 | 0.7172 |   1      |
| clean        | RandomForest | remove:MAD@0.02               |  20 |      0.0763 |       -0.0302 |     10 |       10 | 0.9256 |   1      |
| feature@0.05 | HistGB       | reweight:IsolationForest      |  20 |      0.2509 |        0.1338 |     12 |        8 | 0.2024 |   1      |
| feature@0.05 | HistGB       | reweight:SelfConfidence       |  20 |      0.2046 |        0.0453 |     10 |        9 | 0.6292 |   1      |
| feature@0.05 | HistGB       | remove:RobustMahalanobis@0.05 |  20 |      0.1726 |        0.1388 |     11 |        9 | 0.4304 |   1      |
| feature@0.05 | KNN          | remove:random@0.02            |  20 |      0.1164 |        0.0333 |     10 |        7 | 0.5862 |   1      |
| feature@0.05 | KNN          | remove:SelfConfidence@0.01    |  20 |      0.1155 |        0.0124 |     10 |        9 | 0.3341 |   1      |
| feature@0.05 | KNN          | remove:DataOOB@0.01           |  20 |      0.0849 |        0.015  |     10 |        8 | 0.4204 |   1      |
| feature@0.05 | LogReg       | remove:IsolationForest@0.1    |  20 |      2.5177 |        2.3556 |     19 |        1 | 0      |   0.0139 |
| feature@0.05 | LogReg       | remove:IsolationForest@0.05   |  20 |      2.5036 |        1.8788 |     20 |        0 | 0      |   0.002  |
| feature@0.05 | LogReg       | remove:LOF@0.05               |  20 |      2.4166 |        2.1999 |     18 |        2 | 0      |   0.0199 |
| feature@0.05 | RandomForest | remove:RobustMahalanobis@0.02 |  20 |      0.2131 |        0.0389 |     10 |       10 | 0.498  |   1      |
| feature@0.05 | RandomForest | remove:IsolationForest@0.05   |  20 |      0.2106 |        0.1979 |     13 |        7 | 0.1305 |   1      |
| feature@0.05 | RandomForest | remove:RobustMahalanobis@0.05 |  20 |      0.2072 |        0.1822 |     13 |        7 | 0.0637 |   1      |
| feature@0.1  | HistGB       | remove:MAD@0.1                |  20 |      0.3768 |        0.0675 |     12 |        8 | 0.5459 |   1      |
| feature@0.1  | HistGB       | remove:IsolationForest@0.1    |  20 |      0.3724 |        0.0732 |     13 |        7 | 0.5379 |   1      |
| feature@0.1  | HistGB       | remove:IQR@0.1                |  20 |      0.3391 |       -0.1464 |      9 |       11 | 0.8408 |   1      |
| feature@0.1  | KNN          | remove:MAD@0.02               |  20 |      0.1934 |        0.0576 |     12 |        7 | 0.2432 |   1      |
| feature@0.1  | KNN          | remove:DataOOB@0.02           |  20 |      0.1647 |        0.1432 |     12 |        5 | 0.084  |   1      |
| feature@0.1  | KNN          | remove:IsolationForest@0.1    |  20 |      0.159  |        0.0003 |     11 |        8 | 0.2432 |   1      |
| feature@0.1  | LogReg       | remove:IsolationForest@0.1    |  20 |      4.3932 |        3.9023 |     20 |        0 | 0      |   0.002  |
| feature@0.1  | LogReg       | remove:LOF@0.1                |  20 |      4.2653 |        3.5305 |     20 |        0 | 0      |   0.002  |
| feature@0.1  | LogReg       | remove:RobustMahalanobis@0.1  |  20 |      4.0898 |        3.5    |     18 |        2 | 0      |   0.01   |
| feature@0.1  | RandomForest | remove:IsolationForest@0.05   |  20 |      0.4507 |        0.2271 |     15 |        5 | 0.0072 |   1      |
| feature@0.1  | RandomForest | remove:IQR@0.05               |  20 |      0.4287 |        0.1934 |     13 |        7 | 0.0215 |   1      |
| feature@0.1  | RandomForest | reweight:RobustMahalanobis    |  20 |      0.3494 |        0.2993 |     14 |        5 | 0.0112 |   1      |
| label@0.05   | HistGB       | remove:SelfConfidence@0.05    |  20 |      0.5657 |        0.602  |     14 |        5 | 0.0079 |   1      |
| label@0.05   | HistGB       | reweight:KNNShapley           |  20 |      0.4664 |        0.2124 |     14 |        5 | 0.091  |   1      |
| label@0.05   | HistGB       | reweight:DataOOB              |  20 |      0.4589 |        0.3471 |     15 |        5 | 0.0056 |   1      |
| label@0.05   | KNN          | remove:DataOOB@0.05           |  20 |      0.2659 |        0.1716 |     12 |        8 | 0.1536 |   1      |
| label@0.05   | KNN          | remove:KNNShapley@0.02        |  20 |      0.1447 |        0.0602 |     14 |        4 | 0.0526 |   1      |
| label@0.05   | KNN          | remove:SelfConfidence@0.02    |  20 |      0.0607 |        0.0644 |     13 |        6 | 0.1712 |   1      |
| label@0.05   | LogReg       | remove:DataOOB@0.1            |  20 |      1.1723 |        0.5331 |     14 |        6 | 0.0266 |   1      |
| label@0.05   | LogReg       | remove:SelfConfidence@0.05    |  20 |      1.087  |        0.6363 |     16 |        4 | 0.0073 |   1      |
| label@0.05   | LogReg       | remove:SelfConfidence@0.1     |  20 |      1.0436 |        0.4355 |     14 |        6 | 0.0532 |   1      |
| label@0.05   | RandomForest | remove:DataOOB@0.05           |  20 |      0.04   |        0.027  |     10 |       10 | 0.6012 |   1      |
| label@0.05   | RandomForest | remove:random@0.05            |  20 |      0.0372 |        0.1061 |     12 |        8 | 0.498  |   1      |
| label@0.05   | RandomForest | remove:DataOOB@0.02           |  20 |      0.0217 |       -0.1196 |      8 |       12 | 0.7841 |   1      |
| label@0.1    | HistGB       | remove:DataOOB@0.1            |  20 |      1.0616 |        1.1886 |     17 |        3 | 0.0012 |   1      |
| label@0.1    | HistGB       | remove:SelfConfidence@0.1     |  20 |      0.8176 |        1.1419 |     15 |        5 | 0.0637 |   1      |
| label@0.1    | HistGB       | reweight:DataOOB              |  20 |      0.7807 |        0.6486 |     16 |        4 | 0.0005 |   0.479  |
| label@0.1    | KNN          | remove:DataOOB@0.1            |  20 |      0.863  |        0.7742 |     15 |        5 | 0.0136 |   1      |
| label@0.1    | KNN          | remove:SelfConfidence@0.1     |  20 |      0.8362 |        0.6    |     14 |        6 | 0.0094 |   1      |
| label@0.1    | KNN          | remove:DataOOB@0.05           |  20 |      0.6194 |        0.2111 |     13 |        6 | 0.0298 |   1      |
| label@0.1    | LogReg       | remove:SelfConfidence@0.1     |  20 |      1.6315 |        1.4719 |     13 |        7 | 0.0073 |   1      |
| label@0.1    | LogReg       | remove:DataOOB@0.1            |  20 |      1.5374 |        1.4814 |     15 |        5 | 0.001  |   0.9892 |
| label@0.1    | LogReg       | reweight:SelfConfidence       |  20 |      0.8598 |        0.6206 |     16 |        4 | 0.0017 |   1      |
| label@0.1    | RandomForest | remove:DataOOB@0.02           |  20 |      0.1313 |        0.1309 |     13 |        7 | 0.2611 |   1      |
| label@0.1    | RandomForest | winsorize:winsor_p1_99        |  20 |      0.1251 |        0      |      8 |        6 | 0.1401 |   1      |
| label@0.1    | RandomForest | reweight:MAD                  |  20 |      0.065  |        0.1139 |     11 |        9 | 0.6477 |   1      |
| label@0.2    | HistGB       | remove:DataOOB@0.1            |  20 |      2.1026 |        1.9018 |     17 |        3 | 0      |   0.0374 |
| label@0.2    | HistGB       | remove:SelfConfidence@0.1     |  20 |      2.0229 |        1.6643 |     20 |        0 | 0      |   0.002  |
| label@0.2    | HistGB       | remove:KNNShapley@0.1         |  20 |      1.7318 |        1.6752 |     18 |        2 | 0.0009 |   0.8337 |
| label@0.2    | KNN          | remove:DataOOB@0.1            |  20 |      2.2946 |        2.2931 |     19 |        1 | 0      |   0.006  |
| label@0.2    | KNN          | remove:SelfConfidence@0.1     |  20 |      2.0946 |        1.8109 |     19 |        1 | 0      |   0.01   |
| label@0.2    | KNN          | remove:KNNShapley@0.1         |  20 |      1.5937 |        1.7104 |     18 |        2 | 0.0007 |   0.6984 |
| label@0.2    | LogReg       | remove:DataOOB@0.1            |  20 |      0.4935 |        0.2772 |     14 |        5 | 0.0112 |   1      |
| label@0.2    | LogReg       | reweight:DataOOB              |  20 |      0.4609 |        0.4078 |     16 |        3 | 0.0062 |   1      |
| label@0.2    | LogReg       | reweight:SelfConfidence       |  20 |      0.3484 |        0.2749 |     15 |        4 | 0.0298 |   1      |
| label@0.2    | RandomForest | remove:DataOOB@0.1            |  20 |      0.8956 |        0.8932 |     16 |        2 | 0.005  |   1      |
| label@0.2    | RandomForest | remove:SelfConfidence@0.1     |  20 |      0.8131 |        0.5601 |     14 |        6 | 0.0362 |   1      |
| label@0.2    | RandomForest | remove:SelfConfidence@0.05    |  20 |      0.4569 |        0.4686 |     12 |        7 | 0.0641 |   1      |

## D2. Best option of each family per condition (mean gain over the 4 models)

Note: picking the best option post hoc is optimistic; use it to see trends only.

| family                |   clean |   feature@0.05 |   feature@0.1 |   label@0.05 |   label@0.1 |   label@0.2 |
|:----------------------|--------:|---------------:|--------------:|-------------:|------------:|------------:|
| random-removal        |   -0.01 |          -0.01 |          0.04 |         0.05 |        0.01 |        0.06 |
| remove-classical      |    0    |           0.67 |          1.27 |        -0.02 |        0.04 |        0.08 |
| remove-target-aware   |   -0.01 |           0.01 |          0.05 |         0.38 |        0.88 |        1.45 |
| reweight-classical    |    0.03 |           0.48 |          0.73 |         0.04 |        0.08 |        0.08 |
| reweight-target-aware |    0.03 |           0.23 |          0.21 |         0.36 |        0.52 |        0.42 |
| winsorize             |    0.02 |           0.49 |          0.43 |        -0.02 |        0.07 |        0.04 |

## E. Headroom for a per-dataset recommender (leave-one-seed-out)

A recommender is justified only where `honest` is clearly > 0 (small p) AND clearly above `best_fixed_gain` (one rule for all datasets).

| condition    | model        |   naive |   honest |   pct_honest_gt_1pt |   p_honest_gt_0 |   noise_sd |   best_fixed_gain |
|:-------------|:-------------|--------:|---------:|--------------------:|----------------:|-----------:|------------------:|
| clean        | HistGB       |   1.056 |    0.325 |                  20 |           0.199 |      1.561 |             0.188 |
| clean        | KNN          |   0.825 |    0.102 |                  25 |           0.378 |      1.585 |             0.047 |
| clean        | LogReg       |   0.828 |    0.2   |                   5 |           0.106 |      1.217 |             0.052 |
| clean        | RandomForest |   0.755 |    0.13  |                   5 |           0.174 |      1.163 |             0.134 |
| feature@0.05 | HistGB       |   1.178 |    0.294 |                  20 |           0.205 |      1.715 |             0.251 |
| feature@0.05 | KNN          |   1.006 |    0.281 |                  25 |           0.115 |      1.599 |             0.116 |
| feature@0.05 | LogReg       |   3.078 |    2.359 |                  60 |           0     |      1.907 |             2.518 |
| feature@0.05 | RandomForest |   0.874 |   -0.166 |                   5 |           0.978 |      1.415 |             0.213 |
| feature@0.1  | HistGB       |   1.332 |    0.334 |                  20 |           0.522 |      1.479 |             0.377 |
| feature@0.1  | KNN          |   1.033 |    0.284 |                  25 |           0.147 |      1.578 |             0.193 |
| feature@0.1  | LogReg       |   4.728 |    4.331 |                  95 |           0     |      1.604 |             4.393 |
| feature@0.1  | RandomForest |   1.041 |    0.317 |                  15 |           0.018 |      1.467 |             0.451 |
| label@0.05   | HistGB       |   1.599 |    0.544 |                  25 |           0.01  |      1.56  |             0.566 |
| label@0.05   | KNN          |   1.059 |    0.517 |                  35 |           0.02  |      1.691 |             0.266 |
| label@0.05   | LogReg       |   1.932 |    1.501 |                  50 |           0.001 |      1.201 |             1.172 |
| label@0.05   | RandomForest |   0.984 |    0.057 |                  15 |           0.702 |      1.194 |             0.04  |
| label@0.1    | HistGB       |   1.969 |    1.246 |                  60 |           0.001 |      1.84  |             1.062 |
| label@0.1    | KNN          |   1.639 |    1.043 |                  60 |           0.001 |      1.532 |             0.863 |
| label@0.1    | LogReg       |   2.209 |    1.667 |                  50 |           0.001 |      1.662 |             1.632 |
| label@0.1    | RandomForest |   1.21  |    0.674 |                  35 |           0.024 |      1.379 |             0.131 |
| label@0.2    | HistGB       |   2.872 |    2.097 |                  65 |           0     |      2.352 |             2.103 |
| label@0.2    | KNN          |   2.581 |    2.277 |                  80 |           0     |      2.099 |             2.295 |
| label@0.2    | LogReg       |   1.511 |    0.673 |                  35 |           0.018 |      1.428 |             0.494 |
| label@0.2    | RandomForest |   1.74  |    0.664 |                  40 |           0.057 |      1.827 |             0.896 |

## Notes
failures.log: 8 lines (8 removals skipped because they deleted an entire class).
