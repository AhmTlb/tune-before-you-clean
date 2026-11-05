# Revision reanalysis (post hoc, after the pre-submission review)

Gains in points (x100). 95% CI = percentile bootstrap over datasets (5000 resamples). p = two-sided Wilcoxon signed-rank over datasets unless stated.

## A. Stage 3, clean data, remove top 10%: against no treatment and against random removal

Random removal of the same number of rows separates the cost of *which* rows are removed from the cost of having fewer training rows.

| scorer          | family       |   n | vs_none              |   p_vs_none | vs_random            |   p_vs_random | random_vs_none       |   pHolm_vs_none |   pHolm_vs_random |
|:----------------|:-------------|----:|:---------------------|------------:|:---------------------|--------------:|:---------------------|----------------:|------------------:|
| IsolationForest | LogReg       |  40 | -0.43 [-1.14, +0.22] |      0.0662 | -0.20 [-0.90, +0.44] |        0.2826 | -0.23 [-0.41, -0.07] |          0.0662 |            0.2826 |
| IsolationForest | KNN          |  40 | -1.16 [-1.77, -0.64] |      0      | -0.73 [-1.31, -0.21] |        0.0147 | -0.43 [-0.71, -0.19] |          0.0002 |            0.0587 |
| IsolationForest | RandomForest |  40 | -1.03 [-1.82, -0.38] |      0.0014 | -0.71 [-1.56, -0.06] |        0.0051 | -0.32 [-0.56, -0.07] |          0.0047 |            0.0357 |
| IsolationForest | HistGB       |  40 | -1.00 [-1.63, -0.40] |      0.0012 | -0.61 [-1.20, -0.09] |        0.0368 | -0.39 [-0.68, -0.11] |          0.0047 |            0.1105 |
| DataOOB         | LogReg       |  40 | -0.86 [-1.83, -0.21] |      0.0158 | -0.61 [-1.52, -0.00] |        0.0408 | -0.24 [-0.42, -0.08] |          0.0315 |            0.1105 |
| DataOOB         | KNN          |  40 | -1.40 [-2.13, -0.74] |      0.0005 | -0.98 [-1.69, -0.38] |        0.0053 | -0.43 [-0.71, -0.18] |          0.0029 |            0.0357 |
| DataOOB         | RandomForest |  40 | -2.08 [-2.85, -1.42] |      0      | -1.77 [-2.44, -1.16] |        0      | -0.31 [-0.56, -0.07] |          0      |            0      |
| DataOOB         | HistGB       |  40 | -3.10 [-4.22, -2.08] |      0      | -2.73 [-3.87, -1.69] |        0      | -0.37 [-0.67, -0.08] |          0      |            0      |
| KNNShapley      | LogReg       |  38 | -2.08 [-3.04, -1.31] |      0      | -1.82 [-2.65, -1.11] |        0      | -0.25 [-0.48, -0.07] |          0      |            0      |
| KNNShapley      | KNN          |  38 | -2.61 [-3.57, -1.70] |      0      | -2.18 [-3.07, -1.36] |        0.0002 | -0.43 [-0.70, -0.19] |          0      |            0.0015 |
| KNNShapley      | RandomForest |  38 | -2.89 [-3.83, -1.98] |      0      | -2.58 [-3.45, -1.78] |        0      | -0.31 [-0.56, -0.06] |          0      |            0      |
| KNNShapley      | HistGB       |  38 | -3.08 [-4.22, -2.04] |      0      | -2.91 [-4.53, -1.61] |        0      | -0.17 [-0.67, +0.54] |          0      |            0.0002 |
| SelfConfidence  | LogReg       |  40 | -1.71 [-3.18, -0.56] |      0.0009 | -1.50 [-2.93, -0.39] |        0.0079 | -0.21 [-0.38, -0.06] |          0.0047 |            0.0396 |
| SelfConfidence  | KNN          |  40 | -1.84 [-2.88, -0.96] |      0      | -1.44 [-2.51, -0.60] |        0.001  | -0.39 [-0.66, -0.14] |          0.0001 |            0.0079 |
| SelfConfidence  | RandomForest |  40 | -3.36 [-5.29, -2.01] |      0      | -3.04 [-4.90, -1.75] |        0      | -0.33 [-0.57, -0.09] |          0      |            0      |
| SelfConfidence  | HistGB       |  40 | -3.78 [-5.80, -2.20] |      0      | -3.33 [-5.36, -1.74] |        0      | -0.45 [-0.75, -0.17] |          0      |            0      |

Holm over the 16 rows of each column. Rows where a removal would have deleted a class were skipped in Stage 3, so pairs are inner-joined.

## B. Does the effect of removal depend on the model family? (interaction)

Friedman test over the four families on per-dataset gains (H0: the gain is the same for every family). Paired differences: gain(LogReg) - gain(family).

| comparison                                          |   n_datasets |   friedman_chi2 |   friedman_p | LogReg-KNN           |   p(LogReg-KNN) | LogReg-RandomForest   |   p(LogReg-RandomForest) | LogReg-HistGB        |   p(LogReg-HistGB) |
|:----------------------------------------------------|-------------:|----------------:|-------------:|:---------------------|----------------:|:----------------------|-------------------------:|:---------------------|-------------------:|
| clean, Isolation Forest 10%                         |           40 |            1.65 |       0.6481 | +0.72 [+0.11, +1.42] |          0.0548 | +0.59 [+0.02, +1.25]  |                   0.1659 | +0.56 [-0.10, +1.31] |             0.3011 |
| clean, Data-OOB 10%                                 |           40 |           39.51 |       0      | +0.55 [-0.11, +1.13] |          0.0466 | +1.22 [+0.19, +2.14]  |                   0.0003 | +2.25 [+1.31, +3.23] |             0      |
| corrupted, matched scorer 10%, mean over conditions |           40 |           37.26 |       0      | +2.07 [+1.33, +2.99] |          0      | +2.10 [+1.38, +2.97]  |                   0      | +1.98 [+1.24, +2.92] |             0      |

## C. Equivalence: is matched removal under corruption within +-0.5 points of no treatment?

TOST with two one-sided Wilcoxon tests (Stage 3, matched scorer at 10%, per condition). p_TOST < 0.05 means the gain is shown to lie within +-0.5 points; a large p_TOST means equivalence is NOT shown (the effect may be larger, or the data too noisy).

| condition          | family       | gain                 |   p_TOST |
|:-------------------|:-------------|:---------------------|---------:|
| feature_gross@0.05 | LogReg       | +3.13 [+1.94, +4.47] |   0.9999 |
| feature_gross@0.05 | KNN          | -0.55 [-0.87, -0.25] |   0.4656 |
| feature_gross@0.05 | RandomForest | -0.25 [-0.54, +0.03] |   0.0112 |
| feature_gross@0.05 | HistGB       | -0.50 [-0.88, -0.14] |   0.4288 |
| feature_gross@0.1  | LogReg       | +5.21 [+3.70, +6.93] |   1      |
| feature_gross@0.1  | KNN          | +0.07 [-0.15, +0.30] |   0.0002 |
| feature_gross@0.1  | RandomForest | +0.10 [-0.10, +0.30] |   0.0001 |
| feature_gross@0.1  | HistGB       | +0.04 [-0.24, +0.35] |   0.0001 |
| feature_mild@0.05  | LogReg       | +1.16 [+0.44, +1.93] |   0.9171 |
| feature_mild@0.05  | KNN          | -0.57 [-0.84, -0.32] |   0.5502 |
| feature_mild@0.05  | RandomForest | -0.16 [-0.37, +0.06] |   0.0009 |
| feature_mild@0.05  | HistGB       | -0.53 [-0.95, -0.15] |   0.455  |
| feature_mild@0.1   | LogReg       | +1.76 [+1.05, +2.53] |   0.9992 |
| feature_mild@0.1   | KNN          | -0.34 [-0.63, -0.08] |   0.0265 |
| feature_mild@0.1   | RandomForest | +0.10 [-0.32, +0.45] |   0.0018 |
| feature_mild@0.1   | HistGB       | -0.18 [-0.64, +0.25] |   0.0055 |
| label_asym@0.1     | LogReg       | +0.83 [+0.00, +1.69] |   0.694  |
| label_asym@0.1     | KNN          | +0.07 [-0.58, +0.73] |   0.1243 |
| label_asym@0.1     | RandomForest | -0.21 [-0.67, +0.25] |   0.0859 |
| label_asym@0.1     | HistGB       | +0.14 [-0.73, +0.95] |   0.1889 |
| label_asym@0.2     | LogReg       | +0.11 [-0.68, +0.80] |   0.2956 |
| label_asym@0.2     | KNN          | -0.20 [-1.03, +0.64] |   0.2367 |
| label_asym@0.2     | RandomForest | -0.20 [-0.78, +0.38] |   0.1971 |
| label_asym@0.2     | HistGB       | -0.48 [-1.55, +0.57] |   0.4057 |
| label_sym@0.1      | LogReg       | +2.19 [+1.28, +3.28] |   0.9999 |
| label_sym@0.1      | KNN          | +0.28 [-0.17, +0.72] |   0.2423 |
| label_sym@0.1      | RandomForest | -0.07 [-0.49, +0.35] |   0.0144 |
| label_sym@0.1      | HistGB       | +0.53 [+0.07, +0.95] |   0.6768 |
| label_sym@0.2      | LogReg       | +1.34 [+0.75, +2.01] |   0.9834 |
| label_sym@0.2      | KNN          | +1.42 [+0.94, +1.91] |   0.9997 |
| label_sym@0.2      | RandomForest | +0.49 [+0.12, +0.87] |   0.3673 |
| label_sym@0.2      | HistGB       | +1.71 [+1.12, +2.33] |   0.9999 |

## D. Stage 4: how much of the cleaning gain does a tuned, quantile-transformed LogReg recover?

Per dirty condition: cleaning gain = default LogReg + matched removal - default LogReg (none); robust gain = quantile+tuned LogReg (none) - default LogReg (none). Reported: both gains and their difference (robust - cleaning) with CI, pooled over the corrupted conditions except mixed (whose matched scorer is ambiguous) (dataset x condition means, then averaged per dataset). A ratio of mean gains is shown only for continuity.

| metric   |   n | cleaning_gain        | robust_gain          | robust_minus_cleaning   |      p |   ratio_of_means |
|:---------|----:|:---------------------|:---------------------|:------------------------|-------:|-----------------:|
| BA       |  40 | +2.25 [+1.52, +3.18] | +1.46 [+0.46, +2.69] | -0.79 [-1.82, +0.27]    | 0.0622 |             0.65 |
| BAP      |  40 | +0.93 [+0.28, +1.61] | +0.90 [+0.09, +1.83] | -0.02 [-1.01, +0.97]    | 0.9841 |             0.97 |
| AUC      |  40 | +0.31 [-0.19, +0.78] | +0.37 [-0.14, +0.92] | +0.07 [-0.63, +0.73]    | 0.9417 |             1.22 |

## E. Stage 5: asymmetric 20% label noise - oracle, right-sized Data-OOB and the gap

Gain over no treatment (default models). Gap = oracle - right-sized Data-OOB. Realised flip share = rows removed by the oracle / training rows.

| metric   | task       | family       |   n | oracle                 | rs_DataOOB             | gap                     | random               |
|:---------|:-----------|:-------------|----:|:-----------------------|:-----------------------|:------------------------|:---------------------|
| BA       | binary     | LogReg       |  22 | +11.52 [+7.47, +15.53] | -0.81 [-1.96, +0.34]   | +12.33 [+8.39, +16.24]  | -0.05 [-0.40, +0.27] |
| BA       | binary     | KNN          |  22 | +8.76 [+5.60, +11.93]  | -2.82 [-3.97, -1.71]   | +11.58 [+8.01, +15.20]  | -0.61 [-1.21, -0.10] |
| BA       | binary     | RandomForest |  22 | +12.88 [+8.88, +16.89] | -2.12 [-2.92, -1.30]   | +15.00 [+10.80, +19.60] | -0.08 [-0.35, +0.20] |
| BA       | binary     | HistGB       |  22 | +11.82 [+8.32, +15.19] | -3.99 [-5.50, -2.48]   | +15.82 [+11.92, +19.68] | -0.78 [-1.21, -0.37] |
| BA       | multiclass | LogReg       |  18 | +5.00 [+1.18, +8.25]   | +1.76 [-0.86, +4.07]   | +3.24 [+1.60, +4.66]    | -1.12 [-1.67, -0.59] |
| BA       | multiclass | KNN          |  18 | +4.69 [+3.58, +5.72]   | +2.50 [+1.22, +3.66]   | +2.19 [+1.14, +3.41]    | -0.86 [-1.90, -0.05] |
| BA       | multiclass | RandomForest |  18 | +4.68 [+3.37, +6.19]   | +1.38 [+0.57, +2.31]   | +3.30 [+1.86, +4.99]    | -0.64 [-1.34, +0.07] |
| BA       | multiclass | HistGB       |  18 | +7.25 [+5.59, +8.92]   | +2.84 [+1.84, +3.85]   | +4.41 [+2.94, +6.10]    | -0.98 [-1.67, -0.31] |
| BA       | all        | LogReg       |  40 | +8.58 [+5.66, +11.55]  | +0.34 [-1.04, +1.70]   | +8.24 [+5.61, +10.92]   | -0.53 [-0.88, -0.21] |
| BA       | all        | KNN          |  40 | +6.93 [+5.03, +8.99]   | -0.42 [-1.58, +0.73]   | +7.35 [+5.02, +9.99]    | -0.72 [-1.26, -0.27] |
| BA       | all        | RandomForest |  40 | +9.19 [+6.72, +11.94]  | -0.55 [-1.34, +0.25]   | +9.74 [+6.75, +12.95]   | -0.33 [-0.71, +0.02] |
| BA       | all        | HistGB       |  40 | +9.77 [+7.70, +11.97]  | -0.92 [-2.36, +0.51]   | +10.68 [+8.02, +13.62]  | -0.87 [-1.25, -0.49] |
| BAP      | binary     | LogReg       |  22 | +2.13 [+1.49, +2.77]   | -6.32 [-8.12, -4.63]   | +8.45 [+6.83, +10.15]   | -0.47 [-0.86, -0.08] |
| BAP      | binary     | KNN          |  22 | +2.23 [+1.38, +3.11]   | -9.23 [-11.59, -6.82]  | +11.46 [+9.40, +13.62]  | -0.66 [-1.25, -0.14] |
| BAP      | binary     | RandomForest |  22 | +2.00 [+1.42, +2.63]   | -8.36 [-10.95, -5.92]  | +10.36 [+7.98, +12.84]  | -0.73 [-1.09, -0.41] |
| BAP      | binary     | HistGB       |  22 | +6.60 [+5.04, +8.10]   | -9.27 [-11.48, -7.18]  | +15.87 [+13.11, +18.79] | -0.32 [-0.98, +0.44] |
| BAP      | multiclass | LogReg       |  18 | +4.73 [+2.93, +6.69]   | +1.35 [-0.34, +3.11]   | +3.38 [+2.47, +4.39]    | -1.13 [-1.66, -0.66] |
| BAP      | multiclass | KNN          |  18 | +3.56 [+2.52, +4.47]   | +1.95 [+0.51, +3.33]   | +1.61 [+0.09, +2.85]    | -0.42 [-1.82, +1.01] |
| BAP      | multiclass | RandomForest |  18 | +2.96 [+1.87, +4.20]   | +0.27 [-0.54, +1.15]   | +2.68 [+1.35, +4.27]    | -0.93 [-1.47, -0.33] |
| BAP      | multiclass | HistGB       |  18 | +6.90 [+5.45, +8.34]   | +2.01 [+1.23, +2.81]   | +4.89 [+3.41, +6.49]    | -0.97 [-1.66, -0.29] |
| BAP      | all        | LogReg       |  40 | +3.30 [+2.35, +4.35]   | -2.87 [-4.56, -1.14]   | +6.17 [+4.92, +7.51]    | -0.76 [-1.11, -0.44] |
| BAP      | all        | KNN          |  40 | +2.83 [+2.11, +3.49]   | -4.20 [-6.51, -1.97]   | +7.03 [+5.05, +9.11]    | -0.55 [-1.25, +0.17] |
| BAP      | all        | RandomForest |  40 | +2.43 [+1.82, +3.08]   | -4.48 [-6.45, -2.61]   | +6.91 [+5.09, +8.87]    | -0.82 [-1.14, -0.49] |
| BAP      | all        | HistGB       |  40 | +6.73 [+5.66, +7.77]   | -4.20 [-6.48, -2.09]   | +10.93 [+8.59, +13.45]  | -0.61 [-1.10, -0.08] |
| AUC      | binary     | LogReg       |  22 | +1.33 [+0.82, +1.85]   | -5.64 [-9.46, -2.77]   | +6.97 [+4.32, +10.42]   | -0.40 [-0.54, -0.24] |
| AUC      | binary     | KNN          |  22 | +3.20 [+2.51, +3.94]   | -11.09 [-13.27, -8.91] | +14.29 [+12.14, +16.45] | -0.60 [-1.15, -0.14] |
| AUC      | binary     | RandomForest |  22 | +1.82 [+1.33, +2.32]   | -9.63 [-13.71, -6.22]  | +11.45 [+8.01, +15.14]  | -0.58 [-0.87, -0.34] |
| AUC      | binary     | HistGB       |  22 | +2.21 [+1.59, +2.83]   | -6.38 [-10.46, -2.70]  | +8.59 [+5.21, +12.36]   | -0.30 [-0.70, +0.10] |
| AUC      | multiclass | LogReg       |  18 | +1.39 [+0.98, +1.84]   | +0.28 [-0.18, +0.80]   | +1.10 [+0.82, +1.40]    | -0.40 [-0.57, -0.24] |
| AUC      | multiclass | KNN          |  18 | +1.69 [+1.25, +2.16]   | +0.23 [-0.40, +0.87]   | +1.46 [+0.96, +2.01]    | -0.54 [-0.83, -0.26] |
| AUC      | multiclass | RandomForest |  18 | +1.31 [+0.88, +1.78]   | +0.26 [-0.17, +0.73]   | +1.04 [+0.55, +1.53]    | -0.30 [-0.50, -0.13] |
| AUC      | multiclass | HistGB       |  18 | +2.14 [+1.35, +3.03]   | +0.33 [-0.16, +0.81]   | +1.82 [+0.95, +2.91]    | -0.34 [-0.59, -0.11] |
| AUC      | all        | LogReg       |  40 | +1.36 [+1.01, +1.73]   | -2.97 [-5.28, -1.19]   | +4.33 [+2.66, +6.46]    | -0.40 [-0.51, -0.28] |
| AUC      | all        | KNN          |  40 | +2.52 [+2.02, +3.03]   | -6.00 [-8.15, -3.93]   | +8.52 [+6.29, +10.86]   | -0.57 [-0.91, -0.28] |
| AUC      | all        | RandomForest |  40 | +1.59 [+1.24, +1.94]   | -5.18 [-7.94, -2.84]   | +6.77 [+4.41, +9.44]    | -0.46 [-0.64, -0.30] |
| AUC      | all        | HistGB       |  40 | +2.18 [+1.68, +2.71]   | -3.36 [-5.95, -1.19]   | +5.54 [+3.37, +8.14]    | -0.32 [-0.56, -0.08] |

Realised flip share at nominal 20%: binary: mean 0.130 (range 0.031-0.201, 22 datasets); multiclass: mean 0.200 (range 0.199-0.201, 18 datasets)
