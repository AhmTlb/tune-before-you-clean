# Experiment v5: oracle vs. right-sized vs. fixed removal

Datasets: 40 | seeds: 3 | variants: default

## A. Detector precision and recall

precision = share of removed rows that were corrupted (any type); precision_own_type = share corrupted in the detector's own type; recall = share of corrupted rows removed.

|                                       |   precision |   precision_own_type |   recall |
|:--------------------------------------|------------:|---------------------:|---------:|
| ('feature_gross@0.05', 'fix:IF@0.1')  |       0.5   |                0.5   |    0.998 |
| ('feature_gross@0.05', 'fix:OOB@0.1') |       0.129 |                0     |    0.256 |
| ('feature_gross@0.05', 'rs:IF')       |       0.857 |                0.857 |    0.857 |
| ('feature_gross@0.1', 'fix:IF@0.1')   |       0.95  |                0.95  |    0.95  |
| ('feature_gross@0.1', 'fix:OOB@0.1')  |       0.223 |                0     |    0.223 |
| ('feature_gross@0.1', 'rs:IF')        |       0.95  |                0.95  |    0.95  |
| ('feature_mild@0.1', 'fix:IF@0.1')    |       0.785 |                0.785 |    0.785 |
| ('feature_mild@0.1', 'fix:OOB@0.1')   |       0.209 |                0     |    0.209 |
| ('feature_mild@0.1', 'rs:IF')         |       0.785 |                0.785 |    0.785 |
| ('label_asym@0.2', 'fix:IF@0.1')      |       0.202 |                0     |    0.134 |
| ('label_asym@0.2', 'fix:OOB@0.1')     |       0.368 |                0.368 |    0.233 |
| ('label_asym@0.2', 'rs:OOB')          |       0.349 |                0.349 |    0.349 |
| ('label_sym@0.1', 'fix:IF@0.1')       |       0.099 |                0     |    0.099 |
| ('label_sym@0.1', 'fix:OOB@0.1')      |       0.664 |                0.664 |    0.664 |
| ('label_sym@0.1', 'rs:OOB')           |       0.664 |                0.664 |    0.664 |
| ('label_sym@0.2', 'fix:IF@0.1')       |       0.195 |                0     |    0.098 |
| ('label_sym@0.2', 'fix:OOB@0.1')      |       0.818 |                0.818 |    0.409 |
| ('label_sym@0.2', 'rs:OOB')           |       0.704 |                0.704 |    0.704 |
| ('mixed@0.1', 'fix:IF@0.1')           |       0.957 |                0.951 |    0.479 |
| ('mixed@0.1', 'fix:OOB@0.1')          |       0.703 |                0.651 |    0.351 |
| ('mixed@0.1', 'rs:IF')                |       0.957 |                0.951 |    0.479 |
| ('mixed@0.1', 'rs:OOB')               |       0.703 |                0.651 |    0.351 |
| ('mixed@0.1', 'rs:both')              |       0.827 |                0.827 |    0.803 |

## B. Gains vs. no treatment

### LogReg (default)

| option         |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:---------------|--------:|---------------------:|--------------------:|-------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle         |  nan    |                 3.32 |                5.41 |               2.26 |             8.58 |            2.78 |            4.5  |        7.25 |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        1.58 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        4.09 |
| rs:IF          |  nan    |                 3.13 |                5.17 |               1.79 |           nan    |          nan    |          nan    |        4.02 |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |             0.39 |            2.12 |            2.8  |        1.89 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        6.09 |
| fix:IF@0.1     |   -0.52 |                 3.17 |                5.17 |               1.79 |             0.79 |           -0.5  |           -0.71 |        4.02 |
| fix:OOB@0.1    |   -0.8  |                 1.12 |                1.22 |               0.26 |             0.26 |            2.12 |            1.33 |        1.89 |
| random         |  nan    |                -0.03 |               -0.14 |              -0.17 |            -0.53 |           -0.16 |           -0.38 |       -0.27 |

### KNN (default)

| option         |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:---------------|--------:|---------------------:|--------------------:|-------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle         |  nan    |                 0.02 |               -0.07 |              -0.17 |             6.93 |            0.87 |            2.84 |        0.47 |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        0.53 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |       -0.2  |
| rs:IF          |  nan    |                -0.08 |                0.02 |              -0.42 |           nan    |          nan    |          nan    |       -0.14 |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |            -0.42 |            0.2  |            1.62 |        0.23 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        0.2  |
| fix:IF@0.1     |   -1.29 |                -0.51 |                0.02 |              -0.42 |            -0.76 |           -1.31 |           -0.95 |       -0.14 |
| fix:OOB@0.1    |   -1.33 |                -1.04 |               -0.93 |              -1.07 |            -0.41 |            0.2  |            1.32 |        0.23 |
| random         |  nan    |                -0.15 |               -0.22 |              -0.41 |            -0.72 |           -0.32 |           -0.65 |       -0.98 |

### RandomForest (default)

| option         |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:---------------|--------:|---------------------:|--------------------:|-------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle         |  nan    |                 0.04 |                0.08 |               0.12 |             9.19 |            0.63 |            1.88 |        1.04 |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        0.95 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |       -0.02 |
| rs:IF          |  nan    |                 0.03 |                0.09 |               0.25 |           nan    |          nan    |          nan    |       -0.12 |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |            -0.55 |           -0.3  |            0.49 |       -0.11 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |       -0.33 |
| fix:IF@0.1     |   -1.12 |                -0.34 |                0.09 |               0.25 |            -0.09 |           -1.3  |           -1.27 |       -0.12 |
| fix:OOB@0.1    |   -2.15 |                -2.15 |               -2    |              -1.88 |            -0.39 |           -0.3  |            0.44 |       -0.11 |
| random         |  nan    |                -0.24 |               -0.14 |              -0.19 |            -0.33 |           -0.43 |           -1.08 |       -0.52 |

### HistGB (default)

| option         |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:---------------|--------:|---------------------:|--------------------:|-------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle         |  nan    |                 0.17 |                0.16 |               0.38 |             9.77 |            1.73 |            4.87 |        2.09 |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        2.16 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |       -0.45 |
| rs:IF          |  nan    |                 0.07 |                0.11 |              -0.07 |           nan    |          nan    |          nan    |       -0.37 |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |            -0.92 |            0.3  |            2.07 |        0.76 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        0.75 |
| fix:IF@0.1     |   -1.15 |                -0.5  |                0.11 |              -0.07 |            -0.78 |           -1.19 |           -1.14 |       -0.37 |
| fix:OOB@0.1    |   -2.76 |                -2.56 |               -2.6  |              -2.36 |            -1.14 |            0.3  |            1.71 |        0.76 |
| random         |  nan    |                -0.1  |               -0.18 |              -0.08 |            -0.87 |           -0.35 |           -0.54 |       -0.49 |

## C. Pre-registered hypotheses

### H12a: even oracle removal of feature errors barely helps tree ensembles

| family       | condition          |   oracle_gain |   p_gt0 |
|:-------------|:-------------------|--------------:|--------:|
| RandomForest | feature_gross@0.05 |        0.0386 |  0.3232 |
| RandomForest | feature_gross@0.1  |        0.0758 |  0.2091 |
| RandomForest | feature_mild@0.1   |        0.1164 |  0.234  |
| HistGB       | feature_gross@0.05 |        0.1731 |  0.508  |
| HistGB       | feature_gross@0.1  |        0.1576 |  0.2724 |
| HistGB       | feature_mild@0.1   |        0.3766 |  0.0984 |

Criterion: every oracle_gain < 0.5 -> **SUPPORTED**.

### H12b: where does logistic regression lose, detection or removal?

| condition          | detector   |   oracle |   right_sized |   gap |
|:-------------------|:-----------|---------:|--------------:|------:|
| feature_gross@0.05 | rs:IF      |    3.317 |         3.135 | 0.183 |
| feature_gross@0.1  | rs:IF      |    5.412 |         5.172 | 0.24  |
| feature_mild@0.1   | rs:IF      |    2.259 |         1.794 | 0.464 |
| label_asym@0.2     | rs:OOB     |    8.583 |         0.389 | 8.194 |
| label_sym@0.1      | rs:OOB     |    2.781 |         2.116 | 0.665 |
| label_sym@0.2      | rs:OOB     |    4.504 |         2.801 | 1.703 |

Criterion: gap < 0.5 for feature conditions and > 1 for label_asym -> **SUPPORTED**.

### H12c and H12d (one-sided Wilcoxon, Holm over this table)

| hypothesis                              | condition          | comparison                                |   n |   mean_diff |        p |   p_holm |
|:----------------------------------------|:-------------------|:------------------------------------------|----:|------------:|---------:|---------:|
| H12c right-sized >= fixed 10%           | feature_gross@0.05 | rs:IF - fix:IF@0.1 (mean of 4 families)   |  40 |      0.3327 |   0.0047 |   0.0142 |
| H12c right-sized >= fixed 10%           | feature_gross@0.1  | rs:IF - fix:IF@0.1 (mean of 4 families)   |  40 |      0      | nan      | nan      |
| H12c right-sized >= fixed 10%           | feature_mild@0.1   | rs:IF - fix:IF@0.1 (mean of 4 families)   |  40 |      0      | nan      | nan      |
| H12c right-sized >= fixed 10%           | label_asym@0.2     | rs:OOB - fix:OOB@0.1 (mean of 4 families) |  40 |      0.11   |   0.3136 |   0.3136 |
| H12c right-sized >= fixed 10%           | label_sym@0.1      | rs:OOB - fix:OOB@0.1 (mean of 4 families) |  40 |      0      | nan      | nan      |
| H12c right-sized >= fixed 10%           | label_sym@0.2      | rs:OOB - fix:OOB@0.1 (mean of 4 families) |  40 |      0.544  |   0.01   |   0.02   |
| H12d oracle helps under 20% label noise | label_sym@0.2      | LogReg: oracle - none                     |  40 |      4.504  |   0      |   0      |
| H12d oracle helps under 20% label noise | label_sym@0.2      | KNN: oracle - none                        |  40 |      2.845  |   0      |   0      |
| H12d oracle helps under 20% label noise | label_sym@0.2      | RandomForest: oracle - none               |  40 |      1.878  |   0.0006 |   0.0022 |
| H12d oracle helps under 20% label noise | label_sym@0.2      | HistGB: oracle - none                     |  40 |      4.8716 |   0      |   0      |

H12c: right-sized beats fixed removal significantly in **2 of 6** conditions.
H12d: oracle removal helps significantly for **4 of 4** families.

## D. Removals that deleted a whole class (%)

| option         |   % wiped |
|:---------------|----------:|
| fix:IF@0.1     |      0    |
| fix:OOB@0.1    |      1.41 |
| oracle         |      0    |
| oracle_feature |      0    |
| oracle_label   |      0    |
| random         |      0    |
| rs:IF          |      0    |
| rs:OOB         |      0.47 |
| rs:both        |      0    |
