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
| oracle         |  nan    |                 2.3  |                3.24 |               1.43 |             3.3  |            2.07 |            4.58 |        5.03 |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        1.11 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        2.22 |
| rs:IF          |  nan    |                 2.28 |                2.97 |               1.19 |           nan    |          nan    |          nan    |        2.12 |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |            -2.8  |            1.38 |            2.75 |        1.34 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        3.86 |
| fix:IF@0.1     |   -0.66 |                 1.92 |                2.97 |               1.19 |            -0.93 |           -1.07 |           -1.6  |        2.12 |
| fix:OOB@0.1    |   -1.22 |                 0.31 |                0.03 |              -0.56 |            -2.37 |            1.38 |            1.29 |        1.34 |
| random         |  nan    |                -0.1  |               -0.32 |              -0.34 |            -0.76 |           -0.25 |           -0.69 |       -0.46 |

### KNN (default)

| option         |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:---------------|--------:|---------------------:|--------------------:|-------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle         |  nan    |                -0.05 |               -0.14 |               0.06 |             2.83 |            3.27 |            5.97 |        3.6  |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        3.59 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        0.09 |
| rs:IF          |  nan    |                -0.12 |               -0.22 |              -0.13 |           nan    |          nan    |          nan    |        0    |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |            -4.2  |            2.51 |            3.79 |        2.78 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        2.77 |
| fix:IF@0.1     |   -1.16 |                -0.5  |               -0.22 |              -0.13 |            -0.89 |           -1.42 |           -1.28 |        0    |
| fix:OOB@0.1    |   -2.08 |                -1.45 |               -1.66 |              -1.48 |            -3.8  |            2.51 |            2.46 |        2.78 |
| random         |  nan    |                -0.07 |               -0.28 |              -0.15 |            -0.55 |           -0.35 |           -0.93 |       -0.85 |

### RandomForest (default)

| option         |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:---------------|--------:|---------------------:|--------------------:|-------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle         |  nan    |                -0.24 |               -0.09 |              -0.03 |             2.43 |            2.29 |            5.16 |        2.56 |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        2.89 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |       -0.15 |
| rs:IF          |  nan    |                -0.24 |               -0.12 |              -0.05 |           nan    |          nan    |          nan    |       -0.28 |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |            -4.48 |            0.74 |            2.49 |        1.34 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        1.04 |
| fix:IF@0.1     |   -0.68 |                -0.54 |               -0.12 |              -0.05 |            -0.75 |           -1.04 |           -1.3  |       -0.28 |
| fix:OOB@0.1    |   -2.79 |                -3.21 |               -2.7  |              -2.95 |            -5.35 |            0.74 |            1.61 |        1.34 |
| random         |  nan    |                -0.5  |                0.01 |              -0.2  |            -0.82 |           -0.21 |           -0.54 |       -0.33 |

### HistGB (default)

| option         |   clean |   feature_gross@0.05 |   feature_gross@0.1 |   feature_mild@0.1 |   label_asym@0.2 |   label_sym@0.1 |   label_sym@0.2 |   mixed@0.1 |
|:---------------|--------:|---------------------:|--------------------:|-------------------:|-----------------:|----------------:|----------------:|------------:|
| oracle         |  nan    |                -0.01 |                0.51 |               0.06 |             6.73 |            2.56 |            6.27 |        3.07 |
| oracle_label   |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        2.89 |
| oracle_feature |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |       -0.08 |
| rs:IF          |  nan    |                -0.18 |                0.45 |               0.18 |           nan    |          nan    |          nan    |       -0.03 |
| rs:OOB         |  nan    |               nan    |              nan    |             nan    |            -4.2  |            0.51 |            2.87 |        1.03 |
| rs:both        |  nan    |               nan    |              nan    |             nan    |           nan    |          nan    |          nan    |        1.01 |
| fix:IF@0.1     |   -0.92 |                -0.48 |                0.45 |               0.18 |            -0.55 |           -1.18 |           -1.13 |       -0.03 |
| fix:OOB@0.1    |   -3.78 |                -3.67 |               -3.4  |              -3.36 |            -4.2  |            0.51 |            2.28 |        1.03 |
| random         |  nan    |                -0.12 |               -0.05 |              -0.18 |            -0.61 |           -0.51 |           -0.69 |       -0.5  |

## C. Pre-registered hypotheses

### H12a: even oracle removal of feature errors barely helps tree ensembles

| family       | condition          |   oracle_gain |   p_gt0 |
|:-------------|:-------------------|--------------:|--------:|
| RandomForest | feature_gross@0.05 |       -0.2428 |  0.7868 |
| RandomForest | feature_gross@0.1  |       -0.0945 |  0.6624 |
| RandomForest | feature_mild@0.1   |       -0.0328 |  0.5029 |
| HistGB       | feature_gross@0.05 |       -0.0089 |  0.4611 |
| HistGB       | feature_gross@0.1  |        0.5081 |  0.1107 |
| HistGB       | feature_mild@0.1   |        0.0613 |  0.5992 |

Criterion: every oracle_gain < 0.5 -> **NOT supported**.

### H12b: where does logistic regression lose, detection or removal?

| condition          | detector   |   oracle |   right_sized |   gap |
|:-------------------|:-----------|---------:|--------------:|------:|
| feature_gross@0.05 | rs:IF      |    2.299 |         2.283 | 0.016 |
| feature_gross@0.1  | rs:IF      |    3.241 |         2.965 | 0.276 |
| feature_mild@0.1   | rs:IF      |    1.427 |         1.186 | 0.241 |
| label_asym@0.2     | rs:OOB     |    3.303 |        -2.796 | 6.099 |
| label_sym@0.1      | rs:OOB     |    2.066 |         1.381 | 0.684 |
| label_sym@0.2      | rs:OOB     |    4.582 |         2.748 | 1.834 |

Criterion: gap < 0.5 for feature conditions and > 1 for label_asym -> **SUPPORTED**.

### H12c and H12d (one-sided Wilcoxon, Holm over this table)

| hypothesis                              | condition          | comparison                                |   n |   mean_diff |        p |   p_holm |
|:----------------------------------------|:-------------------|:------------------------------------------|----:|------------:|---------:|---------:|
| H12c right-sized >= fixed 10%           | feature_gross@0.05 | rs:IF - fix:IF@0.1 (mean of 4 families)   |  40 |      0.3387 |   0.0015 |   0.0031 |
| H12c right-sized >= fixed 10%           | feature_gross@0.1  | rs:IF - fix:IF@0.1 (mean of 4 families)   |  40 |      0      | nan      | nan      |
| H12c right-sized >= fixed 10%           | feature_mild@0.1   | rs:IF - fix:IF@0.1 (mean of 4 families)   |  40 |      0      | nan      | nan      |
| H12c right-sized >= fixed 10%           | label_asym@0.2     | rs:OOB - fix:OOB@0.1 (mean of 4 families) |  40 |      0.3175 |   0.6526 |   0.6526 |
| H12c right-sized >= fixed 10%           | label_sym@0.1      | rs:OOB - fix:OOB@0.1 (mean of 4 families) |  40 |      0      | nan      | nan      |
| H12c right-sized >= fixed 10%           | label_sym@0.2      | rs:OOB - fix:OOB@0.1 (mean of 4 families) |  40 |      1.0615 |   0      |   0      |
| H12d oracle helps under 20% label noise | label_sym@0.2      | LogReg: oracle - none                     |  40 |      4.5823 |   0      |   0      |
| H12d oracle helps under 20% label noise | label_sym@0.2      | KNN: oracle - none                        |  40 |      5.9659 |   0      |   0      |
| H12d oracle helps under 20% label noise | label_sym@0.2      | RandomForest: oracle - none               |  40 |      5.1631 |   0      |   0      |
| H12d oracle helps under 20% label noise | label_sym@0.2      | HistGB: oracle - none                     |  40 |      6.2746 |   0      |   0      |

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
