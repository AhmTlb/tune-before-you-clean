# Extra analyses on v3 results

Datasets: 40 | seeds: 5

## 1. Removals that would delete an entire class (skipped in v3)

Total skipped removal settings: **444** across **9** of 40 datasets (200 dataset x seed jobs).

By method (a removal setting = one condition x fraction; each was skipped for all 4 models):

|                                    |   skipped_fits |   datasets |
|:-----------------------------------|---------------:|-----------:|
| ('target-aware', 'KNNShapley')     |            161 |          8 |
| ('target-aware', 'SelfConfidence') |            101 |          7 |
| ('dtt', 'dtt_both')                |             65 |          8 |
| ('target-aware', 'DataOOB')        |             65 |          5 |
| ('dtt', 'dtt_target')              |             52 |          8 |

By method and removal fraction:

| method         |   0.05 |   0.055 |   0.058 |   0.061 |   0.065 |   0.066 |   0.069 |   0.073 |   0.077 |   0.083 |   0.092 |   0.1 |   0.103 |   0.108 |   0.11 |   0.113 |   0.116 |   0.117 |   0.121 |   0.123 |   0.127 |   0.128 |   0.129 |   0.131 |   0.133 |   0.137 |   0.138 |   0.146 |   0.148 |   0.15 |   0.16 |   0.182 |   0.188 |   0.206 |   0.207 |   0.208 |   0.219 |   0.222 |   0.235 |   0.253 |   0.258 |   0.265 |   0.278 |   0.279 |   0.282 |   0.287 |   0.288 |   0.296 |   0.298 |   0.3 |
|:---------------|-------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|------:|--------:|--------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|------:|
| DataOOB        |     13 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |    52 |       0 |       0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |     0 |
| KNNShapley     |     26 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |   135 |       0 |       0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |     0 |
| SelfConfidence |     27 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |    74 |       0 |       0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |     0 |
| dtt_both       |      0 |       0 |       0 |       1 |       0 |       1 |       0 |       0 |       1 |       1 |       0 |     0 |       1 |       0 |      1 |       1 |       1 |       0 |       0 |       0 |       1 |       0 |       0 |       1 |       1 |       0 |       0 |       0 |       0 |      0 |      2 |       1 |       1 |       1 |       1 |       1 |       1 |       1 |       1 |       1 |       1 |       1 |       1 |       3 |       1 |       1 |       1 |       1 |       1 |    32 |
| dtt_target     |      1 |       1 |       1 |       0 |       1 |       0 |       1 |       1 |       0 |       1 |       1 |     2 |       2 |       1 |      0 |       0 |       0 |       1 |       1 |       1 |       0 |       1 |       3 |       0 |       0 |       1 |       1 |       1 |       1 |     28 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |     0 |

By condition:

| condition          |   skipped |
|:-------------------|----------:|
| clean              |        29 |
| feature_gross@0.05 |        24 |
| feature_gross@0.1  |        26 |
| feature_mild@0.05  |        26 |
| feature_mild@0.1   |        25 |
| label_asym@0.1     |       151 |
| label_asym@0.2     |       158 |
| mixed@0.05         |         5 |

## 2. Sensitivity of strategy gains to skipped (class-deleting) removals

If the three policies agree, the skipped cases do not change the conclusion.

|                                                              |   clean |   label |   feature |   mixed |    ALL |
|:-------------------------------------------------------------|--------:|--------:|----------:|--------:|-------:|
| ('always both (DTT sizes)', 'complete cases')                |  -1.553 |   0.088 |    -0.721 |   0.512 | -0.321 |
| ('always both (DTT sizes)', 'optimistic (skip=0)')           |  -1.534 |   0.243 |    -0.701 |   0.539 | -0.208 |
| ('always both (DTT sizes)', 'pessimistic (skip=worst)')      |  -1.566 |   0.048 |    -0.73  |   0.502 | -0.299 |
| ('always classical (IF 5%)', 'complete cases')               |  -0.328 |  -0.24  |     0.457 |   0.527 |  0.145 |
| ('always classical (IF 5%)', 'optimistic (skip=0)')          |  -0.328 |  -0.24  |     0.457 |   0.527 |  0.145 |
| ('always classical (IF 5%)', 'pessimistic (skip=worst)')     |  -0.328 |  -0.24  |     0.457 |   0.527 |  0.145 |
| ('always target-aware (OOB 5%)', 'complete cases')           |  -1.032 |   0.225 |    -0.662 |   0.477 | -0.185 |
| ('always target-aware (OOB 5%)', 'optimistic (skip=0)')      |  -1.032 |   0.222 |    -0.662 |   0.477 | -0.167 |
| ('always target-aware (OOB 5%)', 'pessimistic (skip=worst)') |  -1.032 |   0.188 |    -0.662 |   0.477 | -0.18  |
| ('remove IF 10%', 'complete cases')                          |  -0.904 |  -0.681 |     0.529 |   0.641 | -0.021 |
| ('remove IF 10%', 'optimistic (skip=0)')                     |  -0.904 |  -0.681 |     0.529 |   0.641 | -0.021 |
| ('remove IF 10%', 'pessimistic (skip=worst)')                |  -0.904 |  -0.681 |     0.529 |   0.641 | -0.021 |
| ('remove OOB 10%', 'complete cases')                         |  -1.861 |   0.54  |    -1.255 |   0.322 | -0.399 |
| ('remove OOB 10%', 'optimistic (skip=0)')                    |  -1.809 |   0.52  |    -1.142 |   0.322 | -0.332 |
| ('remove OOB 10%', 'pessimistic (skip=worst)')               |  -1.859 |   0.345 |    -1.258 |   0.322 | -0.443 |
| ('remove SelfConfidence 10%', 'complete cases')              |  -2.67  |   0.295 |    -2     |   0.136 | -0.89  |
| ('remove SelfConfidence 10%', 'optimistic (skip=0)')         |  -2.463 |   0.318 |    -1.821 |   0.136 | -0.746 |
| ('remove SelfConfidence 10%', 'pessimistic (skip=worst)')    |  -2.719 |   0.106 |    -2.017 |   0.136 | -0.917 |

Share of cases skipped (%):

| strategy                     |   clean |   label |   feature |   mixed |   ALL |
|:-----------------------------|--------:|--------:|----------:|--------:|------:|
| always both (DTT sizes)      |     1   |     7.1 |       0.5 |     0.5 |   3   |
| always classical (IF 5%)     |     0   |     0   |       0   |     0   |   0   |
| always target-aware (OOB 5%) |     0   |     1.6 |       0   |     0   |   0.6 |
| remove IF 10%                |     0   |     0   |       0   |     0   |   0   |
| remove OOB 10%               |     0.5 |     5.1 |       1.2 |     0   |   2.4 |
| remove SelfConfidence 10%    |     3.5 |     6   |       2.4 |     0   |   3.4 |

## 3. Noise floor

'Harm' = gain < -1 point. Random removal of the same size is a pure-noise reference: its harm rate is the share of cases that look harmed by chance alone.

| option             |   harm_rate |   mean_gain |   sd_gain |
|:-------------------|------------:|------------:|----------:|
| remove:random@0.02 |      14.761 |      -0.039 |     1.26  |
| remove:random@0.05 |      20.648 |      -0.144 |     1.483 |
| remove:random@0.1  |      27.341 |      -0.304 |     1.756 |

Seed-to-seed SD of the untreated model (points), mean over datasets x conditions:

| model        |   sd_none |
|:-------------|----------:|
| HistGB       |     2.242 |
| KNN          |     2.195 |
| LogReg       |     2.081 |
| RandomForest |     2.149 |

## 4. Break-even probability of contamination (exploratory, post hoc)

Treating is worth it on average when the probability that the data are contaminated exceeds pi*. gain_dirty averages the label, feature and mixed groups equally; it depends on the corruption levels simulated here.

| model        | strategy                |   gain_clean |   gain_dirty | break_even_pi   |
|:-------------|:------------------------|-------------:|-------------:|:----------------|
| HistGB       | remove IF 5%            |       -0.531 |       -0.189 | never worth it  |
| HistGB       | remove OOB 5%           |       -1.666 |       -0.196 | never worth it  |
| HistGB       | remove both (DTT sizes) |       -2.338 |       -0.955 | never worth it  |
| HistGB       | reweight OOB            |       -0.038 |        0.185 | 0.17            |
| HistGB       | winsorize p1-99         |       -0.035 |        0.005 | 0.88            |
| KNN          | remove IF 5%            |       -0.571 |       -0.169 | never worth it  |
| KNN          | remove OOB 5%           |       -0.853 |       -0.02  | never worth it  |
| KNN          | remove both (DTT sizes) |       -1.556 |       -0.879 | never worth it  |
| KNN          | winsorize p1-99         |        0.089 |        0.063 | always worth it |
| LogReg       | remove IF 5%            |       -0.095 |        1.375 | 0.06            |
| LogReg       | remove OOB 5%           |       -0.338 |        0.673 | 0.33            |
| LogReg       | remove both (DTT sizes) |       -0.808 |        2.725 | 0.23            |
| LogReg       | reweight OOB            |       -0.391 |        0.462 | 0.46            |
| LogReg       | winsorize p1-99         |        0.135 |        1.249 | always worth it |
| RandomForest | remove IF 5%            |       -0.116 |       -0.026 | never worth it  |
| RandomForest | remove OOB 5%           |       -1.272 |       -0.404 | never worth it  |
| RandomForest | remove both (DTT sizes) |       -1.509 |       -1.053 | never worth it  |
| RandomForest | reweight OOB            |        0.079 |        0.036 | always worth it |
| RandomForest | winsorize p1-99         |        0.094 |        0.018 | always worth it |
