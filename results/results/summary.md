# Motivating experiment: summary

Datasets: 20. Seeds per dataset: 3.

## Overlap by pair type (mean over datasets)

|   frac | pair_type           |   jaccard |   jaccard_sd |   jaccard_random |   spearman |
|-------:|:--------------------|----------:|-------------:|-----------------:|-----------:|
|   0.05 | cross-family        |     0.031 |        0.028 |            0.026 |      0.073 |
|   0.05 | within-classical    |     0.278 |        0.107 |            0.026 |      0.56  |
|   0.05 | within-target-aware |     0.324 |        0.102 |            0.026 |      0.588 |
|   0.1  | cross-family        |     0.061 |        0.039 |            0.053 |      0.073 |
|   0.1  | within-classical    |     0.318 |        0.09  |            0.053 |      0.56  |
|   0.1  | within-target-aware |     0.376 |        0.095 |            0.053 |      0.588 |

## Overlap per method pair

|   frac | a                 | b                 |   jaccard |   spearman |
|-------:|:------------------|:------------------|----------:|-----------:|
|   0.05 | DataOOB           | KNNShapley        |     0.214 |      0.43  |
|   0.05 | DataOOB           | SelfConfidence    |     0.505 |      0.845 |
|   0.05 | IQR               | DataOOB           |     0.026 |     -0.025 |
|   0.05 | IQR               | IsolationForest   |     0.298 |      0.763 |
|   0.05 | IQR               | KNNShapley        |     0.031 |      0.044 |
|   0.05 | IQR               | LOF               |     0.177 |      0.366 |
|   0.05 | IQR               | MAD               |     0.534 |      0.679 |
|   0.05 | IQR               | RobustMahalanobis |     0.347 |      0.613 |
|   0.05 | IQR               | SelfConfidence    |     0.027 |     -0.062 |
|   0.05 | IsolationForest   | DataOOB           |     0.027 |      0.038 |
|   0.05 | IsolationForest   | KNNShapley        |     0.028 |      0.103 |
|   0.05 | IsolationForest   | SelfConfidence    |     0.028 |      0.02  |
|   0.05 | KNNShapley        | SelfConfidence    |     0.253 |      0.487 |
|   0.05 | LOF               | DataOOB           |     0.059 |      0.219 |
|   0.05 | LOF               | IsolationForest   |     0.254 |      0.546 |
|   0.05 | LOF               | KNNShapley        |     0.025 |      0.254 |
|   0.05 | LOF               | SelfConfidence    |     0.059 |      0.248 |
|   0.05 | MAD               | DataOOB           |     0.021 |     -0.004 |
|   0.05 | MAD               | IsolationForest   |     0.233 |      0.637 |
|   0.05 | MAD               | KNNShapley        |     0.03  |      0.048 |
|   0.05 | MAD               | LOF               |     0.136 |      0.302 |
|   0.05 | MAD               | RobustMahalanobis |     0.197 |      0.484 |
|   0.05 | MAD               | SelfConfidence    |     0.022 |     -0.027 |
|   0.05 | RobustMahalanobis | DataOOB           |     0.023 |      0.057 |
|   0.05 | RobustMahalanobis | IsolationForest   |     0.351 |      0.755 |
|   0.05 | RobustMahalanobis | KNNShapley        |     0.02  |      0.085 |
|   0.05 | RobustMahalanobis | LOF               |     0.217 |      0.539 |
|   0.05 | RobustMahalanobis | SelfConfidence    |     0.025 |      0.077 |
|   0.1  | DataOOB           | KNNShapley        |     0.26  |      0.43  |
|   0.1  | DataOOB           | SelfConfidence    |     0.562 |      0.845 |
|   0.1  | IQR               | DataOOB           |     0.06  |     -0.025 |
|   0.1  | IQR               | IsolationForest   |     0.359 |      0.763 |
|   0.1  | IQR               | KNNShapley        |     0.054 |      0.044 |
|   0.1  | IQR               | LOF               |     0.2   |      0.366 |
|   0.1  | IQR               | MAD               |     0.563 |      0.679 |
|   0.1  | IQR               | RobustMahalanobis |     0.408 |      0.613 |
|   0.1  | IQR               | SelfConfidence    |     0.052 |     -0.062 |
|   0.1  | IsolationForest   | DataOOB           |     0.061 |      0.038 |
|   0.1  | IsolationForest   | KNNShapley        |     0.052 |      0.103 |
|   0.1  | IsolationForest   | SelfConfidence    |     0.055 |      0.02  |
|   0.1  | KNNShapley        | SelfConfidence    |     0.305 |      0.487 |
|   0.1  | LOF               | DataOOB           |     0.103 |      0.219 |
|   0.1  | LOF               | IsolationForest   |     0.29  |      0.546 |
|   0.1  | LOF               | KNNShapley        |     0.065 |      0.254 |
|   0.1  | LOF               | SelfConfidence    |     0.106 |      0.248 |
|   0.1  | MAD               | DataOOB           |     0.05  |     -0.004 |
|   0.1  | MAD               | IsolationForest   |     0.289 |      0.637 |
|   0.1  | MAD               | KNNShapley        |     0.055 |      0.048 |
|   0.1  | MAD               | LOF               |     0.161 |      0.302 |
|   0.1  | MAD               | RobustMahalanobis |     0.265 |      0.484 |
|   0.1  | MAD               | SelfConfidence    |     0.045 |     -0.027 |
|   0.1  | RobustMahalanobis | DataOOB           |     0.042 |      0.057 |
|   0.1  | RobustMahalanobis | IsolationForest   |     0.375 |      0.755 |
|   0.1  | RobustMahalanobis | KNNShapley        |     0.038 |      0.085 |
|   0.1  | RobustMahalanobis | LOF               |     0.236 |      0.539 |
|   0.1  | RobustMahalanobis | SelfConfidence    |     0.038 |      0.077 |

## Downstream effect of removing top-k (balanced accuracy)

Wilcoxon signed-rank across datasets (needs >= 6 datasets). With many tests, apply Holm correction before claiming significance.

|   frac | model        | method            |   mean_delta_vs_none |   wins |   losses |   p_vs_none |   mean_delta_vs_random |   p_vs_random |
|-------:|:-------------|:------------------|---------------------:|-------:|---------:|------------:|-----------------------:|--------------:|
|   0.05 | HistGB       | DataOOB           |              -0.0123 |      4 |       16 |      0.0038 |                -0.0107 |        0.0005 |
|   0.05 | HistGB       | IQR               |              -0.0009 |      7 |       13 |      0.3884 |                 0.0007 |        0.9479 |
|   0.05 | HistGB       | IsolationForest   |              -0.0029 |      5 |       15 |      0.0759 |                -0.0013 |        0.6477 |
|   0.05 | HistGB       | KNNShapley        |              -0.0075 |      6 |       14 |      0.0172 |                -0.0059 |        0.0973 |
|   0.05 | HistGB       | LOF               |              -0.0033 |      6 |       14 |      0.04   |                -0.0017 |        0.4939 |
|   0.05 | HistGB       | MAD               |              -0.0022 |      7 |       13 |      0.2024 |                -0.0006 |        0.9039 |
|   0.05 | HistGB       | RobustMahalanobis |              -0.0005 |      7 |        5 |      0.8501 |                -0      |        0.791  |
|   0.05 | HistGB       | SelfConfidence    |              -0.0131 |      3 |       17 |      0.0009 |                -0.0115 |        0.0083 |
|   0.05 | HistGB       | random            |              -0.0016 |      4 |       16 |      0.0973 |               nan      |      nan      |
|   0.05 | KNN          | DataOOB           |              -0.0037 |      8 |       12 |      0.1893 |                -0.0024 |        0.498  |
|   0.05 | KNN          | IQR               |              -0.0036 |      9 |       11 |      0.0441 |                -0.0024 |        0.165  |
|   0.05 | KNN          | IsolationForest   |              -0.0027 |      5 |       15 |      0.0637 |                -0.0014 |        0.1474 |
|   0.05 | KNN          | KNNShapley        |              -0.0061 |      6 |       14 |      0.0532 |                -0.0048 |        0.1474 |
|   0.05 | KNN          | LOF               |              -0.0046 |      4 |       16 |      0.0083 |                -0.0033 |        0.0637 |
|   0.05 | KNN          | MAD               |              -0.0023 |      8 |       12 |      0.1429 |                -0.001  |        0.7172 |
|   0.05 | KNN          | RobustMahalanobis |              -0.0073 |      3 |        9 |      0.064  |                -0.006  |        0.1099 |
|   0.05 | KNN          | SelfConfidence    |              -0.0037 |      8 |       12 |      0.156  |                -0.0025 |        0.1536 |
|   0.05 | KNN          | random            |              -0.0013 |      7 |       13 |      0.3118 |               nan      |      nan      |
|   0.05 | LogReg       | DataOOB           |              -0.0017 |      6 |       14 |      0.2305 |                -0.0012 |        0.4091 |
|   0.05 | LogReg       | IQR               |              -0.0026 |      8 |       12 |      0.0696 |                -0.0021 |        0.0486 |
|   0.05 | LogReg       | IsolationForest   |              -0.0031 |      7 |       13 |      0.114  |                -0.0027 |        0.0826 |
|   0.05 | LogReg       | KNNShapley        |              -0.0036 |      8 |       12 |      0.0966 |                -0.0031 |        0.1054 |
|   0.05 | LogReg       | LOF               |              -0.0028 |      7 |       12 |      0.2432 |                -0.0024 |        0.159  |
|   0.05 | LogReg       | MAD               |              -0.0014 |      8 |       11 |      0.2273 |                -0.0009 |        0.4009 |
|   0.05 | LogReg       | RobustMahalanobis |              -0.0073 |      1 |       10 |      0.0137 |                -0.0067 |        0.0122 |
|   0.05 | LogReg       | SelfConfidence    |              -0.0018 |      7 |       13 |      0.2043 |                -0.0014 |        0.2774 |
|   0.05 | LogReg       | random            |              -0.0004 |      6 |       13 |      0.3143 |               nan      |      nan      |
|   0.05 | RandomForest | DataOOB           |              -0.0095 |      6 |       13 |      0.0126 |                -0.0097 |        0.0032 |
|   0.05 | RandomForest | IQR               |              -0.0018 |      6 |       14 |      0.1769 |                -0.002  |        0.1429 |
|   0.05 | RandomForest | IsolationForest   |              -0.0023 |      9 |       11 |      0.5217 |                -0.0025 |        0.2471 |
|   0.05 | RandomForest | KNNShapley        |              -0.0083 |      5 |       15 |      0.024  |                -0.0085 |        0.0239 |
|   0.05 | RandomForest | LOF               |              -0.0034 |      8 |       11 |      0.2598 |                -0.0036 |        0.1327 |
|   0.05 | RandomForest | MAD               |              -0.0039 |      7 |       13 |      0.0696 |                -0.0041 |        0.0826 |
|   0.05 | RandomForest | RobustMahalanobis |              -0.0034 |      5 |        7 |      0.7334 |                -0.004  |        0.3652 |
|   0.05 | RandomForest | SelfConfidence    |              -0.0097 |      7 |       13 |      0.0136 |                -0.0098 |        0.0083 |
|   0.05 | RandomForest | random            |               0.0002 |      7 |       12 |      0.4445 |               nan      |      nan      |
|   0.1  | HistGB       | DataOOB           |              -0.0202 |      4 |       16 |      0.0027 |                -0.0182 |        0.002  |
|   0.1  | HistGB       | IQR               |              -0.0058 |      4 |       16 |      0.0073 |                -0.0037 |        0.0583 |
|   0.1  | HistGB       | IsolationForest   |              -0.0079 |      4 |       16 |      0.0073 |                -0.0058 |        0.0586 |
|   0.1  | HistGB       | KNNShapley        |              -0.0157 |      5 |       15 |      0.0192 |                -0.0136 |        0.024  |
|   0.1  | HistGB       | LOF               |              -0.0087 |      5 |       15 |      0.0136 |                -0.0066 |        0.024  |
|   0.1  | HistGB       | MAD               |              -0.0102 |      2 |       18 |      0.0037 |                -0.0081 |        0.0192 |
|   0.1  | HistGB       | RobustMahalanobis |              -0.0059 |      4 |        8 |      0.2334 |                -0.0046 |        0.791  |
|   0.1  | HistGB       | SelfConfidence    |              -0.0218 |      3 |       17 |      0.0002 |                -0.0197 |        0.0004 |
|   0.1  | HistGB       | random            |              -0.0021 |      6 |       14 |      0.2548 |               nan      |      nan      |
|   0.1  | KNN          | DataOOB           |              -0.0059 |      7 |       13 |      0.1327 |                -0.0005 |        0.9273 |
|   0.1  | KNN          | IQR               |              -0.0085 |      3 |       17 |      0.0012 |                -0.0031 |        0.2305 |
|   0.1  | KNN          | IsolationForest   |              -0.0084 |      2 |       18 |      0.0001 |                -0.0029 |        0.1327 |
|   0.1  | KNN          | KNNShapley        |              -0.0171 |      6 |       14 |      0.0037 |                -0.0116 |        0.0442 |
|   0.1  | KNN          | LOF               |              -0.0072 |      6 |       14 |      0.0094 |                -0.0018 |        0.6742 |
|   0.1  | KNN          | MAD               |              -0.0141 |      2 |       18 |      0      |                -0.0087 |        0.0696 |
|   0.1  | KNN          | RobustMahalanobis |              -0.0128 |      1 |       11 |      0.0093 |                -0.0061 |        0.064  |
|   0.1  | KNN          | SelfConfidence    |              -0.0113 |      6 |       14 |      0.0064 |                -0.0059 |        0.1054 |
|   0.1  | KNN          | random            |              -0.0054 |      4 |       16 |      0.0083 |               nan      |      nan      |
|   0.1  | LogReg       | DataOOB           |              -0.002  |      8 |       12 |      0.4091 |                -0.0011 |        0.9563 |
|   0.1  | LogReg       | IQR               |              -0.0049 |      4 |       15 |      0.0242 |                -0.004  |        0.1909 |
|   0.1  | LogReg       | IsolationForest   |              -0.0063 |      5 |       15 |      0.0328 |                -0.0054 |        0.114  |
|   0.1  | LogReg       | KNNShapley        |              -0.0126 |      5 |       15 |      0.0014 |                -0.0117 |        0.0073 |
|   0.1  | LogReg       | LOF               |              -0.0059 |      5 |       15 |      0.0646 |                -0.005  |        0.3317 |
|   0.1  | LogReg       | MAD               |              -0.0116 |      3 |       16 |      0.001  |                -0.0107 |        0.0136 |
|   0.1  | LogReg       | RobustMahalanobis |              -0.005  |      4 |        8 |      0.1294 |                -0.0048 |        0.3804 |
|   0.1  | LogReg       | SelfConfidence    |              -0.0045 |      7 |       13 |      0.0583 |                -0.0036 |        0.2162 |
|   0.1  | LogReg       | random            |              -0.0009 |      5 |       14 |      0.0766 |               nan      |      nan      |
|   0.1  | RandomForest | DataOOB           |              -0.0164 |      3 |       17 |      0.0014 |                -0.014  |        0.0017 |
|   0.1  | RandomForest | IQR               |              -0.004  |      6 |       14 |      0.0759 |                -0.0016 |        0.2122 |
|   0.1  | RandomForest | IsolationForest   |              -0.0061 |      7 |       13 |      0.0894 |                -0.0036 |        0.4009 |
|   0.1  | RandomForest | KNNShapley        |              -0.0191 |      4 |       16 |      0.0002 |                -0.0166 |        0.0001 |
|   0.1  | RandomForest | LOF               |              -0.008  |      6 |       14 |      0.0532 |                -0.0056 |        0.1054 |
|   0.1  | RandomForest | MAD               |              -0.0095 |      4 |       16 |      0.0037 |                -0.0071 |        0.1054 |
|   0.1  | RandomForest | RobustMahalanobis |              -0.0067 |      3 |        9 |      0.1514 |                -0.0037 |        0.3013 |
|   0.1  | RandomForest | SelfConfidence    |              -0.0197 |      3 |       17 |      0.0002 |                -0.0172 |        0.0001 |
|   0.1  | RandomForest | random            |              -0.0025 |      7 |       13 |      0.2305 |               nan      |      nan      |

## How often each option is the best (per dataset)

If no single option dominates, a recommender is justified.

|   frac | model        | method            |   share |
|-------:|:-------------|:------------------|--------:|
|   0.05 | HistGB       | none              |    0.45 |
|   0.05 | HistGB       | KNNShapley        |    0.1  |
|   0.05 | HistGB       | MAD               |    0.1  |
|   0.05 | HistGB       | SelfConfidence    |    0.1  |
|   0.05 | HistGB       | DataOOB           |    0.05 |
|   0.05 | HistGB       | IQR               |    0.05 |
|   0.05 | HistGB       | IsolationForest   |    0.05 |
|   0.05 | HistGB       | LOF               |    0.05 |
|   0.05 | HistGB       | RobustMahalanobis |    0.05 |
|   0.05 | KNN          | none              |    0.3  |
|   0.05 | KNN          | DataOOB           |    0.15 |
|   0.05 | KNN          | IQR               |    0.15 |
|   0.05 | KNN          | KNNShapley        |    0.15 |
|   0.05 | KNN          | SelfConfidence    |    0.1  |
|   0.05 | KNN          | IsolationForest   |    0.05 |
|   0.05 | KNN          | MAD               |    0.05 |
|   0.05 | KNN          | RobustMahalanobis |    0.05 |
|   0.05 | LogReg       | LOF               |    0.25 |
|   0.05 | LogReg       | MAD               |    0.2  |
|   0.05 | LogReg       | KNNShapley        |    0.15 |
|   0.05 | LogReg       | IsolationForest   |    0.1  |
|   0.05 | LogReg       | SelfConfidence    |    0.1  |
|   0.05 | LogReg       | none              |    0.1  |
|   0.05 | LogReg       | DataOOB           |    0.05 |
|   0.05 | LogReg       | IQR               |    0.05 |
|   0.05 | RandomForest | none              |    0.25 |
|   0.05 | RandomForest | IQR               |    0.15 |
|   0.05 | RandomForest | SelfConfidence    |    0.15 |
|   0.05 | RandomForest | DataOOB           |    0.1  |
|   0.05 | RandomForest | IsolationForest   |    0.1  |
|   0.05 | RandomForest | MAD               |    0.1  |
|   0.05 | RandomForest | RobustMahalanobis |    0.1  |
|   0.05 | RandomForest | KNNShapley        |    0.05 |
|   0.1  | HistGB       | none              |    0.5  |
|   0.1  | HistGB       | KNNShapley        |    0.2  |
|   0.1  | HistGB       | LOF               |    0.1  |
|   0.1  | HistGB       | DataOOB           |    0.05 |
|   0.1  | HistGB       | IQR               |    0.05 |
|   0.1  | HistGB       | MAD               |    0.05 |
|   0.1  | HistGB       | RobustMahalanobis |    0.05 |
|   0.1  | KNN          | none              |    0.45 |
|   0.1  | KNN          | DataOOB           |    0.2  |
|   0.1  | KNN          | LOF               |    0.15 |
|   0.1  | KNN          | KNNShapley        |    0.05 |
|   0.1  | KNN          | MAD               |    0.05 |
|   0.1  | KNN          | RobustMahalanobis |    0.05 |
|   0.1  | KNN          | SelfConfidence    |    0.05 |
|   0.1  | LogReg       | none              |    0.3  |
|   0.1  | LogReg       | LOF               |    0.2  |
|   0.1  | LogReg       | DataOOB           |    0.15 |
|   0.1  | LogReg       | SelfConfidence    |    0.15 |
|   0.1  | LogReg       | KNNShapley        |    0.1  |
|   0.1  | LogReg       | IQR               |    0.05 |
|   0.1  | LogReg       | RobustMahalanobis |    0.05 |
|   0.1  | RandomForest | none              |    0.45 |
|   0.1  | RandomForest | LOF               |    0.15 |
|   0.1  | RandomForest | IsolationForest   |    0.1  |
|   0.1  | RandomForest | RobustMahalanobis |    0.1  |
|   0.1  | RandomForest | DataOOB           |    0.05 |
|   0.1  | RandomForest | IQR               |    0.05 |
|   0.1  | RandomForest | KNNShapley        |    0.05 |
|   0.1  | RandomForest | SelfConfidence    |    0.05 |
