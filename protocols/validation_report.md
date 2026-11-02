# Validation of scorer implementations

cleanlab is not installed: checks 3 and 4 skipped (`pip install cleanlab`).

OpenDataVal is not installed: optional check 5 skipped (`pip install opendataval`).

## Summary

| check                                       |   passed |   datasets | result   |
|:--------------------------------------------|---------:|-----------:|:---------|
| 1 KNN-Shapley vs Monte-Carlo Shapley        |        4 |          4 | PASS     |
| 2 Data-OOB vs sklearn BaggingClassifier OOB |        4 |          4 | PASS     |

## Details

| check                                       | dataset       |   n |   spearman |   max_abs_diff | passed   |   auroc_ours |   auroc_ref |
|:--------------------------------------------|:--------------|----:|-----------:|---------------:|:---------|-------------:|------------:|
| 1 KNN-Shapley vs Monte-Carlo Shapley        | breast_cancer |  60 |     0.9893 |         0.0012 | True     |     nan      |    nan      |
| 1 KNN-Shapley vs Monte-Carlo Shapley        | wine          |  60 |     0.9922 |         0.001  | True     |     nan      |    nan      |
| 1 KNN-Shapley vs Monte-Carlo Shapley        | iris          |  60 |     0.9901 |         0.0011 | True     |     nan      |    nan      |
| 1 KNN-Shapley vs Monte-Carlo Shapley        | digits        |  60 |     0.9995 |         0.0005 | True     |     nan      |    nan      |
| 2 Data-OOB vs sklearn BaggingClassifier OOB | breast_cancer | 569 |     0.971  |       nan      | True     |       0.9784 |      0.9803 |
| 2 Data-OOB vs sklearn BaggingClassifier OOB | wine          | 178 |     0.9813 |       nan      | True     |       0.9979 |      0.9979 |
| 2 Data-OOB vs sklearn BaggingClassifier OOB | iris          | 150 |     0.9791 |       nan      | True     |       0.9916 |      0.9921 |
| 2 Data-OOB vs sklearn BaggingClassifier OOB | digits        | 600 |     0.9861 |       nan      | True     |       0.9902 |      0.9902 |
