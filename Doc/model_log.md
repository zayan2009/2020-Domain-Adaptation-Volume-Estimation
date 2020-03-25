## Baseline

| Method |                     Model                      |  MAE  |  MAPE  |  MSPE  | Time  |
| :----: | :--------------------------------------------: | :---: | :----: | :----: | :---: |
|   BL   |               Linear Regression                | 53.66 | 42.99% | 22.67% | 0.896 |
|   BL   |              Random Forest (100)               | 37.79 | 35.91% | 15.17% | 46.7  |
|   BL   |            k-Nearest Neighbor (20)             | 36.74 | 33.18% | 13.04% | 11.3  |
|   BL   | Gradient Boosting Desicion Tree (8, 100, 5000) | 36.82 | 33.20% | 12.96% | 142.3 |
|   BL   |      Multi-Layer Perceptron (0.5-dropout)      | 34.35 | 36.20% | 13.42% | 99.0  |

## Sample-based method

* feature selection by GBM implicit mini-max game
* sample weight by logit
* regress by GBM



