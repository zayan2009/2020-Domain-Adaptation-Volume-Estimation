## Baseline

| Method |                     Model                      |  MAE  |  MAPE  |  MSPE  | Time  |
| :----: | :--------------------------------------------: | :---: | :----: | :----: | :---: |
|   BL   |               Linear Regression                | 53.66 | 42.99% | 22.67% | 0.896 |
|   BL   |              Random Forest (100)               | 37.79 | 35.91% | 15.17% | 46.7  |
|   BL   |            k-Nearest Neighbor (20)             | 36.74 | 33.18% | 13.04% | 11.3  |
|   BL   | Gradient Boosting Desicion Tree (8, 100, 5000) | 36.82 | 33.20% | 12.96% | 142.3 |
|   BL   |      Multi-Layer Perceptron (0.5-dropout)      | 34.35 | 36.20% | 13.42% | 99.0  |

## Sample-based method

| Method |                   Model                    |  MAE  |  MAPE  |  MSPE  | Time  |
| :----: | :----------------------------------------: | :---: | :----: | :----: | :---: |
|   SW   | Logit weights+GBM regressor (all features) | 34.04 | 31.38% | 10.98% | 31.40 |
|   SW   |  MLP weights+GBM regressor (all features)  | 36.04 | 33.29% | 12.53% | 28.72 |
|   SW   | Logit weights+MLP regressor (all features) | 32.17 | 31.98% | 10.83% | 936.3 |
|   SW   |  MLP weights+MLP regressor (all features)  | 33.54 | 34.71% | 12.68% | 969.0 |

## Feature-based method

| Method |                  Model                  |  MAE  |  MAPE  |  MSPE  | Time  |
| :----: | :-------------------------------------: | :---: | :----: | :----: | :---: |
|  LDS   | SVD trans.+GBM regressor (all features) | 40.68 | 36.18% | 14.90% | 152.5 |
|  LDS   | SVD trans.+MLP regressor (all features) |       |        |        |       |
|  NLDS  | AE trans.+GBM regressor (all features)  |       |        |        |       |
|  NLDS  | AE trans.+MLP regressor (all features)  |       |        |        |       |

## Advanced method