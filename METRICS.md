## Regression Metrics (Practical Overview)
| Metric                                | Formula (conceptually)                                                    | Interpretation                                                                                         |   |                                                                                  |
| :------------------------------------ | :------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------- | - | -------------------------------------------------------------------------------- |
| **MSE** (Mean Squared Error)          | ( $\text{MSE} = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$ )                    | Penalizes large errors heavily (quadratic). Sensitive to outliers. Lower is better.                    |   |                                                                                  |
| **MAE** (Mean Absolute Error)         | ( $\text{MAE} = \frac{1}{N}\sum_i   \| y_i - \hat{y}_i \| $) | Average absolute difference in target units (e.g., kW). More robust to outliers. |
| **RMSE** (Root Mean Squared Error)    | ( $\text{RMSE} = \sqrt{\text{MSE}}$ )                                       | Same units as target; highlights large deviations. Easier to interpret than MSE.                       |   |                                                                                  |
| **R² (Coefficient of Determination)** | ( $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$ ) | Measures variance explained by model (1 = perfect fit, 0 = constant prediction, <0 = worse than mean). |   |                                                                                  |

### Quick intuition
- Use MAE for practical “average error” interpretation (e.g., “model misses by ±0.2 kW”).
- Use R² to gauge global performance (“model explains 98.7 % of variance”).
- Use RMSE for a more penalized view (sensitive to spikes/outliers).