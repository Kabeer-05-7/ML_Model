# ML_Model
Author -: Kabeer Sharma
Food_Delivery_Time_Prediction_Project
<br>
It predicts that how much time delivery partner will take to deliver the order.
<br>
Predicts food delivery time (minutes) from order, distance, traffic, weather and rider data.

# Pipeline
- `preprocessing.py`: cleans data (missing values, duplicates, location parsing)
- `feature_engineering.py`: creates features (Haversine distance, peak hour, stress score), encodes and scales
- `model_training.py`: trains Linear, Ridge, Random Forest, Gradient Boosting and XGBoost, then saves the best
- `evaluation_metrics.py`: reports MAE, MSE, RMSE, R² and MAPE

# Results

| Model  | MAE (min) | RMSE (min) | R² |
|--------|-----------|------------|----|
|Baseline|   25.54   | 30.43  | -0.001 |
| Random Forest|26.30| 30.79  | -0.025 |
| Ridge | 27.36      | 31.37  | -0.064 |

# Conclusion

No model beat the baseline of always predicting the average delivery time. The dataset has only 200 rows, the features barely correlate with delivery time (strongest: -0.17), and delivery time looks random, so these features cannot predict it. The pipeline and evaluation work correctly, but a larger, real-world dataset is needed.