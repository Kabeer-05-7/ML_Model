import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluation_metrics(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # MAE: on average, how many minutes is the prediction off?
    mae = mean_absolute_error(y_true, y_pred)

    # MSE: average of squared errors
    mse = mean_squared_error(y_true, y_pred)

    # RMSE: square root of MSE, so it is in minutes again
    rmse = np.sqrt(mse)

    # R2: 1 = perfect, 0 = no better than guessing the average, negative = worse
    r2 = r2_score(y_true, y_pred)

    # MAPE: average error as a percentage of the actual time
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE_%": mape}

data = pd.read_csv("processed_data.csv")

X = data.drop(columns=["Delivery_Time"])
y = data["Delivery_Time"]

# Split -> creates y_test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("best_delivery_time_model.pkl")

predictions = model.predict(X_test)

results = evaluation_metrics(y_test, predictions)

print("------ Evaluation Metrics ------")
for name, value in results.items():
    print(f"{name:<8}: {value:.4f}")