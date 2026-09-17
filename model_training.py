import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# XGBoost is optional — fall back gracefully if it isn't installed
try:
    from importlib import import_module

    XGBRegressor = import_module("xgboost").XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Load the fully processed dataset produced by feature_engineering.py

data = pd.read_csv("processed_data.csv")
print("Shape :", data.shape)

TARGET = "Delivery_Time"

X = data.drop(columns=[TARGET])
y = data[TARGET]

# Train / test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n—-----Train/Test Split------")
print(" Train shape:", X_train.shape, " Test shape:", X_test.shape)


def evaluate(model, X_tr, y_tr, X_te, y_te):
    """Fit a model, return test-set metrics and 5-fold CV RMSE."""
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_tr, y_tr, cv=kf, scoring="neg_root_mean_squared_error"
    )
    cv_rmse = -cv_scores.mean()

    return {"model": model, "rmse": rmse, "mae": mae, "r2": r2, "cv_rmse": cv_rmse}

#   - Linear Regression / Ridge
#   - Random Forest
#   - Gradient Boosting
#   - XGBoost (if available)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    ),
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

print("\n—-----Model Training & Evaluation------")
results = {}
for name, model in models.items():
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    results[name] = metrics
    print(
        f" {name:<20} | Test RMSE: {metrics['rmse']:.4f} | "
        f"Test MAE: {metrics['mae']:.4f} | Test R2: {metrics['r2']:.4f} | "
        f"5-Fold CV RMSE: {metrics['cv_rmse']:.4f}"
    )

# Pick the best model
# Selection metric: 5-fold CV RMSE (more reliable than a single test split)

best_name = min(results, key=lambda n: results[n]["cv_rmse"])
best_model = results[best_name]["model"]

print("\n—-----Best Model Selected------")
print(f" Best model: {best_name}")
print(f" Test RMSE : {results[best_name]['rmse']:.4f}")
print(f" Test MAE  : {results[best_name]['mae']:.4f}")
print(f" Test R2   : {results[best_name]['r2']:.4f}")

# Feature importance (only meaningful for tree-based models)
if hasattr(best_model, "feature_importance_"):
    feature_importance = pd.Series(
        best_model.feature_importance_, index=X_train.columns
    ).sort_values(ascending=False)
    print("\n—-----Top 10 Feature Importance------")
    print(feature_importance.head(10))

# Save the best model
joblib.dump(best_model, "best_delivery_time_model.pkl")   # change path as needed
print("\n-----------Saved best model to 'best_delivery_time_model.pkl'--------------")