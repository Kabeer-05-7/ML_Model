from preprocessing import preprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("Food_Delivery_Time_Prediction.csv")
data = preprocess(data)
def feature_engineering(data):
    # STEP 1 — FEATURE CREATION

    # ── 1a. Haversine Distance (straight-line km between restaurant & customer) ───
    # Why: data["Distance"] is road distance. Haversine gives geographic distance —
    # a different angle on how far the delivery actually is.
    # Formula uses lat/lon coordinates already parsed in preprocessing.
    
    R = 6371  # Earth radius in km
    
    lat1 = np.radians(data["Restaurant_Lat"])
    lat2 = np.radians(data["Customer_Lat"])
    dlat = np.radians(data["Customer_Lat"] - data["Restaurant_Lat"])
    dlon = np.radians(data["Customer_Lon"] - data["Restaurant_Lon"])
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    data["Haversine_Distance_km"] = 2 * R * np.arcsin(np.sqrt(a))
    
    # ── 1b. Is Peak Hour (binary flag: 1 = busy slot, 0 = quiet slot) ─────────────
    # Why: Afternoon & Evening have the highest order volume → longer delivery times.
    # A 0/1 flag is simpler for the model to learn than the full text category.
    
    data["Is_Peak_Hour"] = data["Order_Time"].isin(["Afternoon", "Evening"]).astype(int)
    
    # ── 1c. Delivery Stress Score (traffic severity × weather severity) ────────────
    # Why: High traffic + bad weather together cause far worse delays than either
    # alone. Multiplying their numeric ranks creates a combined pressure index.
    # Range: 1 (Low traffic, Sunny) → 12 (High traffic, Snowy)
    
    traffic_rank = {"Low": 1, "Medium": 2, "High": 3}
    weather_rank = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Snowy": 4}
    
    data["Delivery_Stress_Score"] = (
        data["Traffic_Conditions"].map(traffic_rank).astype(int) *
        data["Weather_Conditions"].map(weather_rank).astype(int)
    )
    
    # ── 1d. Average Rating (mean of restaurant + customer rating) ──────────────────
    # Why: A single satisfaction score summarising both sides of the transaction.
    # Useful when one rating is missing or unreliable in future data.
    
    data["Avg_Rating"] = (data["Restaurant_Rating"] + data["Customer_Rating"]) / 2
    
    print("\nStep 1 — Feature Creation done")
    print("  New columns:", ["Haversine_Distance_km", "Is_Peak_Hour",
                            "Delivery_Stress_Score", "Avg_Rating"])

    # STEP 2 — FEATURE TRANSFORMATION
    # Reshape skewed numeric columns so their distribution is more symmetric.

    # ── 2a. Log transform Distance ─────────────────────────────────────────────────
    # Why: Most orders are short-distance; a few are very long → right skew.
    # np.log1p(x) = log(1 + x) — safe when x can be 0.
    
    data["Log_Distance"] = np.log1p(data["Distance"])
    
    # ── 2b. Log transform Order_Cost ──────────────────────────────────────────────
    # Why: Cheap orders dominate; expensive ones are rare outliers → right skew.
    
    data["Log_Order_Cost"] = np.log1p(data["Order_Cost"])
    
    # ── 2c. Square-root transform Tip_Amount ──────────────────────────────────────
    # Why: Many orders have zero tip — log(0) is undefined.
    # sqrt is a gentler compression that handles zeros safely.
    
    data["Sqrt_Tip_Amount"] = np.sqrt(data["Tip_Amount"])
    
    print("\nStep 2 — Feature Transformation done")
    print("  New columns:", ["Log_Distance", "Log_Order_Cost", "Sqrt_Tip_Amount"])

    # STEP 3 — ENCODING
    # Convert categorical text columns into numbers — models can't do maths on strings.
    # Different encoding strategies for different column types.

    # ── 3a. Ordinal encoding for ranked categories ─────────────────────────────────
    # Used for: Traffic_Conditions, Order_Priority, Order_Time

    data["Traffic_Conditions_Enc"] = data["Traffic_Conditions"].map({"Low": 1, "Medium": 2, "High": 3})
    data["Order_Priority_Enc"]     = data["Order_Priority"].map({"Low": 1, "Medium": 2, "High": 3})
    data["Order_Time_Enc"]         = data["Order_Time"].map({"Morning": 1, "Afternoon": 2,
                                                        "Evening": 3, "Night": 4})
    
    # ── 3b. Label encoding for Vehicle_Type ───────────────────────────────────────
    # Used for: Vehicle_Type (Bicycle, Bike, Car — 3 categories, no rank)
    # Why LabelEncoder? For tree models a single integer column is enough.
    # sklearn's LabelEncoder assigns alphabetical integers automatically.
    
    le = LabelEncoder()
    data["Vehicle_Type_Enc"] = le.fit_transform(data["Vehicle_Type"])
    
    print("\n  Vehicle_Type mapping →", dict(zip(le.classes_, le.transform(le.classes_))))
    
    # ── 3c. One-hot encoding for Weather_Conditions ────────────────────────────────
    # Used for: Weather_Conditions (Sunny, Cloudy, Rainy, Snowy — no rank)
    # Creates: Weather_Cloudy, Weather_Rainy, Weather_Snowy, Weather_Sunny
    
    data = pd.get_dummies(data, columns=["Weather_Conditions"], prefix="Weather", dtype=int)
    
    print("\nStep 3 — Encoding done")
    print("  Encoded columns:", ["Traffic_Conditions_Enc", "Order_Priority_Enc",
                                "Order_Time_Enc", "Vehicle_Type_Enc",
                                "Weather_Cloudy", "Weather_Rainy",
                                "Weather_Snowy",  "Weather_Sunny"])

    # STEP 4 — SCALING
    # Bring all numeric features to the same 0–1 range using MinMaxScaler.
    # Distance ranges 0–25 km; Avg_Rating ranges 0–5 , Without scaling, Distance would dominate in KNN and SVM
    
    cols_to_scale = [
        "Distance",
        "Haversine_Distance_km",
        "Delivery_Person_Experience",
        "Restaurant_Rating",
        "Customer_Rating",
        "Avg_Rating",
        "Order_Cost",
        "Tip_Amount",
        "Delivery_Stress_Score",
        "Log_Distance",
        "Log_Order_Cost",
        "Sqrt_Tip_Amount",
    ]
    
    # Keep only columns that exist (safety guard if a column was dropped earlier)
    cols_to_scale = [c for c in cols_to_scale if c in data.columns]
    
    scaler = MinMaxScaler()                               # scales to [0, 1]
    data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
    
    print("\nStep 4 — Scaling done")
    print("  Scaled columns:", cols_to_scale)

    # STEP 5 — FEATURE INTERACTION
    # Create new columns by combining two existing features.
    # Why: Some effects only appear when two features act together.
    #      Multiplying them lets the model capture that combined signal.

    # ── 5a. Distance × Traffic ─────────────────────────────────────────────────────
    # Why: A 20 km trip in heavy traffic is far worse than the sum of both.
    # The product captures this "compounded delay" effect.
    
    data["Distance_x_Traffic"] = data["Distance"].astype(float) * data["Traffic_Conditions_Enc"].astype(float)
    
    # ── 5b. Experience × Stress Score ─────────────────────────────────────────────
    # Why: A senior rider handles bad weather + heavy traffic better than a rookie.
    # This interaction captures the "skilled rider under pressure" nuance.
    
    data["Experience_x_Stress"] = data["Delivery_Person_Experience"].astype(float) * data["Delivery_Stress_Score"]
    
    print("\nStep 5 — Feature Interaction done")
    print("  Interaction columns:", ["Distance_x_Traffic", "Experience_x_Stress"])

    # STEP 6 — SAVE PROCESSED DATA
    # Drop columns that have been replaced by encoded versions before saving.
    # Keeping originals alongside encoded ones = duplicate / conflicting info.

    cols_to_drop = [
        "Order_ID",            # identifier, not a feature for the model
        "Traffic_Conditions",  # replaced by Traffic_Conditions_Enc
        "Order_Priority",      # replaced by Order_Priority_Enc
        "Order_Time",          # replaced by Order_Time_Enc
        "Vehicle_Type",        # replaced by Vehicle_Type_Enc
    ]
    
    # Only drop columns that actually exist
    cols_to_drop = [c for c in cols_to_drop if c in data.columns]
    data.drop(columns=cols_to_drop, inplace=True)
    
    data.to_csv("processed_data.csv", index=False)    # change path as needed
    print("\nStep 6 — Saved to 'processed_data.csv'")

    # FINAL CHECK
    print("\n── Final shape     :", data.shape)
    print("── Missing values  :", data.isnull().sum().sum())
    print("── Columns         :\n", data.columns.tolist())
    print("\n── First 2 rows ──\n",  data.head(2))
    return data
data = feature_engineering(data)