import numpy as np
import pandas as pd 
import ast
# Loading Dataset
data = pd.read_csv("Food_Delivery_Time_Prediction.csv")
print("Shape :", data.shape)
print("Columns :", data.columns.tolist())
# Handling missing values
print("\nMissing values before:\n", data.isnull().sum())
data.dropna(subset=["Delivery_Time"], inplace=True)

numeric_cols = data.select_dtypes(include=np.number).columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

categorical_cols = data.select_dtypes(exclude=np.number).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing values after :\n", data.isnull().sum())
print("\nDuplicate rows before:", data.duplicated().sum())

data.drop_duplicates(subset=["Order_ID"], keep="first", inplace=True)
data.reset_index(drop=True, inplace=True)
print("Duplicate rows after :", data.duplicated().sum())

# Order_ID → plain string
data["Order_ID"] = data["Order_ID"].astype(str).str.strip()

data["Customer_Lat"] = data["Customer_Location"].apply(lambda x: float(ast.literal_eval(x)[0]))
data["Customer_Lon"] = data["Customer_Location"].apply(lambda x: float(ast.literal_eval(x)[1]))
data.drop(columns=["Customer_Location"], inplace=True)

data["Restaurant_Lat"] = data["Restaurant_Location"].apply(lambda x: float(ast.literal_eval(x)[0]))
data["Restaurant_Lon"] = data["Restaurant_Location"].apply(lambda x: float(ast.literal_eval(x)[1]))
data.drop(columns=["Restaurant_Location"], inplace = True)

data["Distance"] = pd.to_numeric(data["Distance"],errors="coerce").astype("float64")
data["Delivery_Person_Experience"] = pd.to_numeric(data["Delivery_Person_Experience"],errors="coerce").astype("Int64")
data["Restaurant_Rating"] = pd.to_numeric(data["Restaurant_Rating"],errors="coerce").astype("float32")
data["Customer_Rating"] = pd.to_numeric(data["Customer_Rating"],errors="coerce").astype("float32")
data["Delivery_Time"] = pd.to_numeric(data["Delivery_Time"],errors="coerce").astype("float32")
data["Order_Cost"] = pd.to_numeric(data["Order_Cost"],errors="coerce").astype("float32")
data["Tip_Amount"] = pd.to_numeric(data["Tip_Amount"],errors="coerce").astype("float32")
 
# Ordered categorical columns
data["Traffic_Conditions"] = pd.Categorical(data["Traffic_Conditions"],
                               categories=["Low", "Medium", "High"], ordered=True)
data["Order_Priority"]     = pd.Categorical(data["Order_Priority"],
                               categories=["Low", "Medium", "High"], ordered=True)
data["Order_Time"]         = pd.Categorical(data["Order_Time"],
                               categories=["Morning", "Afternoon", "Evening", "Night"], ordered=True)
 
# Unordered categorical columns
# No natural ranking between Sunny / Rainy or Car / Bike — so ordered=False
data["Weather_Conditions"] = pd.Categorical(data["Weather_Conditions"] , ordered = False)
data["Vehicle_Type"] = pd.Categorical(data["Vehicle_Type"] , ordered = False)

print("\n── Shape           :", data.shape)
print("── Missing values  :", data.isnull().sum().sum())
print("\n── dtypes ──\n",      data.dtypes)
print("\n── First 3 rows ──\n", data.head(3))