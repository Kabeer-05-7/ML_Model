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
