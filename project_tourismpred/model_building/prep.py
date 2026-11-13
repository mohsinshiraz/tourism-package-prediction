import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("project_tourismpred/data/tourism.csv")

# Drop identifier / index columns safely (without inplace=True)
cols_to_drop = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=cols_to_drop)

# Clean known category inconsistencies in tourism.csv
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

target_col = "ProdTaken"

# Explicitly separate numeric and categorical feature columns (Pandas 2 & 3 compatible)
numeric_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore")
categorical_cols = df.select_dtypes(exclude=["number"]).columns.drop(target_col, errors="ignore")

# Impute missing values safely by column type
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed splits
os.makedirs("project_tourismpred/data", exist_ok=True)
Xtrain.to_csv("project_tourismpred/data/Xtrain.csv", index=False)
Xtest.to_csv("project_tourismpred/data/Xtest.csv", index=False)
ytrain.to_csv("project_tourismpred/data/ytrain.csv", index=False)
ytest.to_csv("project_tourismpred/data/ytest.csv", index=False)

print("Data preparation completed successfully.")
