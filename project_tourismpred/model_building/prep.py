import pandas as pd
from sklearn.model_selection import train_test_split

# Load the source data from the project data directory.
df = pd.read_csv("project_tourismpred/data/tourism.csv")

# CustomerID is an identifier, not a predictive feature; remove it to avoid spurious learning.
df.drop(columns=["CustomerID"], inplace=True)

# NOTE: 'Type' is intentionally left as raw strings (H/L/M).
# The training pipeline one-hot-encodes it, and the Streamlit app also sends
# raw H/L/M values. Encoding it here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

# Separate input features (X) from the binary outcome label (y).
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Normalize known inconsistent category labels before modeling.
# Fix Typos
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')
df['Occupation'] = df['Occupation'].str.strip()

# Impute missing values: most frequent category for text columns, median for numeric columns.
# Handle Missing Values (Impute with Median for Num, Mode for Cat)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# stratify=y keeps the (imbalanced) ProdTaken ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save each split as an artifact that a later pipeline job can download.
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type of Contact values kept as:", sorted(X["TypeofContact"].unique()))
