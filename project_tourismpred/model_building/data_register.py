# pandas reads the CSV and provides tabular validation utilities.
import pandas as pd

# Keep the raw dataset path in one constant so it is easy to change later.
RAW_PATH = "project_tourismpred/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'CityTier',
    'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
    'ProdTaken', 'DurationOfPitch', 'Age'
]

# Collect all missing required fields so the error message is actionable.
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Product Taken Distribution:")
print(df["ProdTaken"].value_counts())
