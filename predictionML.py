import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Parse arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--output_csv", type=str, required=True)
args = parser.parse_args()

input_csv = args.input_csv
output_csv = args.output_csv

print("Input CSV (raw):", input_csv)
print("Output CSV (raw):", output_csv)

# -------------------------------
# Resolve actual CSV file (AzureML mounts inputs as dirs)
# -------------------------------
if os.path.isdir(input_csv):
    files = [f for f in os.listdir(input_csv) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV file found in {input_csv}")
    input_csv = os.path.join(input_csv, files[0])
    print("Resolved input CSV file:", input_csv)

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv(input_csv)
print("Loaded CSV with shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------------
# Feature engineering
# -------------------------------
# Encode class name
label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['agrim_Object_Detection_Column1_name'])

# Features: class_id + encoded class name
X = df[['agrim_Object_Detection_Column1_class_id', 'class_encoded']]
y = df['agrim_Object_Detection_Column1_score']

# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train regressor
# -------------------------------
regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
regressor.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = regressor.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# -------------------------------
# Add adjusted predictions
# -------------------------------
df['adjusted_confidence'] = regressor.predict(X)

# -------------------------------
# Resolve output path
# -------------------------------
# If user gave a directory, create a file inside it
if os.path.isdir(output_csv):
    output_csv = os.path.join(output_csv, "adjusted_predictions.csv")

# Ensure parent directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# -------------------------------
# Save output
# -------------------------------
df.to_csv(output_csv, index=False)
print("Saved adjusted CSV to", output_csv)
