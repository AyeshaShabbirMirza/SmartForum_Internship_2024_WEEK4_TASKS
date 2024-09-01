# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
file_path = r"C:\Users\Me\Downloads\food_coded.csv"
data = pd.read_csv(file_path)

# Step 1: Handle Missing Values
# Fill numeric missing values with median
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Fill missing object values with 'Unknown'
object_cols = data.select_dtypes(include=['object']).columns
for col in object_cols:
    data[col].fillna('Unknown', inplace=True)

# Step 2: Correct Data Types
# Convert GPA to numeric, handle errors by coercing invalid values to NaN, then fill with median
data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
data['GPA'].fillna(data['GPA'].median(), inplace=True)

# Clean 'weight' column to extract numeric values and convert to float
# Replace non-numeric responses with NaN, then fill with median
data['weight'] = pd.to_numeric(data['weight'].str.extract('(\d+)')[0], errors='coerce')
data['weight'].fillna(data['weight'].median(), inplace=True)

# Step 3: Drop Redundant or Duplicated Columns
# Drop unnecessary columns that appear redundant
data.drop(['comfort_food_reasons_coded.1'], axis=1, inplace=True)

# Step 4: Standardize Text in Object Columns
# Clean and standardize text data by converting to lowercase and removing extra spaces
text_columns = ['comfort_food', 'comfort_food_reasons', 'father_profession', 'mother_profession']
for col in text_columns:
    data[col] = data[col].str.lower().str.strip()

# Step 5: Handle Outliers
# Cap outliers in key numerical columns to a reasonable range based on domain knowledge
data['calories_chicken'] = np.clip(data['calories_chicken'], 200, 800)
data['waffle_calories'] = np.clip(data['waffle_calories'], 600, 1300)

# Additional Check: Validate Changes
# Ensure all numerical data is within expected ranges and no object columns contain unexpected non-standard values
print("Validation Checks:")
print("GPA Range:", data['GPA'].min(), "-", data['GPA'].max())
print("Weight Range:", data['weight'].min(), "-", data['weight'].max())
print(data.describe())  # For numerical range inspection
print(data.select_dtypes(include=['object']).sample(5))  # To review object data consistency

# Save the cleaned dataset
cleaned_file_path = r"C:\Users\Me\Downloads\food_coded_cleaned.csv"
data.to_csv(cleaned_file_path, index=False)

print("Data cleaning completed. Cleaned file saved at:", cleaned_file_path)
