import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("properties.csv")
print("Initial dataset shape:", df.shape)

# Remove duplicates
df = df.drop_duplicates(subset='ID')
print("After removing duplicates:", df.shape)

# Dynamically identify numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Keep only columns that exist in df
numeric_cols = [c for c in numeric_cols if c in df.columns]

# Categorical columns = all others excluding ID
cat_cols = [c for c in df.columns if c not in numeric_cols + ['ID']]

# Fill missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[cat_cols] = df[cat_cols].fillna("Unknown")

# Feature Engineering
df['Price_per_SqFt_calc'] = df['Price_in_Lakhs'] * 100000 / df['Size_in_SqFt']

# Convert Amenities to count (if it’s comma-separated string)
if 'Amenities' in df.columns:
    df['Amenities_count'] = df['Amenities'].apply(lambda x: len(str(x).split(',')) if x != "Unknown" else 0)
else:
    df['Amenities_count'] = 0

# School/Hospital density
if 'Nearby_Schools' in df.columns and 'Size_in_SqFt' in df.columns:
    df['School_Density'] = df['Nearby_Schools'] / df['Size_in_SqFt']
if 'Nearby_Hospitals' in df.columns and 'Size_in_SqFt' in df.columns:
    df['Hospital_Density'] = df['Nearby_Hospitals'] / df['Size_in_SqFt']

# Age category
if 'Age_of_Property' in df.columns:
    bins = [0, 5, 15, 50]
    labels = ['New', 'Moderate', 'Old']
    df['Age_Category'] = pd.cut(df['Age_of_Property'], bins=bins, labels=labels, include_lowest=True)

# Create target: Good Investment (binary)
# Example heuristic: low price per sqft & high amenities
df['Good_Investment'] = ((df['Price_per_SqFt_calc'] < df['Price_per_SqFt_calc'].median()) &
                         (df['Amenities_count'] > df['Amenities_count'].median())).astype(int)

print("Preprocessing complete!")
print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)
print("Target distribution:\n", df['Good_Investment'].value_counts())

# Save preprocessed dataset for EDA and modeling
df.to_csv("properties_preprocessed.csv", index=False)
print("Preprocessed dataset saved as properties_preprocessed.csv")
