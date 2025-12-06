# -----------------------------
# eda_real_estate.py
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
df = pd.read_csv("properties_preprocessed.csv")

# Ensure Good_Investment exists
if 'Good_Investment' not in df.columns:
    raise ValueError("Good_Investment column not found. Run preprocessing first.")

# -----------------------------
# 1️⃣ Basic statistics
# -----------------------------
print("Dataset shape:", df.shape)
print("\nNumeric columns summary:")
print(df.describe())

print("\nCategorical columns summary:")
print(df.select_dtypes(include='object').nunique())

# -----------------------------
# 2️⃣ Target distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Good_Investment', data=df)
plt.title("Distribution of Good Investment")
plt.show()

# -----------------------------
# 3️⃣ Price trends by city
# -----------------------------
plt.figure(figsize=(12,6))
top_cities = df['City'].value_counts().nlargest(10).index
sns.boxplot(x='City', y='Price_in_Lakhs', data=df[df['City'].isin(top_cities)])
plt.title("Price Distribution by Top 10 Cities")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 4️⃣ Correlation matrix
# -----------------------------
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 5️⃣ Impact of crime rate on Good Investment
# -----------------------------
if 'Security' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='Security', hue='Good_Investment', data=df)
    plt.title("Security Level vs Good Investment")
    plt.show()

# -----------------------------
# 6️⃣ Amenities impact
# -----------------------------
if 'Amenities_count' in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Good_Investment', y='Amenities_count', data=df)
    plt.title("Amenities Count vs Good Investment")
    plt.show()

# -----------------------------
# 7️⃣ School density vs Good Investment
# -----------------------------
if 'School_Density' in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Good_Investment', y='School_Density', data=df)
    plt.title("School Density vs Good Investment")
    plt.show()

# -----------------------------
# 8️⃣ Price vs Size
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', hue='Good_Investment', data=df, alpha=0.6)
plt.title("Price vs Size")
plt.show()

print("EDA completed!")
