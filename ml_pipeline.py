# ml_pipeline.py (ultra-fast version)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error
import joblib

DATA_PATH = "properties_preprocessed.csv"

# --------------------------------------
# Load Dataset
# --------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# --------------------------------------
# Define Columns
# --------------------------------------
numeric_cols = [
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
    'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals', 'Price_per_SqFt_calc',
    'Amenities_count', 'School_Density', 'Hospital_Density'
]

categorical_cols = [
    'State', 'City', 'Locality', 'Property_Type', 'Furnished_Status',
    'Public_Transport_Accessibility', 'Parking_Space', 'Security',
    'Amenities', 'Facing', 'Owner_Type', 'Availability_Status', 'Age_Category'
]

# --------------------------------------
# Target Variables
# --------------------------------------
X = df[numeric_cols + categorical_cols]
y_cls = df['Good_Investment']

# If future price exists, use it, else estimate
if "Future_Price_5Y" in df.columns:
    y_reg = df["Future_Price_5Y"]
else:
    y_reg = df['Price_in_Lakhs'] * 1.3  # assume 30% appreciation

# --------------------------------------
# Train/Test Split
# --------------------------------------
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# --------------------------------------
# Preprocessing (Fast)
# --------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ('num', "passthrough", numeric_cols)
    ]
)

# --------------------------------------
# Classifier (Very Fast Model)
# --------------------------------------
classifier = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(random_state=42))
])

classifier.fit(X_train, y_train_cls)
y_pred_cls = classifier.predict(X_test)

print("\nClassification Results:")
print("Accuracy:", round(accuracy_score(y_test_cls, y_pred_cls), 4))
print("Precision:", round(precision_score(y_test_cls, y_pred_cls), 4))
print("Recall:", round(recall_score(y_test_cls, y_pred_cls), 4))
print("ROC-AUC:", round(roc_auc_score(y_test_cls, y_pred_cls), 4))

# --------------------------------------
# Regressor (Future Price Prediction)
# --------------------------------------
regressor = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

regressor.fit(X_train, y_train_reg)
y_pred_reg = regressor.predict(X_test)

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = mse ** 0.5
print("RMSE:", rmse)

# --------------------------------------
# Save Models
# --------------------------------------
joblib.dump(classifier, "real_estate_classifier.pkl")
joblib.dump(regressor, "real_estate_regressor.pkl")

print("\nPipeline executed FAST and models saved successfully!")
