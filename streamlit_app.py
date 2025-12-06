import streamlit as st
import pandas as pd
import joblib

# Load models
classifier = joblib.load("real_estate_classifier.pkl")
regressor = joblib.load("real_estate_regressor.pkl")

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏡", layout="wide")

st.title("🏡 Real Estate Investment Advisor")
st.write("Enter property details below to check investment potential and 5-year projected price.")

# ---------------------------------------------------------
# Define Columns
# ---------------------------------------------------------
numeric_cols = [
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
    'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals', 'Price_per_SqFt_calc',
    'Amenities_count', 'School_Density', 'Hospital_Density'
]

categorical_cols = {
    "State": ["Maharashtra", "Karnataka", "Delhi", "Telangana", "Tamil Nadu", "Other"],
    "City": ["Pune", "Mumbai", "Hyderabad", "Bengaluru", "Delhi", "Chennai", "Other"],
    "Locality": ["Urban", "Semi-Urban", "Rural"],
    "Property_Type": ["Apartment", "Villa", "Row House", "Studio"],
    "Furnished_Status": ["Furnished", "Semi-Furnished", "Unfurnished"],
    "Public_Transport_Accessibility": ["High", "Medium", "Low"],
    "Parking_Space": ["Yes", "No"],
    "Security": ["High", "Medium", "Low"],
    "Amenities": ["Basic", "Standard", "Premium"],
    "Facing": ["East", "West", "North", "South"],
    "Owner_Type": ["Builder", "Owner", "Agent"],
    "Availability_Status": ["Ready to Move", "Under Construction"],
    "Age_Category": ["New", "5-10 Years", "10-20 Years", "20+ Years"]
}

# ---------------------------------------------------------
# Streamlit Form
# ---------------------------------------------------------
st.subheader("📌 Enter Property Details")

input_data = {}

# 2-column layout
col1, col2 = st.columns(2)

with col1:
    # Numeric Inputs (LEFT SIDE)
    input_data["BHK"] = st.number_input("BHK", min_value=1, max_value=10)
    input_data["Size_in_SqFt"] = st.number_input("Size (Sq Ft)", min_value=100)
    input_data["Price_in_Lakhs"] = st.number_input("Current Price (Lakhs)", min_value=1)
    input_data["Price_per_SqFt"] = st.number_input("Price per Sq Ft", min_value=500)
    input_data["Year_Built"] = st.number_input("Year Built", min_value=1950, max_value=2025)
    input_data["Floor_No"] = st.number_input("Floor No.", min_value=0)
    input_data["Total_Floors"] = st.number_input("Total Floors", min_value=1)
    input_data["Age_of_Property"] = st.number_input("Age of Property (years)", min_value=0)
    input_data["Price_per_SqFt_calc"] = st.number_input("Calculated Price Per Sq Ft", min_value=500)

with col2:
    # Numeric Inputs (RIGHT SIDE)
    input_data["Nearby_Schools"] = st.number_input("Nearby Schools", min_value=0)
    input_data["Nearby_Hospitals"] = st.number_input("Nearby Hospitals", min_value=0)
    input_data["Amenities_count"] = st.number_input("Amenities Count", min_value=0)
    input_data["School_Density"] = st.number_input("School Density", min_value=0)
    input_data["Hospital_Density"] = st.number_input("Hospital Density", min_value=0)

# Categorical Inputs (Dropdowns)
st.subheader("📍 Location & Property Features")

for col, options in categorical_cols.items():
    input_data[col] = st.selectbox(col, options)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ---------------------------------------------------------
# Predict Button
# ---------------------------------------------------------
if st.button("🔍 Predict"):
    investment_pred = classifier.predict(input_df)[0]
    price_pred = regressor.predict(input_df)[0]

    st.subheader("📌 Prediction Results")

    if investment_pred == 1:
        st.success("✅ This is a GOOD Investment!")
    else:
        st.error("❌ This is NOT a good investment.")

    st.info(f"💰 **Projected Price After 5 Years:** ₹ {round(price_pred, 2)} Lakhs")
