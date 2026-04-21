import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

# -------------------------------
# Load Model & Preprocessor
# -------------------------------
model = pickle.load(open('CropYeild.pkl', 'rb'))
preprocessor = pickle.load(open('Preprocessor.pkl', 'rb'))

# -------------------------------
# Title Section
# -------------------------------
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Crop Yield Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict crop yield using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Sidebar (Info)
# -------------------------------
st.sidebar.header("📊 Model Info")
st.sidebar.write("Algorithm Used: Decision Tree")
st.sidebar.write("Accuracy: 97%")   # 🔁 replace with your actual accuracy
st.sidebar.write("Dataset: Agricultural Data")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("🔍 Enter Input Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("📅 Year", min_value=1990, max_value=2050, step=1)
    rainfall = st.number_input("🌧️ Rainfall (mm/year)")
    pesticides = st.number_input("🧪 Pesticides (tonnes)")

with col2:
    temperature = st.number_input("🌡️ Temperature (°C)")
    area = st.text_input("📍 Area (State)")
    item = st.text_input("🌱 Crop")

st.markdown("---")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict Yield", use_container_width=True):

    if area == "" or item == "":
        st.warning("⚠️ Please fill all fields properly")
    else:
        try:
            # Create DataFrame
            input_df = pd.DataFrame({
                'Year': [int(year)],
                'average_rain_fall_mm_per_year': [float(rainfall)],
                'pesticides_tonnes': [float(pesticides)],
                'avg_temp': [float(temperature)],
                'Area': [str(area)],
                'Item': [str(item)]
            })

            st.write(input_df)

            # Preprocess
            transformed = preprocessor.transform(input_df)

            # Predict
            prediction = model.predict(transformed)

            # Display Output
            st.success(f"🌾 Predicted Crop Yield: {round(prediction[0], 2)}")

        except Exception as e:
           st.error(f"❌ Error: {e}")
