import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropIQ",
    page_icon="🌾",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        min-height: 100vh;
    }

    h1 {
        font-family: 'Merriweather', serif !important;
        color: #a8d5a2 !important;
        letter-spacing: -0.5px;
    }

    .result-card {
        background: linear-gradient(135deg, #1a3a2a, #1e4035);
        border: 1px solid #2d6a4f;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    }

    .result-number {
        font-size: 3rem;
        font-weight: 700;
        color: #74c69d;
        font-family: 'Merriweather', serif;
    }

    .result-unit {
        font-size: 1rem;
        color: #95d5b2;
        margin-top: 4px;
    }

    .insight-card {
        background: #0f2318;
        border: 1px solid #2d6a4f40;
        border-left: 4px solid #52b788;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
        line-height: 1.8;
        color: #b7e4c7;
        font-size: 0.95rem;
    }

    .stat-pill {
        display: inline-block;
        background: #1b4332;
        border: 1px solid #2d6a4f;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: #95d5b2;
        margin: 4px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #95d5b2 !important;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.03em;
    }

    .section-label {
        color: #52b788;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
COUNTRIES = [
    'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia',
    'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus',
    'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi',
    'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia',
    'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt',
    'El Salvador', 'Eritrea', 'Estonia', 'Finland', 'France', 'Germany',
    'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
    'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy', 'Jamaica',
    'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon', 'Lesotho', 'Libya',
    'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania',
    'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia',
    'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
    'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 'Qatar',
    'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia', 'South Africa',
    'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland',
    'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine',
    'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe'
]

CROPS = [
    'Cassava', 'Maize', 'Plantains and others', 'Potatoes', 'Rice, paddy',
    'Sorghum', 'Soybeans', 'Sweet potatoes', 'Wheat', 'Yams'
]

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model       = pickle.load(open('CropYeild.pkl', 'rb'))
    preprocessor = pickle.load(open('Preprocessor.pkl', 'rb'))
    return model, preprocessor

model, preprocessor = load_model()

# ── LLM Insight ───────────────────────────────────────────────────────────────
def get_llm_insight(country, crop, year, rainfall, temperature, pesticides, predicted_yield):
    """Calls Groq to generate an agricultural insight about the prediction."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""
You are an expert agricultural analyst. A machine learning model predicted the following:

Country: {country}
Crop: {crop}
Year: {year}
Rainfall: {rainfall} mm/year
Average Temperature: {temperature}°C
Pesticides Used: {pesticides} tonnes
Predicted Yield: {round(predicted_yield, 2)} hg/ha

Write a 3-4 sentence agricultural insight that:
1. Contextualizes this yield (is it high or low for this crop/region?)
2. Explains how the given conditions (rainfall, temperature, pesticides) likely influenced this yield
3. Gives one practical recommendation for improving yield

Be factual, specific, and concise. No bullet points — write in flowing sentences.
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate insight: {e}"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;'>🌾 CropIQ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#74c69d; margin-top:-8px;'>ML-powered crop yield prediction with AI agricultural insights</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.markdown('<span class="stat-pill">Algorithm: Decision Tree</span>', unsafe_allow_html=True)
    st.markdown('<span class="stat-pill">Accuracy: 97%</span>', unsafe_allow_html=True)
    st.markdown('<span class="stat-pill">Dataset: FAO 1990–2013</span>', unsafe_allow_html=True)
    st.markdown('<span class="stat-pill">Countries: 101</span>', unsafe_allow_html=True)
    st.markdown('<span class="stat-pill">Crops: 10</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
    CropIQ predicts agricultural yield using environmental and farming data.
    After prediction, an AI analyst explains what the result means and how
    to improve it.
    """)

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📍 Location & Crop</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", options=COUNTRIES, index=COUNTRIES.index("India"))
with col2:
    crop = st.selectbox("Crop", options=CROPS, index=CROPS.index("Wheat"))

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">🌦️ Environmental Conditions</p>', unsafe_allow_html=True)

col3, col4, col5, col6 = st.columns(4)
with col3:
    year = st.number_input("Year", min_value=1990, max_value=2050, value=2010, step=1)
with col4:
    rainfall = st.number_input("Rainfall (mm/yr)", min_value=51.0, max_value=3240.0, value=1000.0, step=10.0)
with col5:
    temperature = st.number_input("Temp (°C)", min_value=1.3, max_value=50.0, value=20.0, step=0.1)
with col6:
    pesticides = st.number_input("Pesticides (t)", min_value=0.0, value=100.0, step=1.0)

if temperature > 30.65:
    st.caption("⚠️ Temperature exceeds training data range — prediction may be less accurate.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ─────────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 2])
with col_btn:
    predict_btn = st.button("🚀 Predict Yield")

st.caption("ℹ️ Model trained on 1990–2013 data. Predictions for future years are extrapolations and may vary.")

# ── Prediction & Insight ───────────────────────────────────────────────────────
if predict_btn:
    try:
        input_df = pd.DataFrame({
            'Year':                          [int(year)],
            'average_rain_fall_mm_per_year': [float(rainfall)],
            'pesticides_tonnes':             [float(pesticides)],
            'avg_temp':                      [float(temperature)],
            'Area':                          [country],
            'Item':                          [crop],
        })

        transformed  = preprocessor.transform(input_df)
        prediction   = model.predict(transformed)
        yield_value  = round(float(prediction[0]), 2)

        # Result card
        st.markdown(f"""
        <div class="result-card">
            <div style="color:#95d5b2; font-size:0.85rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">
                Predicted Yield — {crop} in {country} ({int(year)})
            </div>
            <div class="result-number">{yield_value:,.2f}</div>
            <div class="result-unit">hg/ha (hectograms per hectare)</div>
        </div>
        """, unsafe_allow_html=True)

        # LLM Insight
        with st.spinner("🌱 Generating agricultural insight..."):
            insight = get_llm_insight(
                country, crop, int(year),
                rainfall, temperature, pesticides, yield_value
            )

        st.markdown("**🤖 AI Agricultural Insight**")
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")