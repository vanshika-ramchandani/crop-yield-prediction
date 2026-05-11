## 🚀 Live Demo


# 🌾 CropIQ — Crop Yield Prediction

> ML-powered crop yield prediction with AI-generated agricultural insights.

CropIQ predicts crop yield based on environmental and farming conditions, then uses an LLM to explain what the prediction means and how a farmer or policymaker could improve yield.

---

## ✨ Features

- **ML Yield Prediction** — Decision Tree model trained on FAO data (1990–2013) across 101 countries and 10 crops
- **Dropdown Inputs** — Select country and crop from real dataset values, no typos or invalid inputs
- **AI Agricultural Insight** — After prediction, Groq LLaMA 3.3 explains the result in plain English: what influenced it, how it compares, and what to do next
- **Clean Dashboard UI** — Dark agricultural theme with clear result display

---

## 🧠 How It Works

```
User selects country, crop, year, rainfall, temperature, pesticides
        ↓
Preprocessor applies OneHotEncoding + StandardScaler
        ↓
Decision Tree model predicts yield (hg/ha)
        ↓
Groq LLaMA 3.3 generates a 3-4 sentence agricultural insight
        ↓
Result + insight displayed on dashboard
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| ML Model | Decision Tree (scikit-learn) |
| Preprocessing | ColumnTransformer (OneHotEncoder + StandardScaler) |
| LLM Insights | Groq (LLaMA 3.3 70B) |
| Dataset | FAO Agricultural Data 1990–2013 |

---

## 📊 Model Performance

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | 29920.7 | ~74% |
| Lasso | 29907.6 | ~74% |
| Ridge | 29875.0 | ~74% |
| KNN Regressor | 4868.6 | ~98% |
| **Decision Tree** | 4175.3 | **~97%** |

Decision Tree was selected as the best performing model.

---

## 🌍 Dataset

- **Source:** FAO (Food and Agriculture Organization of the United Nations)
- **Period:** 1990–2013
- **Countries:** 101
- **Crops:** Cassava, Maize, Plantains, Potatoes, Rice, Sorghum, Soybeans, Sweet Potatoes, Wheat, Yams
- **Features:** Year, Rainfall (mm/year), Pesticides (tonnes), Average Temperature (°C), Country, Crop

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/vanshika-ramchandani/crop-yield-prediction
cd crop-yield-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

- [ ] Integrate real-time weather API for auto-filled rainfall and temperature
- [ ] Add satellite NDVI vegetation index as a feature
- [ ] Extend dataset beyond 2013 with recent FAO data
- [ ] Add country-level yield trend visualization

---

## 👩‍💻 Author

**Vanshika Ramchandani**
