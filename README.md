
---

## 🧠 Machine Learning Pipeline

### ✅ Data Collection
- Datasets from multiple cities: Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata.
- Merged and labeled with city names.
  
### ✅ Preprocessing
- Categorical Features: `body_type`, `transmission`, `insurance_validity`, `fuel_type`, `city`, `model_variant`
  - Encoded using **OneHotEncoder**.
- Numerical Features: `km_driven`, `engine_displacement`, `model_year`, `seating_capacity`, `owner`
  - Scaled using **StandardScaler**.
- Target variable: `price` is scaled using **StandardScaler** to improve regression model performance.

### ✅ Model Training
- Algorithm used: **XGBoost Regressor**
- Evaluation metrics: MAE, RMSE, R² score
- Final model saved using `joblib`.

### ✅ Artifacts Saved
- `xgb_model.pkl`: Trained model
- `encoder.pkl`: Fitted OneHotEncoder
- `feature_scaler.pkl`: Scaler for numerical inputs
- `target_scaler.pkl`: Scaler for output price

---

## 💻 Streamlit Web App (`app.py`)

### 🎯 Features
- Interactive dropdowns for **City**, **Brand**, **Model**, and **Variant**.
- Slider for **Kilometers Driven** (mapped internally to numeric values).
- Auto-filled information from the selected variant: transmission, fuel type, engine, seating, etc.
- Predicts car price using the trained model.
- Shows:
  - 💰 **Predicted Price**
  - 🏷️ **Actual Price** (from historical data for comparison)

### 🛠️ Backend Logic
1. Based on user selection, reference data is filtered to get actual car details.
2. Selected variant + inputs are transformed and scaled.
3. Model predicts price.
4. Predicted price is inverse-transformed using `target_scaler`.
5. Both predicted and historical prices are shown.

---

## 📝 How to Run

### 🔧 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
