import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set wide layout
st.set_page_config(layout="wide")

# --- Load ML artifacts safely with caching ---
@st.cache_resource
def load_artifacts():
    try:
        predict_model = joblib.load('xgb_model.pkl')
        encoder = joblib.load('encoder.pkl')
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        df = pd.read_csv('reference_data.csv')
        return predict_model, encoder, feature_scaler, target_scaler, df
    except Exception as e:
        st.error(f"Error loading model artifacts or data: {e}")
        st.stop()

predict_model, encoder, feature_scaler, target_scaler, df = load_artifacts()

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the details of the car below:")

# Layout columns
col1, col2 = st.columns([3, 1])

details = None
if 'details' not in st.session_state:
    st.session_state.details = None

with col1:
    # --- City Selection ---
    cities = ["Select"] + sorted(df['city'].unique())
    selected_city = st.selectbox("City", cities)

    # --- Brand Selection ---
    brands = ["Select"]
    if selected_city != "Select":
        brands += sorted(df[df['city'] == selected_city]['oem'].unique())
    selected_brand = st.selectbox("Brand", brands)

    # --- Model Selection ---
    models = ["Select"]
    if selected_city != "Select" and selected_brand != "Select":
        models += sorted(df[(df['city'] == selected_city) & (df['oem'] == selected_brand)]['model'].unique())
    selected_model = st.selectbox("Model", models)

    # --- Variant Selection ---
    variants = ["Select"]
    if selected_city != "Select" and selected_brand != "Select" and selected_model != "Select":
        variants += sorted(df[(df['city'] == selected_city) & (df['oem'] == selected_brand) & (df['model'] == selected_model)]['variant_name'].unique())
    selected_variant = st.selectbox("Variant", variants)

    # --- Kilometers Driven ---
    km_range = st.select_slider("Kilometers Driven", options=["0-10k", "10k-20k", "20k-30k", "30k-40k", "40k-50k", "50k-60k", "60k-70k", "70k-80k", "80k-90k", "90k-100k", "100k-110k", "110k-120k", "120k+"], value="30k-40k")
    km_map = {
        "0-10k": 5000,
        "10k-20k": 15000,
        "20k-30k": 25000,
        "30k-40k": 35000,
        "40k-50k": 45000,
        "50k-60k": 55000,
        "60k-70k": 65000,
        "70k-80k": 75000,
        "80k-90k": 85000,
        "90k-100k": 95000,
        "100k-110k": 105000,
        "110k-120k": 115000,
        "120k+": 130000
    }
    km_driven = km_map[km_range]

    if all(sel != "Select" for sel in [selected_city, selected_brand, selected_model, selected_variant]):
        st.session_state.details = df[(df['city'] == selected_city) & (df['oem'] == selected_brand) & (df['model'] == selected_model) & (df['variant_name'] == selected_variant)].iloc[0]
        details = st.session_state.details

# Show autofilled info earlier if all selections are made
if all(sel != "Select" for sel in [selected_city, selected_brand, selected_model, selected_variant]):
    if st.session_state.details is not None:
        details = st.session_state.details
        with col2:
            with st.container(border=True):
                st.markdown(f"**Transmission:** {details['transmission']}")
                st.markdown(f"**Fuel Type:** {details['fuel_type']}")
                st.markdown(f"**Body Type:** {details['body_type']}")
                st.markdown(f"**Engine Displacement:** {details['engine_displacement']} cc")
                st.markdown(f"**Actual Kms Driven:** {details['km_driven']:,} km")
                st.markdown(f"**Insurance:** {details['insurance_validity']}")
                st.markdown(f"**Model Year:** {details['model_year']}")
                st.markdown(f"**Seating Capacity:** {details['seating_capacity']}")
                st.markdown(f"**Number of Owners:** {details['owner']}")
                st.markdown(f"**More Info:** [Car Details]({details['car_links']})")

            # --- Allow manual overrides ---
            st.subheader("🔧 Adjust Car Details")
            transmission_input = st.selectbox("Transmission", options=df['transmission'].unique(), index=list(df['transmission'].unique()).index(details['transmission']))
            fuel_input = st.selectbox("Fuel Type", options=df['fuel_type'].unique(), index=list(df['fuel_type'].unique()).index(details['fuel_type']))
            body_input = st.selectbox("Body Type", options=df['body_type'].unique(), index=list(df['body_type'].unique()).index(details['body_type']))

# Predict after button click
if all(sel != "Select" for sel in [selected_city, selected_brand, selected_model, selected_variant]):
    if st.button("Predict Price"):
        details = st.session_state.details

        input_df = pd.DataFrame([[body_input, transmission_input, details['insurance_validity'], fuel_input, selected_city, f"{selected_model} | {selected_variant}",
                                  km_driven, details['engine_displacement'], details['model_year'], details['seating_capacity'], details['owner']]],
                                columns=['body_type', 'transmission', 'insurance_validity', 'fuel_type', 'city', 'model_variant',
                                         'km_driven', 'engine_displacement', 'model_year', 'seating_capacity', 'owner'])

        encoded = encoder.transform(input_df[['body_type', 'transmission', 'insurance_validity', 'fuel_type', 'city', 'model_variant']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

        scaled_numerical = feature_scaler.transform(input_df[['km_driven', 'engine_displacement', 'model_year', 'seating_capacity', 'owner']])
        scaled_df = pd.DataFrame(scaled_numerical, columns=['km_driven', 'engine_displacement', 'model_year', 'seating_capacity', 'owner'])

        final_input = pd.concat([scaled_df, encoded_df], axis=1)
        scaled_pred = predict_model.predict(final_input)
        predicted_price = target_scaler.inverse_transform([[scaled_pred[0]]])[0][0]

        actual_price = details['price']

        st.success(f"💰 Predicted Price: ₹{int(predicted_price):,}")
        st.info(f"🏷️ Actual Price (Historical): ₹{int(actual_price):,}")

# Disclaimer
st.caption("⚠️ Note: Predictions are based on historical data and may vary in real market conditions.")
