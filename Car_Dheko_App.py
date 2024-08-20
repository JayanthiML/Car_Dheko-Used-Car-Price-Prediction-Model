import streamlit as st
import pandas as pd

# 1. Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('Car_Details.csv')
    return data

data = load_data()

# 2. Sidebar - Filtering Options
st.sidebar.title("Car Filter Options")

# Filter by Brand
brands = data['Brand'].unique()
selected_brand = st.sidebar.multiselect("Select Brand", brands, default=brands)

# Filter by Model
models = data['Model'].unique()
selected_model = st.sidebar.multiselect("Select Model", models, default=models)

# Filter by Price Range
price_min = int(data['Price'].min())
price_max = int(data['Price'].max())
selected_price = st.sidebar.slider("Select Price Range", price_min, price_max, (price_min, price_max))

# Filter by Fuel Type
fuel_types = data['Fuel Type'].unique()
selected_fuel = st.sidebar.multiselect("Select Fuel Type", fuel_types, default=fuel_types)

# Filter by Transmission
transmission_types = data['Transmission'].unique()
selected_transmission = st.sidebar.multiselect("Select Transmission Type", transmission_types, default=transmission_types)

# 3. Filter the Data Based on Selections
filtered_data = data[
    (data['Brand'].isin(selected_brand)) &
    (data['Model'].isin(selected_model)) &
    (data['Price'].between(selected_price[0], selected_price[1])) &
    (data['Fuel Type'].isin(selected_fuel)) &
    (data['Transmission'].isin(selected_transmission))
]

# 4. Main Page - Display Results
st.title("CarDekho - Find Your Perfect Car")
st.write(f"Showing {len(filtered_data)} cars based on your filters.")

st.dataframe(filtered_data)

# Optional: Display details of a selected car
if st.checkbox("Show car details"):
    selected_car = st.selectbox("Select a car to view details", filtered_data['Model'])
    car_details = filtered_data[filtered_data['Model'] == selected_car].iloc[0]
    st.write(f"### {car_details['Brand']} {car_details['Model']}")
    st.write(f"**Price:** {car_details['Price']}")
    st.write(f"**Fuel Type:** {car_details['Fuel_Type']}")
    st.write(f"**Transmission:** {car_details['Transmission']}")
    st.write(f"**KMs Driven:** {car_details['KMs_Driven']}")
    st.write(f"**Year:** {car_details['Year']}")












# cd D:\VS_Code\MDE92_Projects\3_CarDheko\Files; streamlit run Car_Dheko_App.py
