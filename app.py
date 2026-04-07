import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv("data.csv")

# Features and target
X = data.drop("Scarcity_Index", axis=1)
y = data["Scarcity_Index"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# UI
st.title("💧 Water Scarcity Prediction Dashboard")

st.sidebar.header("Input Parameters")

rainfall = st.sidebar.slider("Rainfall (mm)", 200, 1500, 800)
temperature = st.sidebar.slider("Temperature (°C)", 20, 40, 30)
population = st.sidebar.slider("Population", 100000, 1500000, 800000)
water_usage = st.sidebar.slider("Water Usage (L/person/day)", 100, 400, 200)
groundwater = st.sidebar.slider("Groundwater Level (%)", 20, 100, 60)

# Prediction
input_data = pd.DataFrame({
    "Rainfall": [rainfall],
    "Temperature": [temperature],
    "Population": [population],
    "Water_Usage": [water_usage],
    "Groundwater_Level": [groundwater]
})

prediction = model.predict(input_data)[0]

# Convert to label
if prediction < 40:
    level = "🟢 Low"
elif prediction < 70:
    level = "🟡 Medium"
else:
    level = "🔴 High"

# Output
st.subheader("Prediction Result")
st.metric("Scarcity Index", f"{prediction:.2f}")
st.write(f"### Scarcity Level: {level}")

# Show data
st.subheader("Dataset Preview")
st.dataframe(data)

# Graph
st.subheader("Rainfall vs Scarcity")
st.line_chart(data[["Rainfall", "Scarcity_Index"]])