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
water_usage = st.sidebar.slider("Water Usage (L/person/day)", 10, 200, 50)
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

st.subheader("💡 Recommendations")

recommendations = []

if rainfall < 600:
    recommendations.append("🌧️ Increase rainwater harvesting systems")

if groundwater < 50:
    recommendations.append("💧 Improve groundwater recharge (recharge pits, wells)")

if water_usage > 120:
    recommendations.append("🚰 Reduce daily water usage and promote conservation")

if population > 1000000:
    recommendations.append("👥 High population detected — future water demand may increase")

# 🔴 High Scarcity
if prediction > 70:
    recommendations.append("🚨 High scarcity risk! Implement strict water management policies")

# 🟡 Medium Scarcity
elif prediction >= 40:
    recommendations.append("⚠️ Moderate scarcity — adopt water-saving measures and monitor usage")

# 🟢 Low Scarcity
if prediction < 40 and not recommendations:
    recommendations.append("✅ Water conditions are stable. Maintain current usage levels")

for rec in recommendations:
    st.write(rec)

# Show data
st.subheader("Dataset Preview")
st.dataframe(data)

# Graph
st.subheader("Rainfall vs Scarcity")
st.line_chart(data[["Rainfall", "Scarcity_Index"]])