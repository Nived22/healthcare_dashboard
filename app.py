import streamlit as st
import pandas as pd
import plotly.express as px
from model import train_model

# Load data
df = pd.read_csv("data/diabetes.csv")

# Page title
st.title("🧠 Healthcare Analytics Dashboard")
st.markdown("This dashboard analyzes and predicts diabetes risk using ML.")

# Display data
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Plot: Age vs Outcome
st.subheader("Distribution by Age and Outcome")
fig = px.histogram(df, x="Age", color="Outcome", barmode="group")
st.plotly_chart(fig)

# Train model
model = train_model(df)

# Sidebar: Prediction input
st.sidebar.header("🔍 Predict Diabetes Risk")

glucose = st.sidebar.slider("Glucose", 0, 200, 120)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 10.0, 70.0, 25.0)
age = st.sidebar.slider("Age", 21, 80, 33)
pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
bp = st.sidebar.slider("Blood Pressure", 40, 130, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

user_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                         columns=df.columns[:-1])

prediction = model.predict(user_data)

st.sidebar.subheader("Prediction:")
st.sidebar.write("Diabetes Risk: ", "✅ Positive" if prediction[0] == 1 else "❌ Negative")
