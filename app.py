
import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details to check whether they would have survived.")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, value=0)
fare = st.slider("Fare Paid", 0.0, 600.0, 50.0)

# Convert categorical input
sex_encoded = 1 if sex == 'male' else 0

# Create input DataFrame
input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare]
})

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ This passenger would have survived!")
    else:
        st.error("❌ This passenger would not have survived.")
