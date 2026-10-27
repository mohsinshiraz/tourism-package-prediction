import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the predicts whether a customer will purchase the newly 
introduced Wellness Tourism Package before contacting them

""")

contact_type = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Female", "Male"])
product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

age = st.number_input("Age", min_value=18.0, max_value=90.0, value=30.0, step=1.0)
income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=20000.0, step=500.0)
pitch_duration = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=120.0, value=10.0)
trips = st.number_input("Number of Trips", min_value=0.0, max_value=20.0, value=2.0)
followups = st.number_input("Number of Followups", min_value=0.0, max_value=6.0, value=3.0)
children = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=1.0)
persons = st.number_input("Total Persons Visiting", min_value=1.0, max_value=10.0, value=2.0)




input_data = pd.DataFrame([{
    "Contact Type": contact_type,
    "Occupation": occupation,
    "Gender": gender,
    "Product": product,
    "Marital Status": marital,
    "Designation": designation,
    "Tier of the City" : city_tier,
    "Star of the Property": property_star,
    "Age": age,
    "Monthly Income": income,
    "Duration Of Pitch": pitch_duration,
    "Number Of Trips": trips,
    "Number Of Followups": followups,
    "Number Of Children Visiting": children,
    "Number Of Person Visiting": persons,
}])

if st.button("Predict Purchase Probability"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase Tourism Package" if prediction == 1 else "Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
