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

age = st.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
preferred_property_star = st.slider("Preferred Property Star", 1, 5, 3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=20000.0)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
duration_of_pitch = st.number_input("Duration of Pitch", min_value=0.0, value=15.0)




input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": num_followups,
        "DurationOfPitch": duration_of_pitch,
}])

if st.button("Predict"):
  try:
    # Line 56 equivalent
    prediction = model.predict(input_data)[0]
    if prediction == 1:
      st.success(
          "The customer is likely to purchase the Tourism Package!"
      )
    else:
      st.info("The customer is unlikely to purchase the package.")
  except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
