import streamlit as st
import requests

st.set_page_config(page_title="Disease Prediction System", page_icon="")

st.title(" Disease Prediction System")
st.write("Select your symptoms below and get a predicted diagnosis.")

symptom_options = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "watering_from_eyes", "stomach_pain", "acidity",
    "vomiting", "cough", "chest_pain", "yellowish_skin", "nausea",
    "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "fatigue",
    "high_fever", "headache", "dark_urine", "muscle_pain", "joint_pain",
    "breathlessness", "sweating", "dehydration", "diarrhoea", "constipation"
]

selected_symptoms = st.multiselect(
    "Select your symptoms:",
    options=symptom_options
)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"symptoms": selected_symptoms}
            )
            result = response.json()
            st.success(f"Predicted Disease: **{result['predicted_disease']}**")
            st.info(f"Confidence: {result['confidence']*100:.1f}%")
        except Exception as e:
            st.error(f"Could not connect to the API. Make sure it's running. Error: {e}")
