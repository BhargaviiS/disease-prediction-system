from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Disease Prediction API")

model = joblib.load("model/disease_model.pkl")
mlb = joblib.load("model/symptom_binarizer.pkl")

class SymptomInput(BaseModel):
    symptoms: list[str]

@app.get("/")
def root():
    return {"message": "Disease Prediction API is running"}

@app.post("/predict")
def predict_disease(input_data: SymptomInput):
    cleaned_symptoms = [s.strip().lower() for s in input_data.symptoms]
    input_vector = mlb.transform([cleaned_symptoms])
    prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    confidence = max(probabilities)
    return {
        "predicted_disease": prediction,
        "confidence": round(float(confidence), 3)
    }
