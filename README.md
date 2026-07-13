# Disease Prediction System

An AI-powered system that predicts the likely disease based on user-input symptoms, built end-to-end: data cleaning, model training, a REST API, and a web interface.

## What it does
A user selects their symptoms from a list, and the system returns a predicted disease along with a confidence score, powered by a trained machine learning model.

## Tech Stack
- **Python** — core language
- **Pandas / NumPy** — data cleaning and preprocessing
- **Scikit-learn** — Random Forest model for multi-class disease classification
- **FastAPI** — REST API serving predictions
- **Streamlit** — web-based user interface
- **Joblib** — model persistence

## How it works
1. **Data pipeline** (`train_model.py`): Loads a symptom-disease dataset, removes duplicate rows, and converts symptoms into a binary feature matrix using `MultiLabelBinarizer`.
2. **Model**: A Random Forest classifier is trained on the cleaned data and saved to disk.
3. **API** (`api/main.py`): A FastAPI backend loads the trained model and exposes a `/predict` endpoint that accepts a list of symptoms and returns a predicted disease with a confidence score.
4. **UI** (`app/streamlit_app.py`): A Streamlit frontend lets users select symptoms from a dropdown and calls the API to display the prediction.

## Running it locally

Clone the repo and install dependencies:
```bash
git clone https://github.com/BhargaviiS/disease-prediction-system.git
cd disease-prediction-system
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn fastapi uvicorn streamlit requests
```

Train the model:
```bash
python train_model.py
```

Run the API (Terminal 1):
```bash
uvicorn api.main:app --reload
```

Run the UI (Terminal 2, with venv activated):
```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

## A note on model accuracy
This model scores 100% accuracy on the test set. That is expected, not a red flag, here: the underlying dataset is a synthetic Kaggle dataset with distinct, non-overlapping symptom-disease mappings, and the training pipeline explicitly deduplicates rows before splitting to avoid data leakage. On real-world clinical data with overlapping symptoms and noise, accuracy would be expected to be significantly lower.

## Future improvements
- Deduplicate and expand the training dataset for more realistic performance
- Add symptom severity weighting using the included `Symptom-severity.csv`
- Add disease descriptions and precautions to the UI using the included reference files
- Deploy the API and UI to a cloud platform
