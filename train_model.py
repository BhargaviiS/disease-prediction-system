import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("data/dataset.csv")
df = df.drop_duplicates()
print("Rows after removing duplicates:", len(df))

# Identify all the symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]

# Clean whitespace in symptom values and combine them into one list per row
def get_symptom_list(row):
    symptoms = []
    for col in symptom_cols:
        val = row[col]
        if pd.notna(val):
            symptoms.append(val.strip().lower())
    return symptoms

df["symptom_list"] = df.apply(get_symptom_list, axis=1)

# Convert the symptom lists into a binary matrix (one column per unique symptom)
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptom_list"])
y = df["Disease"].str.strip()

print("Number of unique symptoms:", len(mlb.classes_))
print("Number of diseases:", y.nunique())
print("Shape of feature matrix:", X.shape)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2%}")

# Save the trained model and the symptom binarizer for later use
joblib.dump(model, "model/disease_model.pkl")
joblib.dump(mlb, "model/symptom_binarizer.pkl")
print("Model and binarizer saved to /model")