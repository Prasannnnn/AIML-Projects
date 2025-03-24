from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
# Load preprocessed dataset
df = pd.read_csv("preprocessed_disease_dataset.csv")

# Split dataset into features (X) and target (y)
X = df.drop("prognosis", axis=1)  # All symptoms
y = df["prognosis"]  # Disease labels

# Split into training & testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("disease_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎯 Model saved as 'disease_prediction_model.pkl'")

import numpy as np


# Load trained model & label encoder
with open("disease_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define sample input (modify for different symptoms)
user_symptoms = np.zeros(len(X.columns))  # Start with all symptoms as 0
user_symptoms[X.columns.get_loc("itching")] = 1  # Example: User has itching
user_symptoms[X.columns.get_loc("skin_rash")] = 1  # Example: User has skin rash

# Predict disease
predicted_index = model.predict([user_symptoms])[0]
predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

print(f"🎯 Predicted Disease: {predicted_disease}")
