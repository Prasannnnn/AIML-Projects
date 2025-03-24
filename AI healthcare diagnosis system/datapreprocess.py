import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("Dataset---Disease-Prediction-Using--Machine-Learning.csv")

# Encode the target column (prognosis)
label_encoder = LabelEncoder()
df["prognosis"] = label_encoder.fit_transform(df["prognosis"])

# Save the label encoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save the preprocessed dataset
df.to_csv("preprocessed_disease_dataset.csv", index=False)

print("✅ Data Preprocessing Complete! Saved as 'preprocessed_disease_dataset.csv'")
