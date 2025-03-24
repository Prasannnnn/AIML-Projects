import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("instagram_data.csv")

# Feature Engineering: Creating meaningful features
df["followers_ratio"] = df["followers"] / (df["following"] + 1)  # Avoid division by zero
df["posts_per_follower"] = df["posts"] / (df["followers"] + 1)

# **Define X (features) and y (target label)**
X = df[["followers", "following", "posts", "bio_length", "followers_ratio", "posts_per_follower"]]
y = df["followers"].apply(lambda x: 1 if x < 50 else 0)  # Labeling: Fake if < 50 followers

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model for later use
joblib.dump(model, "fake_instagram_model.pkl")

print("✅ Model training complete! Saved as 'fake_instagram_model.pkl'.")
