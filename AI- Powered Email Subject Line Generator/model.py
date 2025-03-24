import nltk
import pandas as pd
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (Replace with actual dataset)
data = pd.read_csv("project_requests.csv")
df = pd.DataFrame(data)

# Ensure necessary columns exist
if "Prompt" not in df.columns or "Content" not in df.columns:
    raise ValueError("CSV file must have 'Prompt' and 'Content' columns")

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
        return ' '.join(tokens)
    return ""

# Apply preprocessing
df["Prompt"] = df["Prompt"].apply(preprocess_text)  # Email Subject (Target)
df["Content"] = df["Content"].apply(preprocess_text)  # Email Body (Input)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["Content"], df["Prompt"], test_size=0.2, random_state=42)

# Create TF-IDF Vectorizer and Naïve Bayes classifier
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model
predictions = model.predict(X_test_tfidf)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Save the trained model
with open("email_subject_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the trained vectorizer
with open("tfidf_subject_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Email Subject Line Model and Vectorizer saved successfully!")
