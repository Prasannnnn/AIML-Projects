import streamlit as st
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
with open("email_subject_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_subject_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Streamlit App UI
st.title("📩 AI-Powered Email Subject Line Generator")
st.write("Enter an email body, and the AI will generate a subject line for you!")

# Input text box for email body
email_body = st.text_area("Enter Email Body:", height=200)

# Predict subject line on button click
if st.button("Generate Subject Line"):
    if email_body.strip():
        # Preprocess input
        processed_text = preprocess_text(email_body)
        
        # Transform input using TF-IDF Vectorizer
        input_tfidf = vectorizer.transform([processed_text])
        
        # Predict subject line
        predicted_subject = model.predict(input_tfidf)[0]
        
        # Display the generated subject line
        st.success(f"📌 Suggested Subject Line: **{predicted_subject}**")
    else:
        st.warning("⚠️ Please enter an email body to generate a subject line.")

# Footer
st.markdown("---")
st.markdown("🔹 **Built with Streamlit, Naïve Bayes & TF-IDF**")

