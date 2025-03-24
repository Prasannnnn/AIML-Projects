import streamlit as st
import feedparser
import nltk
from nltk.tokenize import sent_tokenize
import re  # For removing HTML tags

# Download necessary resources
nltk.download("punkt")

# Function to clean HTML tags from text
def clean_html(raw_html):
    clean_text = re.sub(r'<.*?>', '', raw_html)  # Remove HTML tags
    return clean_text.strip()

# Function to summarize text using NLTK
def nltk_summarize(text, num_sentences=2):
    text = clean_html(text)  # Ensure summary is clean
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # If the text is already short, return as is
    return " ".join(sentences[:num_sentences])

# Function to fetch news from Google RSS
def get_news(rss_url):
    feed = feedparser.parse(rss_url)
    articles = []

    for entry in feed.entries[:5]:
        articles.append({
            "title": clean_html(entry.title),  # Clean title
            "summary": clean_html(entry.summary),  # Clean summary
            "link": entry.link
        })
    
    return articles

# Streamlit UI
st.set_page_config(page_title="AI News Aggregator", page_icon="📰", layout="wide")
st.title("📰 AI News Aggregator")

# Category selection
categories = {
    "Technology": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "Sports": "https://news.google.com/rss/search?q=sports&hl=en-IN&gl=IN&ceid=IN:en",
    "Health": "https://news.google.com/rss/search?q=health&hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/search?q=business&hl=en-IN&gl=IN&ceid=IN:en"
}

category = st.selectbox("📌 Choose a news category:", list(categories.keys()))

# Fetch and display news
if category:
    st.subheader(f"📰 Latest {category} News")
    news_articles = get_news(categories[category])

    if st.button("🔥 Five in Five (Summarize Top 5)"):
        st.subheader("📌 AI Summarized News (5 in 5)")
        for i, article in enumerate(news_articles):
            ai_summary = nltk_summarize(article["summary"], num_sentences=2)
            st.write(f"**{i+1}. {article['title']}**")
            st.success(f"**AI Summary:** {ai_summary}")  # Only displays clean text
            st.write("---")

    for i, article in enumerate(news_articles):
        with st.expander(f"🔹 {article['title']}"):
            if st.button(f"Summarize {i+1}"):
                ai_summary = nltk_summarize(article["summary"], num_sentences=2)
                st.success(f"**AI Summary:** {ai_summary}")  # Displays only plain text
