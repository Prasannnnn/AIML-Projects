import streamlit as st
import feedparser
import random

# Function to fetch news from Google RSS
def get_news(rss_url):
    feed = feedparser.parse(rss_url)
    articles = [{"title": entry.title, "link": entry.link, "summary": entry.summary} for entry in feed.entries[:5]]
    return articles

# Streamlit App UI
st.title("📰 AI News Aggregator")

# Category selection
categories = {
    "Technology": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "Sports": "https://news.google.com/rss/search?q=sports&hl=en-IN&gl=IN&ceid=IN:en",
    "Health": "https://news.google.com/rss/search?q=health&hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/search?q=business&hl=en-IN&gl=IN&ceid=IN:en"
}

category = st.selectbox("Choose a news category:", list(categories.keys()))

# Fetch news based on the selected category
if category:
    st.subheader(f"Latest {category} News")
    news_articles = get_news(categories[category])

    for i, article in enumerate(news_articles):
        st.write(f"**{i+1}. {article['title']}**")
        st.write(f"[Read more]({article['link']})")

        # AI-like Summary (Truncated for Now)
        if st.button(f"Summarize {i+1}"):
            summary = article["summary"][:100] + "..."
            st.success(f"**Summary:** {summary}")
