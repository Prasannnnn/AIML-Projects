import feedparser

# Google News RSS URL (Change 'technology' to your preferred category)
rss_url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

# Parse RSS feed
news_feed = feedparser.parse(rss_url)

# Print latest headlines
print("\n📰 Latest Google News Headlines:\n")
for idx, entry in enumerate(news_feed.entries[:5], 1):
    print(f"{idx}. {entry.title}")
    print(f"   🔗 {entry.link}\n")
