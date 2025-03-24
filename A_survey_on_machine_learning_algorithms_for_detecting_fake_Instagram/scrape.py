import instaloader
import pandas as pd

# Initialize Instaloader
loader = instaloader.Instaloader()

# List of usernames to scrape
usernames = ["prasan_4362", "mahi7781", "virat.kohli","rohitsharma45","chennaiipl","raagavarshini91"]  # Replace with actual usernames

data = []

for username in usernames:
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        data.append({
            "username": username,
            "followers": profile.followers,
            "following": profile.followees,
            "posts": profile.mediacount,
            "is_verified": int(profile.is_verified),
            "bio_length": len(profile.biography),
        })
    except Exception as e:
        print(f"Error fetching {username}: {e}")

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("instagram_data.csv", index=False)

print("Data collection complete!")
