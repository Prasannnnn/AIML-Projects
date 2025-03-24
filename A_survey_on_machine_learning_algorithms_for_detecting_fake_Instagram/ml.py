import pandas as pd

# Load dataset
df = pd.read_csv("instagram_data.csv")

# Feature Engineering
df["followers_ratio"] = df["followers"] / (df["following"] + 1)  # Avoid division by zero
df["posts_per_follower"] = df["posts"] / (df["followers"] + 1)

# Label accounts as Fake (1) or Real (0) [Example threshold-based labeling]
df["fake"] = df["followers"].apply(lambda x: 1 if x < 50 else 0)

# Select relevant features
X = df[["followers", "following", "posts", "bio_length", "followers_ratio", "posts_per_follower"]]
y = df["fake"]

print("Data preprocessing complete!")
