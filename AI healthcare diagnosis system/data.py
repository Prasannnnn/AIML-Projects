import pandas as pd

# Sample DataFrame
data = pd.read_csv("Dataset---Disease-Prediction-Using--Machine-Learning.csv")

df = pd.DataFrame(data)

# Get unique value counts for each column
for col in df.columns:
    print(f"Column: {col}")
    print(df[col].value_counts())  # Counts of each unique value
    print("\n")
