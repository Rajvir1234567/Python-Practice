import pandas as pd
df = pd.read_csv("uncleaned_data.csv")
df["full_name"] = df["full_name"].fillna("Unknown")
df["full_name"] = df["full_name"].str.strip()
df["full_name"] = df["full_name"].str.replace(r'\s+', ' ', regex=True)
df["full_name"] = df["full_name"].str.title()
print("Updated Dataframe:")
print(df)