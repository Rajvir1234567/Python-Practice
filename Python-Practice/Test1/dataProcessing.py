import pandas as pd
df = pd.read_csv("uncleaned_data.csv")
df["signup_date"] = pd.to_datetime(df["signup_date"], errors='coerce')
earliest_signup = df["signup_date"].min()
latest_signup = df["signup_date"].max()
print(f"Earliest signup date: {earliest_signup}")
print(f"Latest signup date: {latest_signup}")
df["signup_year"] = df["signup_date"].dt.year
users_per_year = df["signup_year"].value_counts().sort_index()
print("\nUsers who signed up each year:")
print(users_per_year)