import pandas as pd
df = pd.read_csv("uncleaned_data.csv")
df["address"] = df["address"].fillna("Unknown")
df["state"] = df["address"].str.split(",").str[-1]
df["state"] = df["state"].str.strip().str.title()
state_counts = df["state"].value_counts()
print("Number of customers from each state :")
print(state_counts)
top_five_states = state_counts.head(5)
print("\nTop 5 states with the most customers:")
print(top_five_states)