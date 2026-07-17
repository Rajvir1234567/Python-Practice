import pandas as pd
df = pd.read_csv("uncleaned_data.csv")
rows_before = len(df)
duplicate_records = df[df.duplicated(keep=False)]
print("Duplicate Records:")
print(duplicate_records)
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()
rows_after = len(df)
print("\nRows before removing duplicates:", rows_before)
print("Rows after removing duplicates:", rows_after)
print("Rows removed:", rows_before - rows_after)