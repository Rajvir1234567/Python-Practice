import pandas as pd
df = pd.read_csv("uncleaned_data.csv")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_report = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
print("Missing Values Report:")
print(missing_report)
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
for column in numeric_columns:
    df[column]= df[column].fillna(df[column].median())
text_columns=df.select_dtypes(include=["object"]).columns
for column in text_columns:
    if not df[column].mode().empty:
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna("Unknown")
print("\nMissing values after replacement:")
print(df.isnull().sum())