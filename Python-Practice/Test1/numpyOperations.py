import pandas as pd
import numpy as np
df = pd.read_csv("uncleaned_data.csv")
df["id"] = pd.to_numeric(df["id"], errors='coerce')
clean_ids = df["id"].dropna()
id_array = clean_ids.to_numpy()
print("NumPy ID array:")
print(id_array)
print("\nMinimum ID:", np.min(id_array))
print("Maximum ID:", np.max(id_array))
print("Mean ID:", np.mean(id_array))
print("Median ID:", np.median(id_array))
print("Standard deviation:", np.std(id_array))
even_ids = id_array[id_array%2==0]
print("\nEven IDs:")
print(even_ids)