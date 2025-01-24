import pandas as pd

# Create an empty DataFrame with no columns but with specified row index names
df = pd.DataFrame(index=["elon", "bill", "robert"])

# Save it to a CSV file, keeping the index
df.to_csv('attend.csv')

print("CSV file with specified row index names created successfully.")
