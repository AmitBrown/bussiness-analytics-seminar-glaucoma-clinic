import pandas as pd

# Load the Excel file
optidisc_df = pd.read_excel("optidisc_ready.xlsx")

# Convert Gender: FEMALE -> 0, MALE -> 1
if 'Gender' in optidisc_df.columns:
    optidisc_df['Gender'] = optidisc_df['Gender'].map({'FEMALE': 0, 'MALE': 1})

# Convert Laterality to numeric encoding
if 'Laterality' in optidisc_df.columns:
    optidisc_df['Laterality'] = optidisc_df['Laterality'].map({'LEFT': 0, 'RIGHT': 1})

# Convert Date of Birth to just the year
if 'Date of Birth' in optidisc_df.columns:
    optidisc_df['Date of Birth'] = pd.to_datetime(optidisc_df['Date of Birth'], errors='coerce').dt.year

# Show summary: shape, columns, dtypes, and describe
print(f"Shape: {optidisc_df.shape}")
print(f"Columns: {optidisc_df.columns.tolist()}")
print("\nData types:\n", optidisc_df.dtypes)
print("\nSummary statistics:\n", optidisc_df.describe(include='all'))

optidisc_df.to_csv("optidisc_ready.csv", index=False)