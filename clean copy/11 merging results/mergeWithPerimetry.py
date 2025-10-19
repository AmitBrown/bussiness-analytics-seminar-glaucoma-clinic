import pandas as pd
import numpy as np

# --- Load data ---
macular_optidisc_df = pd.read_csv("macular_optidisc_joined.csv")   # OCT table
perimetry_df = pd.read_csv("perimetry_cleaned.csv")                 # Perimetry table

# --- Convert date columns ---
macular_optidisc_df["OCT Scan Date"] = pd.to_datetime(macular_optidisc_df["OCT Scan Date"], errors="coerce")
perimetry_df["Perimetry Exam Date"] = pd.to_datetime(perimetry_df["Perimetry Exam Date"], errors="coerce")

# --- Function to find nearest perimetry row for one macular row ---
def find_closest_perimetry(row):
    patient_id = row["Patient ID"]
    oct_date = row["OCT Scan Date"]
    # all perimetry rows of same patient
    sub = perimetry_df[perimetry_df["Patient ID"] == patient_id]
    if sub.empty or pd.isna(oct_date):
        return None
    # compute absolute time difference
    sub = sub.assign(days_diff=(sub["Perimetry Exam Date"] - oct_date).abs())
    # find the minimum difference
    min_row = sub.loc[sub["days_diff"].idxmin()]
    if min_row["days_diff"].days <= 60:
        return min_row
    else:
        return None

# --- Iterate through macular and build matched/unmatched ---
matched_rows = []
unmatched_rows = []

print("Starting perimetry matching process...")

for idx, mac_row in macular_optidisc_df.iterrows():
    match = find_closest_perimetry(mac_row)
    if match is not None:
        combined = pd.concat([mac_row, match.add_prefix("Perimetry_")])
        matched_rows.append(combined)
    else:
        unmatched_rows.append(mac_row)

# --- Convert lists to DataFrames ---
matched_df = pd.DataFrame(matched_rows)
unmatched_df = pd.DataFrame(unmatched_rows)

# --- Save results ---
matched_df.to_csv("macular_with_matched_perimetry.csv", index=False)
unmatched_df.to_csv("macular_without_match.csv", index=False)

print(f"✅ Matched rows: {len(matched_df)}")
print(f"❌ Unmatched rows: {len(unmatched_df)}")
print("Files saved as 'macular_with_matched_perimetry.csv' and 'macular_without_match.csv'.")
