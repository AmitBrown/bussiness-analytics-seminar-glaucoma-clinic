import pandas as pd
import numpy as np

# =============================================================
# 1. Load all three datasets
# =============================================================
print("=== Loading files ===")
perimetry_df = pd.read_csv("perimetry.csv")
macular_df = pd.read_csv("macularCube.csv")
optidisc_df = pd.read_csv("optidisc.csv")

# =============================================================
# 2. Merge macularCube with optidisc by Study Instance UID + Laterality
# =============================================================
print("=== Joining macularCube with optidisc ===")
joined_rows = []
missing_log = []

for idx, mac_row in macular_df.iterrows():
    instance_uid = mac_row.get("Study Instance UID")
    laterality = mac_row.get("Laterality")
    match = optidisc_df[
        (optidisc_df["Study Instance UID"] == instance_uid) &
        (optidisc_df["Laterality"] == laterality)
    ]
    if not match.empty:
        joined_row = {**mac_row.to_dict(), **match.iloc[0].to_dict()}
        joined_rows.append(joined_row)
    else:
        missing_log.append((instance_uid, laterality))

joined_df = pd.DataFrame(joined_rows)
print(f"✅ Joined {len(joined_df)} macular rows with optidisc.")
if missing_log:
    print(f"⚠️ Missing matches: {len(missing_log)} rows (saved separately).")
    pd.DataFrame(missing_log, columns=["Study Instance UID", "Laterality"]).to_csv(
        "missing_optidisc_matches.csv", index=False
    )

# =============================================================
# 3. Convert date columns to datetime for matching perimetry
# =============================================================
print("=== Converting date columns ===")
joined_df["OCT Scan Date"] = pd.to_datetime(joined_df["OCT Scan Date"], errors="coerce")
perimetry_df["Perimetry Exam Date"] = pd.to_datetime(perimetry_df["Perimetry Exam Date"], errors="coerce")

# =============================================================
# 4. Match perimetry rows within ±60 days by Patient ID
# =============================================================
print("=== Matching perimetry data ===")

def find_closest_perimetry(row):
    patient_id = row["Patient ID"]
    oct_date = row["OCT Scan Date"]
    sub = perimetry_df[perimetry_df["Patient ID"] == patient_id]
    if sub.empty or pd.isna(oct_date):
        return None
    sub = sub.assign(days_diff=(sub["Perimetry Exam Date"] - oct_date).abs())
    min_row = sub.loc[sub["days_diff"].idxmin()]
    if min_row["days_diff"].days <= 60:
        return min_row
    return None

matched_rows = []
unmatched_rows = []

for idx, mac_row in joined_df.iterrows():
    match = find_closest_perimetry(mac_row)
    if match is not None:
        combined = pd.concat([mac_row, match.add_prefix("Perimetry_")])
        matched_rows.append(combined)
    else:
        unmatched_rows.append(mac_row)

matched_df = pd.DataFrame(matched_rows)
unmatched_df = pd.DataFrame(unmatched_rows)

# =============================================================
# 5. Save results
# =============================================================
matched_df.to_csv("final_matched.csv", index=False)
unmatched_df.to_csv("final_unmatched.csv", index=False)

print("\n=== Merge Summary ===")
print(f"✅ Matched rows (with perimetry): {len(matched_df)}")
print(f"❌ Unmatched rows: {len(unmatched_df)}")
print("📁 Saved as 'final_matched.csv' and 'final_unmatched.csv'.")
print("📁 Missing optidisc matches saved as 'missing_optidisc_matches.csv' (if any).")
