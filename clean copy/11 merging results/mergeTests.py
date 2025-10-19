import pandas as pd

optidisc_df = pd.read_csv("optidisc_ready.csv")
macular_df = pd.read_csv("glaucoma_macular_cube.csv")

joined_rows = []
missing_log = []
print("Starting join process...")

for idx, mac_row in macular_df.iterrows():
    instance_uid = mac_row.get("Study Instance UID")
    laterality = mac_row.get("Laterality")
    match = optidisc_df[
        (optidisc_df["Study Instance UID"] == instance_uid) &
        (optidisc_df["Laterality"] == laterality)
    ]
    if not match.empty:
        # If multiple matches, take the first
        joined_row = {**mac_row.to_dict(), **match.iloc[0].to_dict()}
        joined_rows.append(joined_row)
    else:
        missing_log.append((instance_uid, laterality))

if missing_log:
    print("Missing matches num of rows: ", len(missing_log))

joined_df = pd.DataFrame(joined_rows)
missing_log_df = pd.DataFrame(missing_log)
joined_df.to_csv("macular_optidisc_joined.csv", index=False)
missing_log_df.to_csv("macular_optidisc_missing.csv", index=False)

