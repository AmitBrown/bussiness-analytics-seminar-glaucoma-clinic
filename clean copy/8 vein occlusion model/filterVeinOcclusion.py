import pandas as pd

df = pd.read_csv("unlabeled_flagged_vo.csv")
patient_ids = set(df["Patient ID"].unique())
print("Unique Patient IDs:", len(patient_ids))

unlabeled_filtered = pd.read_csv("unlabeled_filtered_7.csv")
filtered = unlabeled_filtered[~unlabeled_filtered["Patient ID"].isin(patient_ids)]
filtered.to_csv("unlabeled_filtered_8.csv", index=False)

print("Number of rows:", len(df))

