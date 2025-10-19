import pandas as pd

df = pd.read_csv("unlabeled_flagged_amd.csv")
patient_ids = set(df["Patient ID"].unique())
print("Unique Patient IDs:", len(patient_ids))

unlabeled_filtered = pd.read_csv("unlabeled_filtered_cme_neuro_retina.csv")
filtered = unlabeled_filtered[~unlabeled_filtered["Patient ID"].isin(patient_ids)]
filtered.to_csv("unlabeled_filtered_cme_neuro_retina_amd.csv", index=False)

print("Number of rows:", len(df))

