import pandas as pd

# Load data
untagged = pd.read_csv("untagged.csv")
scores = pd.read_csv("unlabeled_scores.csv")
labeled = pd.read_csv("labeled_neuro.csv")

# Get Patient IDs flagged as neuro
flagged_ids = set(scores.loc[scores["flag_neuro_like"] == 1, "Patient ID"])
labeled_ids = set(labeled["Patient ID"])

# Filter out flagged neuro rows
not_flagged = scores.loc[scores["flag_neuro_like"] != 1, "Patient ID"]
not_flagged_ids = set(not_flagged)

# Remove rows where Patient ID is in flagged or labeled
exclude_ids = flagged_ids | labeled_ids
filtered = untagged[~untagged["Patient ID"].isin(exclude_ids)]

# Combine patient ids from labeled_neuro and unlabeled_flagged_neurolike
ids_to_move = pd.concat([
    labeled["Patient ID"],
    scores[scores["flag_neuro_like"] == 1]["Patient ID"]
]).unique()

# Select rows in untagged with matching patient ids
rows_to_move = untagged[untagged["Patient ID"].isin(ids_to_move)]

# Count unique patient ids and total rows moved
num_unique_patients_moved = rows_to_move["Patient ID"].nunique()
num_rows_moved = len(rows_to_move)
print(f"Unique Patient IDs moved: {num_unique_patients_moved}")
print(f"Total rows moved: {num_rows_moved}")

# Remove these rows from untagged
untagged = untagged[~untagged["Patient ID"].isin(ids_to_move)]

# Add them to labeled_neuro
labeled = pd.concat([labeled, rows_to_move], ignore_index=True)

filtered.to_csv("unlabeled_filtered.csv", index=False)
labeled.to_csv("updated_labeled_neuro.csv", index=False)
