import pandas as pd

tagged_not_glaucoma = pd.read_csv("tagged_not_glaucoma.csv")
glaucoma_tagged = pd.read_csv("glaucoma_tagged.csv")
sus_glaucoma = pd.read_csv("sus_glaucoma.csv")


patient_ids = set(glaucoma_tagged["Patient ID"].unique())
print("Number of unique Patient IDs in glaucoma_tagged:", len(patient_ids))

rows_in_sus_with_glaucoma_id = sus_glaucoma[sus_glaucoma["Patient ID"].isin(patient_ids)]
print("Rows in sus_glaucoma with Patient ID in glaucoma_tagged:", len(rows_in_sus_with_glaucoma_id))

# Add rows to glaucoma_tagged and remove from sus_glaucoma
glaucoma_tagged = pd.concat([glaucoma_tagged, rows_in_sus_with_glaucoma_id], ignore_index=True)
sus_glaucoma = sus_glaucoma[~sus_glaucoma["Patient ID"].isin(patient_ids)]

patient_ids_not_glaucoma = set(tagged_not_glaucoma["Patient ID"].unique())
print("Number of unique Patient IDs in tagged_not_glaucoma:", len(patient_ids_not_glaucoma))

# Remove rows from sus_glaucoma with Patient IDs in tagged_not_glaucoma
before_removal = len(sus_glaucoma)
sus_glaucoma = sus_glaucoma[~sus_glaucoma["Patient ID"].isin(patient_ids_not_glaucoma)]
removed_count = before_removal - len(sus_glaucoma)
print("Rows removed from sus_glaucoma by tagged_not_glaucoma Patient IDs:", removed_count)

# Remove rows with Signal Strength < 0.6
glaucoma_tagged = glaucoma_tagged[glaucoma_tagged["Signal Strength"] >= 0.6]
sus_glaucoma = sus_glaucoma[sus_glaucoma["Signal Strength"] >= 0.6]

print("Number of rows in glaucoma_tagged:", len(glaucoma_tagged))
print("Number of rows in sus_glaucoma:", len(sus_glaucoma))

glaucoma_tagged.to_csv("glaucoma_positives.csv", index=False)
sus_glaucoma.to_csv("glaucoma_suspects.csv", index=False)

