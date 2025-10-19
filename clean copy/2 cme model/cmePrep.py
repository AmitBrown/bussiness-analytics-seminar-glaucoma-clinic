import pandas as pd

tagged = pd.read_csv("tagged_not_glaucoma.csv")
tagged_cme = tagged[tagged['Patient Category Name'].str.contains('CME', na=False)]

# Drop Patient Category Name column from tagged_cme
if "Patient Category Name" in tagged_cme.columns:
    tagged_cme = tagged_cme.drop(columns=["Patient Category Name"])

# Save tagged_cme to CSV
tagged_cme.to_csv("tagged_cme.csv", index=False)

# Remove CME rows from tagged_not_glaucoma and save
tagged_no_cme = tagged[~tagged['Patient Category Name'].str.contains('CME', na=False)]
tagged_no_cme.to_csv("tagged_not_glaucoma_no_cme.csv", index=False)