import pandas as pd

tagged = pd.read_csv("tagged_not_glaucoma.csv")
tagged_CNV = tagged[tagged['Patient Category Name'].str.contains('CNV', case=False, na=False)]
print("Number of rows in tagged_CNV:", len(tagged_CNV))

if "Patient Category Name" in tagged_CNV.columns:
    tagged_CNV = tagged_CNV.drop(columns=["Patient Category Name"])

tagged_CNV.to_csv("tagged_CNV.csv", index=False)