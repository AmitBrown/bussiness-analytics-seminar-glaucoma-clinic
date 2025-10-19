import pandas as pd

tagged = pd.read_csv("tagged_not_glaucoma.csv")
tagged_MH = tagged[tagged['Patient Category Name'].str.contains('macular', case=False, na=False)]
print("Number of rows in tagged_MH:", len(tagged_MH))

if "Patient Category Name" in tagged_MH.columns:
    tagged_MH = tagged_MH.drop(columns=["Patient Category Name"])

tagged_MH.to_csv("tagged_MH.csv", index=False)