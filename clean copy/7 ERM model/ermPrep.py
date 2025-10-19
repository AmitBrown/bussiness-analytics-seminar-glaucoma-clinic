import pandas as pd

tagged = pd.read_csv("tagged_not_glaucoma.csv")
tagged_ERM = tagged[tagged['Patient Category Name'].str.contains('erm', case=False, na=False)]
print("Number of rows in tagged_ERM:", len(tagged_ERM))

if "Patient Category Name" in tagged_ERM.columns:
    tagged_ERM = tagged_ERM.drop(columns=["Patient Category Name"])

tagged_ERM.to_csv("tagged_ERM.csv", index=False)