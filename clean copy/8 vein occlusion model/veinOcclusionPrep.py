import pandas as pd

tagged = pd.read_csv("tagged_not_glaucoma.csv")
tagged_VO = tagged[tagged['Patient Category Name'].str.contains('vein', case=False, na=False)]
print("Number of rows in tagged_VO:", len(tagged_VO))

if "Patient Category Name" in tagged_VO.columns:
    tagged_VO = tagged_VO.drop(columns=["Patient Category Name"])

tagged_VO.to_csv("tagged_VO.csv", index=False)