import pandas as pd

df_flagged = pd.read_csv("unlabeled_flagged_pnu.csv")
df_glaucoma = pd.read_csv("glaucoma_positives.csv")

common_cols = list(df_glaucoma.columns)
df_flagged_common = df_flagged[common_cols].copy()
combined = pd.concat([df_flagged_common, df_glaucoma], ignore_index=True)
combined.to_csv("glaucoma_macular_cube.csv", index=False)