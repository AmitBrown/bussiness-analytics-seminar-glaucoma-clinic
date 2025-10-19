import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import os

# ==============================
# 1. Load CSV
# ==============================
#csv_path = input("Enter CSV path (e.g., eye_data.csv): ").strip()
df = pd.read_excel("macularcube_ready.xlsx") 

TAG_COL = "Patient Category Name"

# ==============================
# 2. Optional filter by Signal Strength
# ==============================
if "Signal Strength" in df.columns:
    answer = input("Remove rows with Signal Strength < 0.6? (y/n): ").strip().lower()
    if answer == "y":
        before = len(df)
        df = df[df["Signal Strength"] >= 0.6]
        print(f"Removed {before - len(df)} rows with weak signal strength.")
    else:
        print("Continuing without filtering by Signal Strength.")
else:
    print("Column 'Signal Strength' not found — skipping filter.")

# ==============================
# 3. Clean tag column
# ==============================
df[TAG_COL] = df[TAG_COL].astype(str).str.lower().str.strip()

tagged_df = df[df[TAG_COL].notna() & (df[TAG_COL] != "nan")]
untagged_df = df[df[TAG_COL].isna() | (df[TAG_COL] == "nan")]

# ==============================
# 4. Merge similar tag names
# ==============================
unique_tags = tagged_df[TAG_COL].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
X_tags = vectorizer.fit_transform(unique_tags)
similarity_matrix = cosine_similarity(X_tags)

merged_tags = {}
threshold = 0.8  # similarity threshold
for i, tag in enumerate(unique_tags):
    if tag in merged_tags:
        continue
    similar = [unique_tags[j] for j in range(len(unique_tags)) if similarity_matrix[i, j] > threshold]
    for s in similar:
        merged_tags[s] = tag

tagged_df[TAG_COL] = tagged_df[TAG_COL].map(merged_tags).fillna(tagged_df[TAG_COL])

# ==============================
# 5. Ignore irrelevant columns
# ==============================
ignore_cols = [
    "Date of Birth",
    "Patient ID",
    "Patient Category Name",
    "Study Instance UID",
    "OCT Scan Date",
    "Pattern Type",
    "Laterality"
]

features = [col for col in df.columns if col not in ignore_cols]

# ==============================
# 6. Create per-tag DataFrames (only tags > 500 samples)
# ==============================
tag_counts = tagged_df[TAG_COL].value_counts()
major_tags = tag_counts[tag_counts > 500].index.tolist()

tag_dfs = {}
for tag in major_tags:
    tag_df = tagged_df[tagged_df[TAG_COL] == tag].copy()
    tag_dfs[tag] = tag_df
    globals()[f"{tag}_df"] = tag_df

print(f"Created {len(tag_dfs)} tag dataframes (each >500 samples).")

# ==============================
# 7. Prepare ML data
# ==============================
le = LabelEncoder()
y = le.fit_transform(tagged_df[TAG_COL])
X = tagged_df[features].select_dtypes(include=[np.number]).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==============================
# 8. Train several models
# ==============================
models = {
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "lgbm": LGBMClassifier(random_state=42),
    "logreg": LogisticRegression(max_iter=1000, multi_class="multinomial")
}

print("Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"✔ Trained {name}")

# ==============================
# 9. Predict diseases for untagged data
# ==============================
X_un = untagged_df[features].select_dtypes(include=[np.number]).fillna(0)
results = []

xgb_model = models["xgb"]

for i, row in X_un.iterrows():
    row_arr = row.values.reshape(1, -1)
    probs = xgb_model.predict_proba(row_arr)[0]
    best_idx = np.argmax(probs)
    best_tag = le.inverse_transform([best_idx])[0]
    glaucoma_idx = np.where(le.classes_ == "glaucoma")[0]
    glaucoma_score = probs[glaucoma_idx[0]] if len(glaucoma_idx) else 0.0
    results.append({
        "row_index": i,
        "best_tag": best_tag,
        "best_score": probs[best_idx],
        "glaucoma_score": glaucoma_score
    })

pred_df = pd.DataFrame(results)
untagged_df = untagged_df.reset_index(drop=True)
untagged_df["best_tag"] = pred_df["best_tag"]
untagged_df["best_score"] = pred_df["best_score"]
untagged_df["glaucoma_score"] = pred_df["glaucoma_score"]

# ==============================
# 10. Merge predictions into tag-specific DataFrames
# ==============================
for tag in major_tags:
    tag_rows = untagged_df[untagged_df["best_tag"] == tag]
    tag_dfs[tag] = pd.concat([tag_dfs[tag], tag_rows], ignore_index=True)

# ==============================
# 11. Save all DataFrames
# ==============================
os.makedirs("disease_dfs", exist_ok=True)
for tag, tag_df in tag_dfs.items():
    safe_name = tag.replace(" ", "_")
    out_path = f"disease_dfs/{safe_name}_df.csv"
    tag_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

print("\n✅ All disease DataFrames saved successfully in 'disease_dfs/' folder.")
