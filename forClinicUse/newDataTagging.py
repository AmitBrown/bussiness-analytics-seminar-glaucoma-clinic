import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  Function 1: runML
# ============================================================
def runML():
    """
    1. Automatically load 'untagged_data.csv' from current folder
    2. Load tagged disease data from /tagged data/
    3. Train XGBoost classifier
    4. Predict probabilities for untagged_data.csv
    5. Save results.csv with per-disease probabilities and best_score
    """
    print("=== Eye Disease Classifier ===")

    input_path = "untagged_data.csv"
    if not os.path.exists(input_path):
        print("❌ File 'untagged_data.csv' not found in the current folder.")
        return

    print("Loading untagged data...")
    new_data = pd.read_csv(input_path)

    # ========================================================
    # Load all tagged data files
    # ========================================================
    tagged_path = os.path.join(os.getcwd(), "tagged data")
    if not os.path.exists(tagged_path):
        print("❌ Folder 'tagged data' not found.")
        return

    dfs = []
    for f in os.listdir(tagged_path):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(tagged_path, f))
            df["source_tag"] = f.replace("_df.csv", "").replace(".csv", "")
            dfs.append(df)

    if not dfs:
        print("❌ No tagged CSVs found in 'tagged data' folder.")
        return

    tagged_df = pd.concat(dfs, ignore_index=True)

    TAG_COL = "source_tag"

    # ========================================================
    # Ignore metadata columns
    # ========================================================
    ignore_cols = [
        "Date of Birth",
        "Patient ID",
        "Patient Category Name",
        "Study Instance UID",
        "OCT Scan Date",
        "Pattern Type",
        "Laterality"
    ]

    features = [col for col in tagged_df.columns if col not in ignore_cols + [TAG_COL]]

    # ========================================================
    # Prepare training data
    # ========================================================
    X = tagged_df[features].select_dtypes(include=[np.number]).fillna(0)
    y = tagged_df[TAG_COL].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ========================================================
    # Train best model (XGBoost)
    # ========================================================
    print("Training XGBoost model on tagged data...")
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model.fit(X, y_enc)
    print("✅ Model training complete.")

    # Save model & encoder for reuse
    joblib.dump(model, "model_xgb.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # ========================================================
    # Predict on new (untagged) data
    # ========================================================
    X_new = new_data[features].select_dtypes(include=[np.number]).fillna(0)
    probs = model.predict_proba(X_new)

    # Build results dataframe
    results = pd.DataFrame(probs, columns=le.classes_)
    best_tags = le.inverse_transform(np.argmax(probs, axis=1))
    best_scores = np.max(probs, axis=1)

    results["best_tag"] = best_tags
    results["best_score"] = best_scores

    results_full = pd.concat([new_data.reset_index(drop=True), results], axis=1)
    results_full.to_csv("results.csv", index=False)
    print("✅ Results saved to results.csv")


# ============================================================
#  Function 2: breakTaggedData
# ============================================================
def breakTaggedData():
    """
    Reads results.csv and appends each newly tagged row
    to its corresponding disease CSV in /tagged data/.
    """
    results_path = "results.csv"
    tagged_path = os.path.join(os.getcwd(), "tagged data")

    if not os.path.exists(results_path):
        print("❌ results.csv not found. Run runML() first.")
        return

    if not os.path.exists(tagged_path):
        print("❌ 'tagged data' folder not found.")
        return

    results_df = pd.read_csv(results_path)
    if "best_tag" not in results_df.columns:
        print("❌ 'best_tag' column missing in results.csv.")
        return

    # Append rows to each disease file
    for tag in results_df["best_tag"].unique():
        tag_df = results_df[results_df["best_tag"] == tag]
        safe_name = tag.replace(" ", "_")
        file_path = os.path.join(tagged_path, f"{safe_name}_df.csv")

        if os.path.exists(file_path):
            existing = pd.read_csv(file_path)
            combined = pd.concat([existing, tag_df], ignore_index=True)
        else:
            combined = tag_df

        combined.to_csv(file_path, index=False)
        print(f"✔ Updated {safe_name}_df.csv with {len(tag_df)} new rows.")

    print("✅ All tagged data updated successfully.")


# ============================================================
#  Run if executed directly
# ============================================================
if __name__ == "__main__":
    print("\nOptions:")
    print("1. Run disease prediction (runML)")
    print("2. Break tagged data (breakTaggedData)")
    choice = input("Select option (1/2): ").strip()

    if choice == "1":
        runML()
    elif choice == "2":
        breakTaggedData()
    else:
        print("❌ Invalid option.")
