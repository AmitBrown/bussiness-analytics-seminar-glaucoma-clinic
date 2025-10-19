import pandas as pd

def load_and_analyze_perimetry(file_path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    dropped_rows_report = {}

    # Drop rows where Stimulus Size == 5 and notify how many were dropped
    if 'Stimulus Size' in df.columns:
        initial_count = len(df)
        df = df[df['Stimulus Size'] != 5]
        dropped_count = initial_count - len(df)
        dropped_rows_report['Stimulus Size == 5'] = dropped_count

    # Drop rows where False Positives in % > 15%
    if 'False Positives in %' in df.columns:
        initial_count = len(df)
        df = df[df['False Positives in %'] <= 15]
        dropped_count = initial_count - len(df)
        dropped_rows_report['False Positives in % > 15%'] = dropped_count

    # Drop rows where False Negatives in % > 25%
    if 'False Negatives in %' in df.columns:
        initial_count = len(df)
        df = df[df['False Negatives in %'] <= 25]
        dropped_count = initial_count - len(df)
        dropped_rows_report['False Negatives in % > 25%'] = dropped_count

    # Convert Gender: FEMALE -> 0, MALE -> 1
    # 0 = FEMALE, 1 = MALE
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'FEMALE': 0, 'MALE': 1})


    # Convert Laterality to numeric encoding
    if 'Laterality' in df.columns:
        df['Laterality'] = df['Laterality'].map({'LEFT': 0, 'RIGHT': 1})
        
    # Convert Date of Birth to year only (as integer)
    # Example: 1959-01-01 -> 1959
    if 'Date of Birth' in df.columns:
        df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce').dt.year

    # Drop the Fixation Monitor column if it exists
    if 'Fixation Monitor' in df.columns:
        df = df.drop(columns=['Fixation Monitor'])

    # Keep only rows where Test Pattern == CENTRAL_24_2_THRESHOLD_TEST
    if 'Test Pattern' in df.columns:
        initial_count = len(df)
        df = df[df['Test Pattern'] == 'CENTRAL_24_2_THRESHOLD_TEST']
        dropped_count = initial_count - len(df)
        dropped_rows_report['Test Pattern != CENTRAL_24_2_THRESHOLD_TEST'] = dropped_count

    # Example analysis: show summary statistics
    summary = df.describe()

    # Save cleaned DataFrame
    df.to_csv("perimetryCleaned.csv", index=False)

    return summary, dropped_rows_report

if __name__ == "__main__":
    file_path = "perimetry_clean_excel.xlsx"  # Update with your actual file path
    summary, dropped_rows_report = load_and_analyze_perimetry(file_path)
    print("Summary Statistics:\n", summary)
    print("\nFinalized Cleaning Report:")
    for reason, count in dropped_rows_report.items():
        print(f"Rows dropped due to {reason}: {count}")


