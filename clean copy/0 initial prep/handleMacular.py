import pandas as pd

def load_and_clean_excel(path: str) -> pd.DataFrame:
    """
    Load an Excel file and apply standard cleaning:
      - Convert 'Gender' column: FEMALE -> 0, MALE -> 1
      - Convert 'Date of Birth' column to year only
    """
    df = pd.read_excel(path)
    
    # Convert Gender to numeric encoding
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'FEMALE': 0, 'MALE': 1})
        
    # Convert Laterality to numeric encoding
    if 'Laterality' in df.columns:
        df['Laterality'] = df['Laterality'].map({'LEFT': 0, 'RIGHT': 1})

    # Convert Date of Birth to year
    if 'Date of Birth' in df.columns:
        df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce').dt.year
    
    return df


# Usage
macular_df = load_and_clean_excel("initial prep/macularcube_ready.xlsx")

# Drop the Pattern Type column if it exists
if 'Pattern Type' in macular_df.columns:
    macular_df = macular_df.drop(columns=['Pattern Type'])

# Normalize Patient Category Name values containing "glaucoma"
if 'Patient Category Name' in macular_df.columns:
    macular_df['Patient Category Name'] = macular_df['Patient Category Name'].apply(
        lambda x: "glaucoma" if isinstance(x, str) and "glaucoma" in x.lower() else x
    )

# Identify NEUROPHTHALMOLOGY rows
if 'Patient Category Name' in macular_df.columns:
    NEUROPHTHALMOLOGY_df = macular_df[macular_df['Patient Category Name'] == 'NEUROPHTHALMOLOGY']
    glaucoma_df = macular_df[macular_df['Patient Category Name'] == 'glaucoma']

    # Untagged: only rows with null Patient Category Name
    untagged_df = macular_df[macular_df['Patient Category Name'].isnull()]

    # Tagged: rows with Patient Category Name not null and not glaucoma
    tagged_df = macular_df[
        macular_df['Patient Category Name'].notnull() &
        (macular_df['Patient Category Name'] != 'glaucoma')
    ]
    
    # Drop the Patient Category Name if it exists,but not from the tagged df
    if 'Patient Category Name' in macular_df.columns:
        untagged_df = untagged_df.drop(columns=['Patient Category Name'])
        glaucoma_df = glaucoma_df.drop(columns=['Patient Category Name'])
        NEUROPHTHALMOLOGY_df = NEUROPHTHALMOLOGY_df.drop(columns=['Patient Category Name'])

    # the following lines are commented out to prevent file writing during testing
    NEUROPHTHALMOLOGY_df.to_csv("labeled_neuro.csv", index=False)
    glaucoma_df.to_csv("glaucoma_tagged.csv", index=False)
    untagged_df.to_csv("untagged.csv", index=False)
    tagged_df.to_csv("tagged_not_glaucoma.csv", index=False)
