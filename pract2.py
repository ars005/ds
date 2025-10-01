#AIM 2c
import pandas as pd
import numpy as np

def clean_dataset(df):
    print("----Handling Missing Values----") 
    print("Missing values before cleaning:\n", df.isnull().sum())

    # Handle numeric columns
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # Handle object (categorical) columns
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values after filling:\n", df.isnull().sum())

    # --- Type Conversion ---
    print("\n--- Type Conversion ---")
    if 'some_numeric_column_as_string' in df.columns:
        df['some_numeric_column_as_string'] = pd.to_numeric(
            df['some_numeric_column_as_string'], errors='coerce'
        )
        df['some_numeric_column_as_string'] = df['some_numeric_column_as_string'].fillna(
            df['some_numeric_column_as_string'].mean()
        )
        print("Converted 'some_numeric_column_as_string' to numeric.")

    if 'date_column' in df.columns:
        df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
        print("Converted 'date_column' to datetime.")

    # --- Data Transformation ---
    print("\n--- Data Transformation ---")
    if "column_a" in df.columns and "column_b" in df.columns:
        df['new_feature'] = df['column_a'] * df['column_b']
        print("Created 'new_feature' by multiplying 'column_a' and 'column_b'.")

    # --- Removing Duplicates ---
    print("\n--- Removing Duplicates ---")
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")

    return df

# Main block
if __name__ == "__main__":
    data = {
        'numerical_col_1': [1, 2, np.nan, 4, 5],
        'numerical_col_2': [10, 5, 11, 2, 10],
        'categorical_col': ['A', 'B', 'C', 'A', np.nan],
        'some_numeric_column_as_string': ['100', '200', 'abc', '400', '500'],
        'date_column': ["2023-01-01", "2023-01-02", "invalid_date", "2023-01-04", "2023-01-05"],
        'column_a': [1, 2, 3, 4, 5],
        'column_b': [5, 4, 3, 2, 1]
    }

    sample_df = pd.DataFrame(data)
    print("Original DataFrame:\n", sample_df)

    cleaned_df = clean_dataset(sample_df.copy())
    print("\nCleaned DataFrame:\n", cleaned_df)
