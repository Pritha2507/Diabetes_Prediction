import pandas as pd

def load_data(file_path):
    """
    Load patient data from CSV file.

    Parameters:
    - file_path (str): Path to the CSV file containing patient data.

    Returns:
    - df (DataFrame): DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Clean and preprocess patient data.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.

    Returns:
    - df_cleaned (DataFrame): Cleaned DataFrame.
    """
    # Handle missing values
    df_cleaned = df.dropna()

    # Remove outliers (you can implement outlier detection/removal techniques here)

    return df_cleaned

def encode_categorical_variables(df):
    """
    Encode categorical variables in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.

    Returns:
    - df_encoded (DataFrame): DataFrame with categorical variables encoded.
    """
    # Example: Encode gender (assuming 0 for male and 1 for female)
    df_encoded = df.replace({'gender': {'Male': 0, 'Female': 1}})

    # Encode other categorical variables as needed

    return df_encoded

def main():
    # Load data
    file_path = '/Users/admin/Desktop/diabetes/diabetes.csv'  # Replace with the actual file path
    df = load_data(file_path)

    # Clean data
    df_cleaned = clean_data(df)

    # Encode categorical variables
    df_encoded = encode_categorical_variables(df_cleaned)

    # Save cleaned and encoded data
    df_encoded.to_csv('cleaned_encoded_diabetes_data.csv', index=False)

if __name__ == "__main__":
    main()
