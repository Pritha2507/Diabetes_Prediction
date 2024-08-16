import pandas as pd

def create_new_features(df):
    """
    Create new features based on existing features.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.

    Returns:
    - df_with_new_features (DataFrame): DataFrame with new features added.
    """
    # Example: Creating BMI categories
    df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Example: Creating blood glucose level categories
    df['Glucose_category'] = pd.cut(df['Glucose'], bins=[0, 99, 125, float('inf')], labels=['Normal', 'Prediabetic', 'Diabetic'])

    # You can create other new features based on domain knowledge or insights from EDA

    return df

def select_relevant_features(df):
    """
    Select relevant features for modeling.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.

    Returns:
    - selected_features (list): List of selected feature names.
    """
    # Example: Selecting features based on domain knowledge or feature importance from modeling
    selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_category', 'Glucose_category', 'Outcome']

    return selected_features

def main():
    # Load cleaned and encoded data
    df = pd.read_csv('cleaned_encoded_diabetes_data.csv')

    # Create new features
    df_with_new_features = create_new_features(df)

    # Select relevant features for modeling
    selected_features = select_relevant_features(df_with_new_features)

    # Save the DataFrame with new features
    df_with_new_features.to_csv('data_with_new_features.csv', index=False)

    print("Selected Features:", selected_features)

if __name__ == "__main__":
    main()
