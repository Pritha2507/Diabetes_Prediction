import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_diabetes_distribution(df, target_column):
    """
    Visualize the distribution of the target variable.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.
    - target_column (str): Name of the target variable column.
    """
    sns.countplot(x=target_column, data=df)
    plt.title('Distribution of Diabetes Risk Levels')
    plt.xlabel('Diabetes Risk Level')
    plt.ylabel('Count')
    plt.show()

def analyze_correlations(df):
    """
    Analyze correlations between features and diabetes risk.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.
    """
    correlations = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def identify_patterns_and_trends(df):
    """
    Identify patterns and trends in patient data.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.
    """
    # Example: Pairplot for visualization of relationships between features
    sns.pairplot(df)
    plt.title('Pairplot of Patient Data')
    plt.show()

def main():
    # Load cleaned and encoded data
    df = pd.read_csv('cleaned_encoded_diabetes_data.csv')

    # Assuming the target variable column name is 'Outcome'
    target_column = 'Outcome'

    # Visualize distribution of the target variable
    visualize_diabetes_distribution(df, target_column)

    # Analyze correlations between features and diabetes risk
    analyze_correlations(df)

    # Identify patterns and trends in patient data
    identify_patterns_and_trends(df)

if __name__ == "__main__":
    main()
