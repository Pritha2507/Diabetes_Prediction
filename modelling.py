import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.
    - target_column (str): Name of the target variable column.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train (DataFrame): Training features.
    - X_test (DataFrame): Testing features.
    - y_train (Series): Training target variable.
    - y_test (Series): Testing target variable.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_knn_model(df, target_column, n_neighbors=5):
    """
    Train k-NN model to predict diabetes risk.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.
    - target_column (str): Name of the target variable column.
    - n_neighbors (int): Number of neighbors to consider.

    Returns:
    - knn_model: Trained k-NN model.
    """
    X_train, _, y_train, _ = split_data(df, target_column)
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

    Parameters:
    - y_true (Series): True target variable values.
    - y_pred (Series): Predicted target variable values.

    Returns:
    - accuracy (float): Accuracy score.
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f1 (float): F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1
