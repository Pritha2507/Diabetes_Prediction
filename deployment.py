import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modelling import train_knn_model, evaluate_model

def monitor_model_performance(df, target_column, trained_model):
    """
    Monitor model performance over time.

    Parameters:
    - df (DataFrame): DataFrame containing patient data.
    - target_column (str): Name of the target variable column.
    - trained_model: Trained model object.
    """
    X_test, _, y_test, _ = train_test_split(df.drop(columns=[target_column]), df[target_column], test_size=0.2, random_state=42)
    y_pred = trained_model.predict(X_test)

    # Example: Calculate and log accuracy, precision, recall, and F1-score
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print("Model Performance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
