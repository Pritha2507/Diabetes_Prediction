# Diabetes Risk Prediction Flask App

A web-based diabetes risk prediction tool built with Flask and a K-Nearest Neighbors (KNN) model. Users can input their health metrics, and the app predicts the likelihood of diabetes based on this information. The app also includes a monitoring function to assess the model's performance continuously.

# Overview
This Flask application uses a KNN model to predict diabetes risk based on health metrics such as glucose levels, blood pressure, BMI, and more. It includes:
1. A form for user input.
2. A backend model that processes input and generates a prediction.
3. Real-time monitoring of model performance.

# Features
1. User-friendly Interface: Input health metrics through a web form.
2. Diabetes Prediction: Provides a diabetes risk prediction based on health data.
3. Model Performance Monitoring: Monitors the performance of the KNN model each time a prediction is made.

# Installation
1. Clone the Repository
git clone https://github.com/Pritha2507/diabetes-prediction-flask-app.git
cd diabetes-prediction-flask-app
2. Install Dependencies Make sure you have Python 3.7 or higher, then install dependencies:
pip install -r requirements.txt
3. Download the Dataset Place the diabetes.csv file in the root of the project directory. The file should contain the following columns:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

# Usage
Run the Application Start the Flask server by running:
python app.py
By default, the application will be available at http://127.0.0.1:5000.

Navigate to the Home Page Open your web browser and go to http://127.0.0.1:5000. Enter your health metrics in the form and submit to see the diabetes risk prediction.

# File Structure

diabetes-prediction-flask-app/
├── app.py               # Main application script

├── deployment.py        # Model training and performance monitoring functions

├── diabetes.csv         # Dataset for model training

├── requirements.txt     # List of dependencies

└── templates/           # HTML templates

    ├── index.html       # Home page form for input
    
    └── result.html      # Page to display prediction result

# Contributing
1. Fork the repository and create a new branch.
2.Make your changes and ensure tests pass.
3. Submit a pull request describing the changes.
