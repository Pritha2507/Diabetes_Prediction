import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
from deployment import train_knn_model, monitor_model_performance

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Train the model
trained_model = train_knn_model(df, 'Outcome')

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission and display result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = int(request.form['age'])

        # Create a DataFrame from form data
        data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age]
        })

        # Predict diabetes risk
        prediction = trained_model.predict(data)

        # Monitor model performance
        monitor_model_performance(df, 'Outcome', trained_model)

        # Return prediction
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
