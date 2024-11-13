from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Dummy dataset setup (Replace with actual path to your data)
data = pd.read_csv('Crop_recommendation.csv')  
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = data['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit-prediction', methods=['POST'])
def submit_prediction():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Validation ranges for each input
        min_values = [0, 0, 0, -10, 0, 0, 0]
        max_values = [200, 200, 200, 50, 100, 14, 500]
        inputs = [N, P, K, temperature, humidity, ph, rainfall]

        for i, val in enumerate(inputs):
            if val < min_values[i] or val > max_values[i]:
                return render_template("predict.html", error=f"Value for feature {['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'][i]} out of range ({min_values[i]}-{max_values[i]}). Please try again.")

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = RF.predict(input_data)[0]
        
        return render_template('result.html', prediction=prediction)
    
    except ValueError:
        return render_template("predict.html", error="Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)
