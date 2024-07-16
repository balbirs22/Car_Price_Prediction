from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Correctly load the pickle model
with open("LinearRegressionModel.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the dataset
car = pd.read_csv('Cleaned Car.csv')


@app.route('/')
def index():
    companies = sorted(car['Company'].unique())
    car_models = sorted(car['Name'].unique())
    year = sorted(car['Year'].unique(), reverse=True)
    fuel_type = car['Fuel_type'].unique()
    location = sorted(car['Location'].unique())
    print(companies)
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type,
                           locations=location)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    location = request.form.get('location')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company, car_model, year, fuel_type, location, kms_driven)

    # Predict using the model
    prediction = model.predict(pd.DataFrame([[car_model, company, year, fuel_type, location, kms_driven]],
                                            columns=['Name', 'Company', 'Year', 'Fuel_type', 'Location', 'Kms_driven']))
    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
