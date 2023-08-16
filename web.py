from flask import Flask, render_template, request
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)


with open('logistic_regression_model.pkl', 'rb') as f:
    Lr_Over = pickle.load(f)
    f.close()
with open('label_encoders.pkl', 'rb') as f:
    label_encoders  = pickle.load(f)
    f.close()
with open('minmax_scaler.pkl', 'rb') as f:
    scaler  = pickle.load(f)
    f.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form', methods=['GET', 'POST'])
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict_fraud():
    if request.method == 'POST':
        # Extract form data
        Month = str(request.form['Month'])
        Make = str(request.form['Make'])
        AccidentArea = str(request.form['AccidentArea'])
        Sex = str(request.form['Sex'])
        MaritalStatus = str(request.form['MaritalStatus'])
        Age = float(request.form['Age'])
        Fault = str(request.form['Fault'])
        VehicleCategory = str(request.form['VehicleCategory'])
        VehiclePrice = str(request.form['VehiclePrice'])
        PastNumberOfClaims = str(request.form['PastNumberOfClaims'])
        AgeOfVehicle = str(request.form['AgeOfVehicle'])
        PoliceReportFiled = str(request.form['PoliceReportFiled'])
        WitnessPresent = str(request.form['WitnessPresent'])
        NumberOfSuppliments = str(request.form['NumberOfSuppliments'])
        NumberOfCars = str(request.form['NumberOfCars'])
        Year = int(request.form['Year'])
        BasePolicy = str(request.form['BasePolicy'])

    

        transformed_values = {}
        transformed_values['Month'] = label_encoders['Month'].transform([Month])[0]
        transformed_values['Make'] = label_encoders['Make'].transform([Make])[0]
        transformed_values['AccidentArea'] = label_encoders['AccidentArea'].transform([AccidentArea])[0]
        transformed_values['Sex'] = label_encoders['Sex'].transform([Sex])[0]
        transformed_values['MaritalStatus'] = label_encoders['MaritalStatus'].transform([MaritalStatus])[0]
        transformed_values['Age'] = scaler.transform([[Age]])[0][0]  # Scale Age
        transformed_values['Fault'] = label_encoders['Fault'].transform([Fault])[0]
        transformed_values['VehicleCategory'] = label_encoders['VehicleCategory'].transform([VehicleCategory])[0]
        transformed_values['VehiclePrice'] = label_encoders['VehiclePrice'].transform([VehiclePrice])[0]
        transformed_values['PastNumberOfClaims'] = label_encoders['PastNumberOfClaims'].transform([PastNumberOfClaims])[0]
        transformed_values['AgeOfVehicle'] = label_encoders['AgeOfVehicle'].transform([AgeOfVehicle])[0]
        transformed_values['PoliceReportFiled'] = label_encoders['PoliceReportFiled'].transform([PoliceReportFiled])[0]
        transformed_values['WitnessPresent'] = label_encoders['WitnessPresent'].transform([WitnessPresent])[0]
        transformed_values['NumberOfSuppliments'] = label_encoders['NumberOfSuppliments'].transform([NumberOfSuppliments])[0]
        transformed_values['NumberOfCars'] = label_encoders['NumberOfCars'].transform([NumberOfCars])[0]
        transformed_values['Year'] = Year  # Place 'Year' in second-to-last position
        transformed_values['BasePolicy'] = label_encoders['BasePolicy'].transform([BasePolicy])[0]
        input_df = pd.DataFrame([transformed_values])

        


        

        # Predict
        prediction = Lr_Over.predict(input_df)[0]
        fraud_status = 'Fraudulent' if prediction == 1 else 'Not Fraudulent'

        return render_template('result.html', fraud_status=fraud_status)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)