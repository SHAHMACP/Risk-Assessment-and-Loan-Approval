from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the models and scaler
with open('models/reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open ('models/clf_model.pkl','rb') as f:
    clf_model = pickle.load(f)    

status_map = {'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2}
edu_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
features_to_log1p = ['LoanAmount', 'MonthlyIncome', 'NetWorth']
num_cols_to_standardize = [
    'Age', 'CreditScore', 'LoanAmount', 'LoanDuration',
    'CreditCardUtilizationRate', 'LengthOfCreditHistory',
    'MonthlyIncome', 'NetWorth', 'InterestRate'
]
@app.route('/')
def home():
    return render_template('index.html',prediction = "")

@app.route('/predict', methods=['POST'])
def index():
    
    # Collect form data
    input_data = {
        'Age': int(request.form['Age']),
        'CreditScore': int(request.form['CreditScore']),
        'EmploymentStatus': request.form['EmploymentStatus'],
        'EducationLevel': request.form['EducationLevel'],
        'LoanAmount': float(request.form['LoanAmount']),
        'LoanDuration': int(request.form['LoanDuration']),
        'CreditCardUtilizationRate': float(request.form['CreditCardUtilizationRate']),
        'BankruptcyHistory': int(request.form['BankruptcyHistory']),
        'PreviousLoanDefaults': int(request.form['PreviousLoanDefaults']),
        'LengthOfCreditHistory': int(request.form['LengthOfCreditHistory']),
        'MonthlyIncome': float(request.form['MonthlyIncome']),
        'NetWorth': float(request.form['NetWorth']),
        'InterestRate': float(request.form['InterestRate']),
    }

    input_df = pd.DataFrame([input_data])

    # Encoding
    input_df['EmploymentStatus'] = input_df['EmploymentStatus'].map(status_map)
    input_df['EducationLevel'] = input_df['EducationLevel'].map(edu_map)

    # log1p transform
    input_df[features_to_log1p] = input_df[features_to_log1p].apply(np.log1p)

    # Standardize
    input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])

    # Predict
    risk_score = reg_model.predict(input_df)[0]
    risk_score = round(risk_score,2)
    approval_status = clf_model.predict([[risk_score]])
    if approval_status == 0:
        prediction_text = f"Risk Score: {risk_score} \n Your Application Rejected!"
        prediction_color = "red"
    else:
        prediction_text = f"Risk Score: {risk_score} \n Your Loan Approved"
        prediction_color = "green"

    return render_template('index.html', prediction=prediction_text, color=prediction_color)



if __name__ == '__main__':
    app.run(debug=True)
