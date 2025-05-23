import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Expected columns after preprocessing
columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
           'Credit_History', 'Gender_Male', 'Married_Yes', 'Dependents_0',
           'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Graduate',
           'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban',
           'Property_Area_Urban']


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    data = request.form

    # Raw input
    raw_input = {
        'Gender': data['Gender'],
        'Married': data['Married'],
        'Dependents': data['Dependents'],
        'Education': data['Education'],
        'Self_Employed': data['Self_Employed'],
        'ApplicantIncome': float(data['ApplicantIncome']),
        'CoapplicantIncome': float(data['CoApplicantIncome']),
        'LoanAmount': float(data['LoanAmount']),
        'Loan_Amount_Term': float(data['Loan_Amount_Term']),
        'Credit_History': float(data['Credit_History']),
        'Property_Area': data['Property_Area']
    }

    # DataFrame with single row
    df = pd.DataFrame([raw_input])

    # Square root transformation (as per your model)
    df['ApplicantIncome'] = np.sqrt(df['ApplicantIncome'])
    df['CoapplicantIncome'] = np.sqrt(df['CoapplicantIncome'])
    df['LoanAmount'] = np.sqrt(df['LoanAmount'])

    # One-hot encode
    df = pd.get_dummies(df)

    # Ensure all required columns are present
    for col in columns:
        if col not in df:
            df[col] = 0
    df = df[columns]

    # Predict
    prediction = model.predict(df)[0]
    status = "Approved" if prediction == 1 else "Rejected"

    return render_template("index.html", prediction_text=f"Your Loan is: {status}")

if __name__ == "__main__":
    app.run(debug=True)
