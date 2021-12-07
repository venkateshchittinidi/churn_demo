from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__, template_folder='template')
adaboost = pickle.load(open('model_pkl', 'rb'))

@app.route('/')
def home():
    return render_template("homepage.html")



def get_data():
    SeniorCitizen_data = request.form.get('SeniorCitizen')
    SeniorCitizen = int(SeniorCitizen_data)
    Partner_data = request.form.get('Partner')
    Partner = int(Partner_data)
    Tenure_data = request.form.get('Tenure')
    MonthlyCharges_data = request.form.get('MonthlyCharges')
    TotalCharges_data = request.form.get('TotalCharges')
    Gender = request.form.get('Gender')
    Dependents_data = request.form.get('Dependents')
    Dependents = int(Dependents_data)
    PhoneService_data = request.form.get('PhoneService')
    PhoneService = int(PhoneService_data)
    MultipleLines = request.form.get('MultipleLines')
    InternetService = request.form.get('InternetService')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = request.form.get('DeviceProtection') 
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')
    #Churn = request.form.get('Churn')
    PaperlessBilling_data = request.form.get('PaperlessBilling')
    PaperlessBilling = int(PaperlessBilling_data)
    PaymentMethod = request.form.get('PaymentMethod')

    Tenure = int(Tenure_data)
    MonthlyCharges = int(MonthlyCharges_data)
    TotalCharges = int(TotalCharges_data)


    d_dict = {'SeniorCitizen': [SeniorCitizen],'Partner': [Partner],'Dependents': [Dependents],'Tenure': [Tenure], 'PhoneService': [PhoneService],
              'PaperlessBilling': [PaperlessBilling],'MonthlyCharges': [MonthlyCharges],'TotalCharges': [TotalCharges],'Gender_female': [0],
              'Gender_male': [0],'MultipleLines_no': [0], 'MultipleLines_nophoneservice': [0],'MultipleLines_yes': [0],
              'InternetService_dsl': [0], 'InternetService_fiberoptic': [0],'InternetService_no': [0], 'OnlineSecurity_no': [0],
               'OnlineSecurity_nointernetservice': [0],'OnlineSecurity_yes': [0],'OnlineBackup_no': [0], 'OnlineBackup_nointernetservice': [0],
               'OnlineBackup_yes': [0],'DeviceProtection_no': [0], 'DeviceProtection_nointernetservice': [0],'DeviceProtection_yes': [0],
               'TechSupport_no': [0], 'TechSupport_nointernetservice': [0],'TechSupport_yes': [0],'StreamingTV_no': [0], 'StreamingTV_nointernetservice': [0],
               'StreamingTV_yes': [0],'StreamingMovies_no': [0], 'StreamingMovies_nointernetservice': [0],'StreamingMovies_yes': [0],
               'PaymentMethod_banktransfer(automatic)': [0], 'PaymentMethod_creditcard(automatic)': [0],
              'PaymentMethod_electroniccheck': [0], 'PaymentMethod_mailedcheck': [0]}


    data = {'Tenure': [Tenure], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges], 'Gender_female': [0],
              'Gender_male': [0], 'SeniorCitizen': [SeniorCitizen],  'Dependents': [Dependents], 'PhoneService': [PhoneService],
              'MultipleLines_no': [0], 'MultipleLines_nophoneservice': [0],'MultipleLines_yes': [0], 
              'InternetService_dsl': [0], 'InternetService_fiberoptic': [0],'InternetService_no': [0], 
              'OnlineSecurity_no': [0], 'OnlineSecurity_nointernetservice': [0],'OnlineSecurity_yes': [0],
              'OnlineBackup_no': [0], 'OnlineBackup_nointernetservice': [0],'OnlineBackup_yes': [0],
              'DeviceProtection_no': [0], 'DeviceProtection_nointernetservice': [0],'DeviceProtection_yes': [0],
              'TechSupport_no': [0], 'TechSupport_nointernetservice': [0],'TechSupport_yes': [0], 
              'StreamingTV_no': [0], 'StreamingTV_nointernetservice': [0],'StreamingTV_yes': [0], 
              'StreamingMovies_no': [0], 'StreamingMovies_nointernetservice': [0],'StreamingMovies_yes': [0],
              #'Churn': [Churn],
              'PaperlessBilling': [PaperlessBilling],
              'PaymentMethod_banktransfer(automatic)': [0], 'PaymentMethod_creditcard(automatic)': [0],
              'PaymentMethod_electroniccheck': [0], 'PaymentMethod_mailedcheck': [0]}
    print(d_dict,"beforeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    replace_list = [Gender, MultipleLines,
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                    TechSupport, StreamingTV, StreamingMovies, PaymentMethod]

    for key, value in d_dict.items():
        if key in replace_list:
            d_dict[key] = 1

    print(d_dict,"afterrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
    data = pd.DataFrame.from_dict(d_dict, orient='columns')
    data.head()
    print(data.dtypes)
    return pd.DataFrame.from_dict(d_dict, orient='columns')

def feature_imp(model, data):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_30 = indices[:30]
    data = data.iloc[:, top_30]
    return data

def min_max_scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler.fit(data)
    data_scaled = scaler.fit_transform(data.values.reshape(30, -1))
    data = data_scaled.reshape(-1, 30)
    return pd.DataFrame(data)

@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    #featured_data = feature_imp(adaboost, df)
    #scaled_data = min_max_scale(featured_data)
    prediction = adaboost.predict(df)
    print(prediction,"ooooooooooooooooooutput!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return render_template('results.html', tables = [df.to_html(classes='data', header=True)],result = prediction[0])
    #return render_template('results.html',result = prediction[0])



if __name__=="__main__":
    app.run(debug=True)