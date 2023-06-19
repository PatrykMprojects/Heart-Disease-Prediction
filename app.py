import pickle
import pandas as pd
from IPython.display import display
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import pickle
import tabpfn
import streamlit as st

#load model and test dataset

X_test = pd.read_csv('path_to_your_saved_test_data')
#load model
with open('Path_to_your_saved_model', 'rb') as file:
    model = pickle.load(file)

print(model.predict(X_test))

def predict(input):
    #change input data to an array
    input =  [float(x) for x in input]
    input_as_array = np.array(input)
    #reshape the array for prediction of one patient
    input_reshaped = input_as_array.reshape(1,-1)

    prediction = model.predict(input_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'Healthy patient'
    else:
        return 'Possible Heart Disease \n' \
               'better check with your doctor '

def main():

    # Title
    st.title(' Heart Disease Prediction ')
    age = st.text_input('Age')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Cholesterol [mm/dl]')
    thalach = st.text_input('Maximum Heart Rate')
    oldpeak = st.text_input('Oldpeak, ST depression (Yes[1]/No[0])')
    sex = st.text_input('Sex (Male[1]/Female[0])')
    fbs = st.text_input('Fasting Blood Pressure [1: if FastingBS > 120 mg/dl, 0: otherwise]')
    exang =  st.text_input('Exercise induced angina (Yes[1]/No[0]')
    cp_typicalangina = st.text_input('Typical angina (Yes[1]/No[0]')
    cp_atypical_angina = st.text_input('Atypical angina (Yes[1]/No[0]')
    cp_non_anginalpain = st.text_input('Non anginal pain (Yes[1]/No[0]')
    cp_asymptomatic = st.text_input('Asymptomatic (Yes[1]/No[0]')
    restecg_normal = st.text_input('resting electrocardiogram results Normal (Yes[1]/No[0]')
    restecg_abnormal =  st.text_input('having ST-T wave abnormality (Yes[1]/No[0]')
    restecg_hypertrophy =  st.text_input('LVH: showing probable or definite left ventricular hypertrophy (Yes[1]/No[0]')
    slope_up = st.text_input('slope of the peak exercise upsloping (Yes[1]/No[0]')
    slope_flat = st.text_input('slope of the peak exercise flat (Yes[1]/No[0]')
    slope_down = st.text_input('slope of the peak exercise downsloping (Yes[1]/No[0]')

    # code for prediction

    output = ''

    # button

    if st.button('Predict'):
        output = predict([age, trestbps, chol, thalach, oldpeak, sex, fbs, exang, cp_typicalangina, cp_atypical_angina,
                          cp_non_anginalpain, cp_asymptomatic, restecg_normal, restecg_abnormal, restecg_hypertrophy,
                          slope_up, slope_flat, slope_down])

    st.success(output)

if __name__ == '__main__':
    main()
