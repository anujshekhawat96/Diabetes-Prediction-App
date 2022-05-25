# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:24:14 2022

@author: shekh
"""


import numpy as np
import pickle
import streamlit as st


# loading the saved model

#loaded_model = pickle.load(open('D:/PROJECTS\ML_867/trained_model.sav', 'rb'))

loaded_model = pickle.load(open('trained_model.sav','rb'))


#loaded_model = pickle.load(open('D:/PROJECTS\ML_867/model_pkl.pkl','rb'))

#loaded_model = pickle.load(open('model_pkl.pkl','rb'))

# creating a function for Prediction


def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

    

def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies', placeholder='Please Enter Value between 0 to 10')
    Glucose = st.text_input('Glucose Level', placeholder='Please Enter Value between 20 to 200')
    BloodPressure = st.text_input('Blood Pressure value', placeholder='Please Enter Value between 20 to 150')
    SkinThickness = st.text_input('Skin Thickness value', placeholder='Please Enter Value between 10 to 100')
    Insulin = st.text_input('Insulin Level', placeholder='Please Enter Value between 20 to 800')
    BMI = st.text_input('BMI value', placeholder='Please Enter Value between 5 to 70')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value',placeholder='Please Enter Value between 0 to 3' )
    Age = st.text_input('Age of the Person', placeholder='Please Enter Value between 21 to 90')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
