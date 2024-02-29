import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

# Load model
model = joblib.load('model2.pkl')

# Function to preprocess input data
def preprocess_input(model_general, vehicle_age, rate_of_service):
    # One-hot encode 'Model_general'
    model_general_encoded = pd.get_dummies(pd.Series(model_general))
    
    # One-hot encode 'Vehicle_age'
    vehicle_age_encoded = pd.get_dummies(pd.Series(vehicle_age))
    
    # Scale 'rate_of_service' using MinMaxScaler
    scaler = MinMaxScaler()
    rate_of_service_normalized = scaler.fit_transform([[rate_of_service]])[0][0]
    
    # Combine encoded features
    input_data = pd.concat([model_general_encoded, vehicle_age_encoded], axis=1)
    input_data['rate_of_service_normalized'] = rate_of_service_normalized
    
    return input_data

# Streamlit interface
st.title('Churn Prediction App')

# Input fields
model_general = st.selectbox('Select Model General', ['ACCORD', 'BR-V', 'CITY', 'CIVIC', 'CR-V', 'CR-Z', 'FREED', 'HR-V', 'INSIGHT', 'JAZZ', 'ODYSSEY', 'OTHERS', 'PRELUDE', 'STREAM'])
vehicle_age = st.selectbox('Select Vehicle Age', ['0', '1', '10', '11', '12', '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9', '>15'])
rate_of_service = st.slider('Rate of Service', min_value=0.0, max_value=6.0, step=0.1)

# Predict button
if st.button('Predict'):
    input_data = preprocess_input(model_general, vehicle_age, rate_of_service)
    prediction = model.predict(input_data)
    st.write(f'Churn Prediction: {prediction}')
