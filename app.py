import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

# Load model
model = joblib.load('model2.pkl')

# Function to preprocess input data
def preprocess_input(model_general, vehicle_age, rate_of_service):
    # One-hot encode 'Model_general'
    model_general_encoded = pd.get_dummies(pd.Series(model_general), dtype = int)
    
    # One-hot encode 'Vehicle_age'
    vehicle_age_encoded = pd.get_dummies(pd.Series(vehicle_age), dtype = int)
    
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
model_general = st.selectbox('Select Model General', ['Model_general_ACCORD', 'Model_general_BR-V', 'Model_general_CITY',
       'Model_general_CIVIC', 'Model_general_CR-V', 'Model_general_CR-Z',
       'Model_general_FREED', 'Model_general_HR-V', 'Model_general_INSIGHT',
       'Model_general_JAZZ', 'Model_general_ODYSSEY', 'Model_general_OTHERS',
       'Model_general_PRELUDE', 'Model_general_STREAM'])
vehicle_age = st.selectbox('Select Vehicle Age', ['Vehicle_age_0',
       'Vehicle_age_1', 'Vehicle_age_10', 'Vehicle_age_11', 'Vehicle_age_12',
       'Vehicle_age_13', 'Vehicle_age_14', 'Vehicle_age_15', 'Vehicle_age_2',
       'Vehicle_age_3', 'Vehicle_age_4', 'Vehicle_age_5', 'Vehicle_age_6',
       'Vehicle_age_7', 'Vehicle_age_8', 'Vehicle_age_9', 'Vehicle_age_>15'])
rate_of_service = st.slider('Rate of Service', min_value=0.0, max_value=6.0, step=0.1)

# Predict button
if st.button('Predict'):
    input_data = preprocess_input(model_general, vehicle_age, rate_of_service)
    prediction = model.predict(input_data)
    st.write(f'Churn Prediction: {prediction}')
