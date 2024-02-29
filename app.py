import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

# Load model
model = joblib.load('model2.pkl')

# Function to preprocess input data
# Function to preprocess input data
def preprocess_input(model_general, vehicle_age, rate_of_service):
    # Create DataFrame with input data
    input_data = pd.DataFrame({'model_general': [model_general], 'vehicle_age': [vehicle_age], 'rate_of_service': [rate_of_service]})
    
    # One-hot encode 'model_general'
    model_general_encoded = pd.get_dummies(input_data['model_general'], prefix='Model_general')
    # Ensure all possible columns are present, even if the current selection doesn't cover all of them
    all_model_general_columns = ['Model_general_ACCORD', 'Model_general_BR-V', 'Model_general_CITY', 'Model_general_CIVIC', 'Model_general_CR-V', 'Model_general_CR-Z', 'Model_general_FREED', 'Model_general_HR-V', 'Model_general_INSIGHT', 'Model_general_JAZZ', 'Model_general_ODYSSEY', 'Model_general_OTHERS', 'Model_general_PRELUDE', 'Model_general_STREAM']
    
  
    for column in all_model_general_columns:
        if column in model_general_encoded.columns:
            model_general_encoded[column] = 1
        else:
            model_general_encoded[column] = 0
    
    # One-hot encode 'vehicle_age'
    vehicle_age_encoded = pd.get_dummies(input_data['vehicle_age'], prefix='Vehicle_age')
    # Ensure all possible columns are present, even if the current selection doesn't cover all of them
    all_vehicle_age_columns = ['Vehicle_age_0', 'Vehicle_age_1', 'Vehicle_age_10', 'Vehicle_age_11', 'Vehicle_age_12', 'Vehicle_age_13', 'Vehicle_age_14', 'Vehicle_age_15', 'Vehicle_age_2', 'Vehicle_age_3', 'Vehicle_age_4', 'Vehicle_age_5', 'Vehicle_age_6', 'Vehicle_age_7', 'Vehicle_age_8', 'Vehicle_age_9', 'Vehicle_age_>15']
   
    for column in all_vehicle_age_columns:
        if column in vehicle_age_encoded.columns:
            vehicle_age_encoded[column] = 1
        else:
            vehicle_age_encoded[column] = 0

    
    scaler = MinMaxScaler()
    rate_of_service_normalized = scaler.fit_transform([[rate_of_service]])[0][0]
    
    # Combine encoded features with 'rate_of_service'
    input_data = pd.concat([model_general_encoded, vehicle_age_encoded, pd.Series(rate_of_service_normalized, name='rate_of_service_normalized')], axis=1)

  
  
    
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
    
    expected_columns = [
    'Model_general_ACCORD', 'Model_general_BR-V', 'Model_general_CITY',
    'Model_general_CIVIC', 'Model_general_CR-V', 'Model_general_CR-Z',
    'Model_general_FREED', 'Model_general_HR-V', 'Model_general_INSIGHT',
    'Model_general_JAZZ', 'Model_general_ODYSSEY', 'Model_general_OTHERS',
    'Model_general_PRELUDE', 'Model_general_STREAM', 'Vehicle_age_0',
    'Vehicle_age_1', 'Vehicle_age_10', 'Vehicle_age_11', 'Vehicle_age_12',
    'Vehicle_age_13', 'Vehicle_age_14', 'Vehicle_age_15', 'Vehicle_age_2',
    'Vehicle_age_3', 'Vehicle_age_4', 'Vehicle_age_5', 'Vehicle_age_6',
    'Vehicle_age_7', 'Vehicle_age_8', 'Vehicle_age_9', 'Vehicle_age_>15',
    'rate_of_service_normalized'
]

    if set(input_data.columns) == set(expected_columns):
        print("The input_data matches the expected feature columns.")
    else:
        print("There's a mismatch between the input_data and expected feature columns.")




