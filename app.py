import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler  # Assuming MinMaxScaler was used for scaling

# Load the pickled model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:  # Assuming scaler was saved as scaler.pkl
    scaler = pickle.load(f)

# Define the Streamlit app
def main():
    st.title('After Sales Churn Predictor')
    st.sidebar.header('Data Input')

    # Collect user input features
    rate_of_service = st.sidebar.slider('Rate of service', 0.0, 6.0, 3.0, step=0.1)
    
    # Apply scaling to numerical features
    scaled_rate_of_service = scaler.transform([[rate_of_service]])[0][0]
 
    # Apply scaling to more numerical features...

    # Categorical features


    # Define dropdown lists for each one-hot encoded categorical feature
    model = st.sidebar.selectbox('Vehicle model', ['Model_general_ACCORD', 'Model_general_BR-V', 'Model_general_CITY',
       'Model_general_CIVIC', 'Model_general_CR-V', 'Model_general_CR-Z',
       'Model_general_FREED', 'Model_general_HR-V', 'Model_general_INSIGHT',
       'Model_general_JAZZ', 'Model_general_ODYSSEY', 'Model_general_OTHERS',
       'Model_general_PRELUDE', 'Model_general_STREAM'])
    age = st.sidebar.selectbox('Vehicle age', ['Vehicle_age_0',
       'Vehicle_age_1', 'Vehicle_age_10', 'Vehicle_age_11', 'Vehicle_age_12',
       'Vehicle_age_13', 'Vehicle_age_14', 'Vehicle_age_15', 'Vehicle_age_2',
       'Vehicle_age_3', 'Vehicle_age_4', 'Vehicle_age_5', 'Vehicle_age_6',
       'Vehicle_age_7', 'Vehicle_age_8', 'Vehicle_age_9', 'Vehicle_age_>15'])


    # Map dropdown list values to numerical values
    model_mapping = {'Model_general_ACCORD': 0, 'Model_general_BR-V': 1, 'Model_general_CITY': 2,
       'Model_general_CIVIC': 3, 'Model_general_CR-V': 4, 'Model_general_CR-Z': 5,
       'Model_general_FREED': 6, 'Model_general_HR-V': 7, 'Model_general_INSIGHT': 8,
       'Model_general_JAZZ': 9, 'Model_general_ODYSSEY': 10, 'Model_general_OTHERS': 11,
       'Model_general_PRELUDE': 12, 'Model_general_STREAM': 13}
    model_encoded = model_mapping[model]

    age_mapping = {'Vehicle_age_0': 0,
       'Vehicle_age_1': 1, 'Vehicle_age_10': 2, 'Vehicle_age_11': 3, 'Vehicle_age_12': 4,
       'Vehicle_age_13': 5, 'Vehicle_age_14': 6, 'Vehicle_age_15': 7, 'Vehicle_age_2': 8,
       'Vehicle_age_3': 9, 'Vehicle_age_4': 10, 'Vehicle_age_5': 11, 'Vehicle_age_6': 12,
       'Vehicle_age_7': 13, 'Vehicle_age_8': 14, 'Vehicle_age_9': 15, 'Vehicle_age_>15': 16}
    age_encoded = age_mapping[age]

    # Encode more categorical features...

    # Make predictions
    input_data = {
     
        'rate_of_service_normalized': scaled_rate_of_service,
        'Model_general_ACCORD': 1 if model_encoded == 0 else 0,
        'Model_general_BR-V': 1 if model_encoded == 1 else 0,
        'Model_general_CITY': 1 if model_encoded == 2 else 0,
    'Model_general_CIVIC': 1 if model_encoded == 3 else 0,
         'Model_general_CR-V': 1 if model_encoded == 4 else 0,
           'Model_general_CR-Z': 1 if model_encoded == 5 else 0,
            'Model_general_FREED': 1 if model_encoded == 6 else 0,
                    'Model_general_HR-V': 1 if model_encoded == 7 else 0,
                    'Model_general_INSIGHT': 1 if model_encoded == 8 else 0,
                    'Model_general_JAZZ': 1 if model_encoded == 9 else 0,
                    'Model_general_ODYSSEY': 1 if model_encoded == 10 else 0,
                    'Model_general_OTHERS': 1 if model_encoded == 11 else 0,
                    'Model_general_PRELUDE': 1 if model_encoded == 12 else 0,
                    'Model_general_STREAM': 1 if model_encoded == 13 else 0,
               
        'Vehicle_age_0': 1 if age_encoded == 0 else 0,
         'Vehicle_age_1': 1 if age_encoded == 1 else 0,
            'Vehicle_age_10': 1 if age_encoded == 2 else 0,
         'Vehicle_age_11': 1 if age_encoded == 3 else 0,
         'Vehicle_age_12': 1 if age_encoded == 4 else 0,
         'Vehicle_age_13': 1 if age_encoded == 5 else 0,
         'Vehicle_age_14': 1 if age_encoded == 6 else 0,
         'Vehicle_age_15': 1 if age_encoded == 7 else 0,
         'Vehicle_age_2': 1 if age_encoded == 8 else 0,
         'Vehicle_age_3': 1 if age_encoded == 9 else 0,
         'Vehicle_age_4': 1 if age_encoded == 10 else 0,
         'Vehicle_age_5': 1 if age_encoded == 11 else 0,
         'Vehicle_age_6': 1 if age_encoded == 12 else 0,
         'Vehicle_age_7': 1 if age_encoded == 13 else 0,
         'Vehicle_age_8': 1 if age_encoded == 14 else 0,
         'Vehicle_age_8': 1 if age_encoded == 14 else 0,
         'Vehicle_age_8': 1 if age_encoded == 14 else 0,
         'Vehicle_age_9': 1 if age_encoded == 15 else 0,
         'Vehicle_age_>15': 1 if age_encoded == 16 else 0,
       
      
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    st.subheader('Prediction')
    if prediction[0] == 0:
        st.write('No Churn')
    else:
        st.write('Churn')

if __name__ == '__main__':
    main()
