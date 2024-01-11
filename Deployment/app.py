import pandas as pd 
import streamlit as st 
import joblib 

#loading Model weights and Labels
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_label_encoders = joblib.load("target_label_encoder.pkl")

#function to encode input data
def encode_input(input_data, encoders):
    for column, encoder in encoders.items():
        input_data[column] = encoder.transform(input_data[column])
    return input_data

#function to decode predictions
def decode_prediction(prediction, encoder):
    return encoder.inverse_transform(prediction)

#Streamlit App
def main():
    st.title("Bank Product Subscription Prediction")

    # Input fields for the features
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    job = st.selectbox('Job', options=label_encoders['job'].classes_)
    marital = st.selectbox('Marital', options=label_encoders['marital'].classes_)
    education = st.selectbox('Education', options=label_encoders['education'].classes_)
    default = st.selectbox('Default', options=label_encoders['default'].classes_)
    balance = st.number_input('Balance', value=0)
    housing = st.selectbox('Housing', options=label_encoders['housing'].classes_)
    loan = st.selectbox('Loan', options=label_encoders['loan'].classes_)
    
     # Button to make prediction
    if st.button('Predict'):
        # Create a dataframe with the input features
        input_data = pd.DataFrame([[
            age, job, marital, education, default, balance, housing, loan
        ]], columns=[
            'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan'
        ])

        # Encode the input data
        input_data_encoded = encode_input(input_data, label_encoders)

        # Make prediction
        prediction = model.predict(input_data_encoded)

        # Decode the prediction
        prediction_decoded = decode_prediction(prediction, target_label_encoders)

        # Display the prediction
        st.success(f'The prediction is: {prediction_decoded[0]}')

if __name__ == "__main__":
    main()
