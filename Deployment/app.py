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
    age = st.number_input('Whats your Age', min_value=18, max_value=100, value=30)
    job = st.selectbox('Which Job Category do you belong to', options=label_encoders['job'].classes_)
    marital = st.selectbox('Marital Status', options=label_encoders['marital'].classes_)
    education = st.selectbox('Whats your Education level', options=label_encoders['education'].classes_)
    default = st.selectbox('Have you ever Defaulted on a Loan', options=label_encoders['default'].classes_)
    balance = st.number_input('Whats your Average Account Balance yearly in Ksh', value=0)
    housing = st.selectbox('Do you have a house Loan?', options=label_encoders['housing'].classes_)
    loan = st.selectbox('Do you currently have a Personal Loan', options=label_encoders['loan'].classes_)
    contact = st.selectbox('What Methos was Used to reach you', options=label_encoders['contact'].classes_)
    day = st.number_input('Approximately how many days ago did someone reach you out', min_value=1, max_value=31, value=15)
    month = st.selectbox('Lat month contact with you', options=label_encoders['month'].classes_)
    duration = st.number_input('When was the last contact Duration in Days', min_value=0, value=100)
    campaign = st.number_input('How many times did someone contact you', min_value=1, value=1)
    #pdays = st.number_input('How many days passed since last contact', min_value=-1, value=-1)
    #previous = st.number_input('Number of times someone ever contacted you before this Campaign', min_value=0, value=0)
    #poutcome = st.selectbox('How would you rate your contact experience', options=label_encoders['poutcome'].classes_)
     # Button to make prediction
    if st.button('Predict'):
        # Create a dataframe with the input features
        input_data = pd.DataFrame([[
            age, job, marital, education, default, balance, housing, loan,
            contact, day, month, duration, campaign
        ]], columns=[
            'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
            'contact', 'day', 'month', 'duration', 'campaign'
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
