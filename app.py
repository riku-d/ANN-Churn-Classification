import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Caching model and objects
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_data
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

model = load_model()
onehot_encoder_geo = load_pickle('one_hot_encoder.pkl')
label_encoder_gender = load_pickle('label_encoder_gender.pkl')
scaler = load_pickle('scaler.pkl')

# Streamlit app
st.title("📊 Customer Churn Prediction App")

# User input
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('⚥ Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92)
balance = st.number_input('💰 Balance')
credit_score = st.number_input('💳 Credit Score')
estimated_salary = st.number_input('🧾 Estimated Salary')
tenure = st.slider('📆 Tenure', 0, 10)
num_of_products = st.slider('🛒 Number of Products', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('✅ Is Active Member', ['Yes', 'No'])

# Predict button
if st.button('🔍 Predict'):

    # Encode inputs
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    input_data = {
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
    input_df = pd.DataFrame(input_data)

    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine
    full_input = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(full_input)

    # Predict
    prediction = model.predict(input_scaled)
    prob = prediction[0][0]

    # Output
    if prob > 0.5:
        st.error(f"🚨 The customer is likely to churn (Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ The customer is unlikely to churn (Probability: {(1 - prob)*100:.2f}%)")
