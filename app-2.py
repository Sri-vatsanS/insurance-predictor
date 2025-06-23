
import streamlit as st
import joblib
import pandas as pd

st.title('Insurance Charges Prediction')

# Load the model and expected features
try:
    model_bundle = joblib.load('gbmreg_model.pkl')
    model = model_bundle['model']
    expected_columns = model_bundle['features']
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'gbmreg_model.pkl' exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.write("Enter the user details to predict insurance charges:")

# Collect user input
age = st.slider('Age', 18, 65, 30)
sex = st.selectbox('Sex', ['female', 'male'])
bmi = st.number_input('BMI', 10.0, 60.0, 25.0)
children = st.slider('Number of Children', 0, 5, 0)
smoker = st.selectbox('Smoker', ['no', 'yes'])
region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

# Build input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Preprocess input (match training logic)
try:
    input_df = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=False)

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

except Exception as e:
    st.error(f"Error preparing input features: {e}")
    st.stop()

# Predict
if st.button('Predict Charges'):
    try:
        prediction = model.predict(input_df)
        st.success(f'Predicted Insurance Charges: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
