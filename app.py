# import relevant packages 
import streamlit as st 
import numpy as np
import pandas as pd
import joblib
from PIL import Image


# define title and info 
st.title("Diabetes Predictor App💊")
st.info("Find out if you are at-risk of diabetes! Please note that this is **not** a substitute for professional medical advice, diagnosis, or treatment.")

with st.sidebar:
    st.header("BMI Calculator 🔢")
    st.write("As we will need your BMI index, please enter your weight and height for calculation.")
    weight = st.number_input('Enter weight (kg):', min_value=10.0, max_value=150.0)
    height = st.number_input('Enter height (m):', min_value=1.0, max_value=2.2)

    if st.button("Calculate"):
        bmi_calculator = round(weight/(height**2), 1)

        # condition for category
        if bmi_calculator < 16.0:
            category = "Underweight (Severe thinness)"
        elif 16.0 <= bmi_calculator < 17.0:
            category = "Underweight (Moderate thinness)"
        elif 17.0 <= bmi_calculator < 18.5:
            category = "Underweight (Mild thinness)"
        elif 18.5 <= bmi_calculator < 25.0:
            category = "Normal range"
        elif 25.0 <= bmi_calculator < 30.0:
            category = "Overweight (Pre-obese)"
        elif 30.0 <= bmi_calculator < 35.0:
            category = "Obese (Class I)"
        elif 35.0 <= bmi_calculator < 40.0:
            category = "Obese (Class II)"
        else:
            category = "Obese (Class III)"
        
        st.session_state.bmi = bmi_calculator
        st.session_state.category = category

    # display results
    if 'bmi' in st.session_state:
        st.write(f"BMI Index: {st.session_state.bmi}")
        st.write(f"You are **{st.session_state.category}**.")

# define input features 
st.header("Input features")
age = st.slider("Age", 0, 100)
bmi = st.slider("BMI Index (kg/m²)", 10.0, 50.0)
HbA1c_level = st.slider("Latest HbA1c_level (g/dl)", 1.0, 20.0)
blood_glucose_level = st.slider("Latest Glucose Level (mg/dL)", 50.0, 300.0)
gender = st.selectbox("Gender", ("Female", "Male", "Other"))
smoking_history = st.selectbox("Smoking History", ("current", "former", "never"))
hypertension = st.selectbox("History of Hypertension", ("Yes", "No"))
heart_disease = st.selectbox("History of Heart Disease", ("Yes", "No"))

# create dataframe to store users' inputs 
data = {
    "age": int(age),
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "bmi": round(float(bmi), 2),
    "HbA1c_level": round(float(HbA1c_level), 2),
    "blood_glucose_level": round(float(blood_glucose_level), 2),
    "gender": gender,
    "smoking_history": smoking_history, 
}
input_df = pd.DataFrame(data, index=[0])

# data preparation 
processed_df = input_df.copy()

# convert yes no to binary values 
processed_df['hypertension'] = processed_df['hypertension'].replace({'Yes': 1, 'No': 0})
processed_df['heart_disease'] = processed_df['heart_disease'].replace({'Yes': 1, 'No': 0})

# encoding categorical features
processed_df = pd.get_dummies(processed_df, columns=['gender', 'smoking_history']).astype(int)
expected_columns = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
    'gender_Female',
    'gender_Male',
    'gender_Other',
    'smoking_history_current',
    'smoking_history_former',
    'smoking_history_never'
]

for column in expected_columns:
    if column not in processed_df.columns:
        processed_df[column] = 0 

# reorder the features same as the model 
processed_df = processed_df.reindex(columns=expected_columns, fill_value=0)

# load model
if st.button("Predict"):
    rf_model = joblib.load('rf_model.pkl')
    gb_model = joblib.load('grad_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    meta_model = joblib.load('lightgbm_meta_model.pkl')

    # predict models
    rf_pred = rf_model.predict_proba(processed_df)[:, 1] 
    gb_pred = gb_model.predict_proba(processed_df)[:, 1] 
    xgb_pred = xgb_model.predict_proba(processed_df)[:, 1] 

    # stack models
    stacked_input = np.column_stack((rf_pred, gb_pred, xgb_pred))
    meta_pred = meta_model.predict(stacked_input)  
    meta_prediction_proba = meta_model.predict_proba(stacked_input)[:, 1] 

    # display results 
    st.subheader("Results")
    if meta_pred[0] == 1:
        st.write("You are predicted to be at-risk of diabetes!")
        st.dataframe(pd.DataFrame({'Probability': meta_prediction_proba}),
                    column_config={
                        'Probability': st.column_config.ProgressColumn(
                            'Probability',
                            format='%0.2f',
                            width='large',
                            min_value=0,
                            max_value=1
                        ),
                    })
    st.info(
        """Preventing diabetes involves a combination of healthy lifestyle choices and regular medical check-ups:<br>
        - **Balanced Diet**: Eat whole grains, fruits, vegetables, and lean proteins.<br>
        - **Limit Sugars**: Reduce intake of sugary and processed foods.<br>
        - **Regular Exercise**: Aim for at least 150 minutes of moderate activity weekly.<br>
        - **Weight Management**: Maintain a healthy weight to improve insulin sensitivity.<br>
        - **Stress Management**: Practice stress-reduction techniques like mindfulness or yoga.<br>
        - **Regular Check-Ups**: Get routine health screenings to catch early signs of diabetes.<br>
        - **Stay Hydrated**: Drink plenty of water and limit sugary beverages."""
    )
    else:
        st.write("You are predicted to be not at-risk of diabetes!")
        st.dataframe(pd.DataFrame({'Probability': meta_prediction_proba}),
                    column_config={
                        'Probability': st.column_config.ProgressColumn(
                            'Probability',
                            format='%0.2f',
                            width='large',
                            min_value=0,
                            max_value=1
                        ),
                    })
