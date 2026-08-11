import pandas as pd
import streamlit as st
import joblib
from joblib import load

st.title("Diabetes Prediction")
st.subheader("_______________________________")
st.write("Click Predict Button to predict weather you are diabetic or not.")

pred = st.number_input("Pregnancies", 0,200,2)
gluc = st.number_input("Glucose", 0,200,120)
blood = st.number_input("Blood Pressure", 0,200,70)
skin = st.number_input("Skin Thickness", 0,200,20)
insulin = st.number_input("Insulin", 0,200,80)
bmi = st.number_input("BMI", 0,200,25)
dpf = st.number_input("Diabetes Pedigree Function", 0.0,2.0,0.5)
age = st.number_input("Age", 0,80,15)

new_data = pd.DataFrame({
    "Pregnancies": [pred],
    "Glucose": [gluc],
    "BloodPressure": [blood],
    "SkinThickness": [skin],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

model = joblib.load("model_dir/diabetes_model.joblib") 

if st.button("Predict"):
    p = model.predict(new_data)
    if p == 1:
        st.error("You are Diabetic")
        st.write("precaution: 1. Maintain a healthy lifestyle and regular checkups to prevent diabetes."
                 "2. Eat a balanced diet, exercise regularly, and avoid smoking and excessive alcohol consumption."
                 "3. Monitor your blood sugar levels and take prescribed medications as directed."
                 "4. Stay hydrated and keep a normal weight to reduce the risk of developing diabetes."
                 "5. Maintain a healthy lifestyle avoid stress and anger"
                 "6. Avoid junk foods and starch foods and try eating green leafy veges and fruits.")
    if p == 0:
        st.success("You are not Diabetic")
        st.balloons()
