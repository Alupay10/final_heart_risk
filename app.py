import streamlit as st
import joblib
import pandas as pd

# Load model and columns
model = joblib.load('heart_disease_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️")
st.title("Heart Disease Risk Predictor")
st.markdown("**Developed by Louie Anton Alupay**")


# Input fields
def user_input():
    age = st.slider("Age", 18, 100)
    bmi = st.number_input("BMI")
    physical_health = st.slider("Physical Health (last 30 days)", 0, 30)
    sleep_time = st.slider("Sleep Time (hours/day)", 0, 24)
    
    # Binary inputs
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
    stroke = st.selectbox("History of stroke?", ["No", "Yes"])
    diffwalking = st.selectbox("Difficulty walking?", ["No", "Yes"])
    physactivity = st.selectbox("Physical activity?", ["No", "Yes"])
    asthma = st.selectbox("Asthma?", ["No", "Yes"])
    kidney = st.selectbox("Kidney Disease?", ["No", "Yes"])
    skin = st.selectbox("Skin Cancer?", ["No", "Yes"])
    
    sex = st.selectbox("Sex", ["Female", "Male"])
    race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])
    genhlth = st.selectbox("General Health", ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])

    # Construct dataframe
    data = {
        'AgeCategory_65-69': int(age >= 65 and age <= 69),
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'SleepTime': sleep_time,
        'Smoking': 1 if smoking == "Yes" else 0,
        'AlcoholDrinking': 1 if alcohol == "Yes" else 0,
        'Stroke': 1 if stroke == "Yes" else 0,
        'DiffWalking': 1 if diffwalking == "Yes" else 0,
        'PhysicalActivity': 1 if physactivity == "Yes" else 0,
        'Asthma': 1 if asthma == "Yes" else 0,
        'KidneyDisease': 1 if kidney == "Yes" else 0,
        'SkinCancer': 1 if skin == "Yes" else 0,
        'Sex_Male': 1 if sex == "Male" else 0,
        'Race_White': 1 if race == "White" else 0,
        'Race_Black': 1 if race == "Black" else 0,
        'Race_Asian': 1 if race == "Asian" else 0,
        'Race_Hispanic': 1 if race == "Hispanic" else 0,
        'Race_Other': 1 if race == "Other" else 0,
        'GenHealth_Excellent': 1 if genhlth == "Excellent" else 0,
        'GenHealth_Fair': 1 if genhlth == "Fair" else 0,
        'GenHealth_Good': 1 if genhlth == "Good" else 0,
        'GenHealth_Poor': 1 if genhlth == "Poor" else 0,
        'GenHealth_Very good': 1 if genhlth == "Very good" else 0,
    }

    df_input = pd.DataFrame([data], columns=model_columns).fillna(0)
    return df_input

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1 if prediction == 1 else 0]
    st.subheader("Result:")
    st.write("**At Risk**" if prediction == 1 else "**Not At Risk**")
    st.write(f"Confidence Score: {confidence:.2f}")

# Optional: Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<small>Developed by <strong>Louie Anton Alupay</strong></small>"
    "</div>",
    unsafe_allow_html=True
)
