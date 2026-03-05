import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model (file same repo mein hoga)
model = joblib.load('psyche_risk_model_full.joblib')

st.title("PsycheRisk – Mental Health Risk Predictor")
st.markdown("**Demo only** — not medical advice. Enter details below.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Self-Employed"])

    sleep_hours = st.slider("Sleep Hours/Night", 3.0, 10.0, 7.0, 0.1)
    physical_activity = st.slider("Physical Activity (h/week)", 0.0, 15.0, 5.0, 0.1)
    screen_time = st.slider("Screen Time (h/day)", 1.0, 12.0, 6.0, 0.1)
    social_support = st.slider("Social Support (1-10)", 1, 10, 6)
    work_stress = st.slider("Work Stress (1-10)", 1, 10, 5)

with col2:
    st.subheader("More Scores")
    academic_pressure = st.slider("Academic Pressure", 1, 10, 4)
    job_satisfaction = st.slider("Job Satisfaction", 1, 10, 6)
    financial_stress = st.slider("Financial Stress", 1, 10, 4)
    working_hours = st.slider("Working Hours/Week", 10, 80, 40)
    anxiety_score = st.slider("Anxiety Score", 1, 10, 4)
    depression_score = st.slider("Depression Score", 1, 10, 4)
    stress_level = st.slider("Stress Level", 1, 10, 5)
    mood_swings = st.slider("Mood Swings Freq", 1, 10, 3)
    concentration_difficulty = st.slider("Concentration Difficulty", 1, 10, 4)

    st.subheader("History")
    panic_attack = st.checkbox("Panic Attack History")
    family_history = st.checkbox("Family Mental Illness History")
    prev_diagnosis = st.checkbox("Previous Diagnosis")
    therapy = st.checkbox("Therapy History")
    substance = st.checkbox("Substance Use")

if st.button("Predict Risk", type="primary"):
    data = {
        'age': age, 'gender': gender, 'marital_status': marital_status,
        'education_level': education_level, 'employment_status': employment_status,
        'sleep_hours': sleep_hours, 'physical_activity_hours_per_week': physical_activity,
        'screen_time_hours_per_day': screen_time, 'social_support_score': social_support,
        'work_stress_level': work_stress, 'academic_pressure_level': academic_pressure,
        'job_satisfaction_score': job_satisfaction, 'financial_stress_level': financial_stress,
        'working_hours_per_week': working_hours, 'anxiety_score': anxiety_score,
        'depression_score': depression_score, 'stress_level': stress_level,
        'mood_swings_frequency': mood_swings, 'concentration_difficulty_level': concentration_difficulty,
        'panic_attack_history': int(panic_attack), 'family_history_mental_illness': int(family_history),
        'previous_mental_health_diagnosis': int(prev_diagnosis),
        'therapy_history': int(therapy), 'substance_use': int(substance)
    }
    df = pd.DataFrame([data])

    # Engineered features
    df['negative_load'] = (
        df['work_stress_level'] + df['academic_pressure_level'] +
        df['financial_stress_level'] + df['anxiety_score'] +
        df['depression_score'] + df['stress_level'] +
        df['mood_swings_frequency'] + df['concentration_difficulty_level']
    ) / 8.0

    df['protective_factors'] = (
        df['social_support_score'] +
        df['job_satisfaction_score'] +
        (10 - df['screen_time_hours_per_day'].clip(0,10))
    ) / 3.0

    df['sleep_quality'] = np.where(
        df['sleep_hours'].between(7, 9), 1,
        np.where(df['sleep_hours'] < 6, -1, 0)
    )

    df['work_life_imbalance'] = df['working_hours_per_week'] / 40.0 - 1.0

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    labels = ["Low Risk", "Medium Risk", "High Risk"]
    st.success(f"**Predicted: {labels[pred]}**")

    st.markdown("Probabilities:")
    for l, p in zip(labels, probs):
        st.write(f"- {l}: {p:.1%}")
