import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------- LOAD FILES -------- #

model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------- VALID VALUES -------- #

valid_plans = ["EPO", "POS", "HMO", "PPO", "HDHP"]

procedure_map = {
    "29881": "Knee Arthroscopy Surgery",
    "36415": "Blood Draw",
    "71045": "Chest X-ray",
    "93000": "ECG",
    "99213": "Office Visit Low",
    "99214": "Office Visit Moderate",
    "99283": "Emergency Visit",
    "G0439": "Annual Wellness"
}

diagnosis_map = {
    "E11.9": "Diabetes",
    "F32.9": "Depression",
    "I10": "Hypertension",
    "J45.909": "Asthma",
    "M54.5": "Back Pain",
    "N39.0": "UTI",
    "R05": "Cough",
    "Z00.00": "General Check-up"
}

# -------- UI -------- #

st.title("🏥 Claim Denial Prediction System")

age = st.number_input("Enter Age", min_value=0, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
network = st.selectbox("In Network?", ["Yes", "No"])
prior_auth = st.selectbox("Prior Authorization", ["Yes", "No"])
billing = st.number_input("Billed Amount", min_value=0.0)
delay = st.number_input("Days Delay", min_value=0, step=1)
plan = st.selectbox("Insurance Plan", valid_plans)
procedure = st.selectbox("Procedure Code", list(procedure_map.keys()))
diagnosis = st.selectbox("Diagnosis Code", list(diagnosis_map.keys()))

# -------- PREDICT BUTTON -------- #

if st.button("Predict Claim Status"):

    if age <= 0:
        st.error("❌ Age must be greater than 0")
        st.stop()

    # -------- FEATURE BUILD -------- #

    network_val = 1 if network == "Yes" else 0
    prior_auth_val = 1 if prior_auth == "Yes" else 0

    gender_val = le.transform([gender])[0]

    user_data = pd.DataFrame(columns=columns)
    user_data.loc[0] = 0

    user_data['patient_age_years'] = age
    user_data['patient_gender'] = gender_val
    user_data['is_in_network'] = network_val
    user_data['prior_auth_required'] = prior_auth_val
    user_data['billed_amount_usd'] = billing
    user_data['days_between_service_and_submission'] = delay

    # One-hot encoding mapping
    if f"insurance_plan_type_{plan}" in user_data.columns:
        user_data[f"insurance_plan_type_{plan}"] = 1

    if f"procedure_code_cpt_{procedure}" in user_data.columns:
        user_data[f"procedure_code_cpt_{procedure}"] = 1

    if f"primary_diagnosis_code_icd10_{diagnosis}" in user_data.columns:
        user_data[f"primary_diagnosis_code_icd10_{diagnosis}"] = 1

    # -------- SCALING -------- #

    user_scaled = scaler.transform(user_data)

    # -------- PREDICTION -------- #

    prob = model.predict_proba(user_scaled)[0][1] * 100

    # -------- EXPLANATION -------- #

    reasons = []
    force_denied = False

    if network_val == 0:
        reasons.append("Out-of-network")

    if prior_auth == "No":
        reasons.append("Authorization missing")
        force_denied = True

    if billing > 10000:
        reasons.append("High billing")

    if delay > 30:
        reasons.append("Late submission")

    system_reason = ", ".join(reasons) if reasons else "All good"

    # -------- DECISION -------- #

    if force_denied:
        status = "DENIED"
        prob = max(prob, 90)

    elif len(reasons) >= 3:
        status = "DENIED"
        prob = max(prob, 80)

    elif len(reasons) == 2:
        status = "RISK"
        prob = max(prob, 50)

    else:
        if prob >= 65:
            status = "DENIED"
        elif prob >= 40:
            status = "RISK"
        else:
            status = "APPROVED"

    # -------- OUTPUT -------- #

    st.subheader("📊 Result")

    st.write("**Claim Status:**", status)
    st.write("**Denial Probability:**", round(prob, 2), "%")

    st.subheader("🧠 Explanation")
    st.write(system_reason)

    st.subheader("🩺 Medical Details")
    st.write(f"Procedure: {procedure} → {procedure_map[procedure]}")
    st.write(f"Diagnosis: {diagnosis} → {diagnosis_map[diagnosis]}")
