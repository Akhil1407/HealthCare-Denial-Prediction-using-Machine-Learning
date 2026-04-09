import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import pytesseract
import re
import os
import zipfile

# -------- DETECT CLOUD -------- #
IS_CLOUD = os.getenv("STREAMLIT_SERVER_PORT") is not None

# -------- UNZIP MODEL -------- #
if not os.path.exists("model.pkl"):
    if os.path.exists("model.zip"):
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall()

# -------- FIND MODEL -------- #
model_path = None
for root, dirs, files in os.walk("."):
    if "model.pkl" in files:
        model_path = os.path.join(root, "model.pkl")
        break

if model_path is None:
    raise FileNotFoundError("model.pkl not found")

# -------- LOAD FILES -------- #
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------- LOCAL TESSERACT PATH -------- #
if not IS_CLOUD:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="Claim AI", layout="wide")

# -------- SESSION -------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -------- MAPS -------- #
plan_map = {
    "HMO": "Health Maintenance Organization",
    "PPO": "Preferred Provider Organization",
    "EPO": "Exclusive Provider Organization",
    "POS": "Point of Service Plan",
    "HDHP": "High Deductible Health Plan"
}

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

plan_description = {
    "HMO": "Use only network hospitals/doctors and referral is usually needed.",
    "PPO": "Flexible plan allowing any doctor visit.",
    "EPO": "Only network providers are covered.",
    "POS": "Hybrid plan with referral support.",
    "HDHP": "Lower premium but higher deductible."
}

# -------- OCR FUNCTION -------- #
def extract_details(text):
    age = None
    diagnosis = None
    procedure = None

    age_match = re.search(r'Age[:\s]*(\d+)', text, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))

    for code in diagnosis_map.keys():
        if code in text:
            diagnosis = code

    for code in procedure_map.keys():
        if code in text:
            procedure = code

    return age, diagnosis, procedure

# -------- SIDEBAR -------- #
page = st.sidebar.radio("Navigate", ["🏠 Project Info", "📊 Prediction"])

# -------- PAGE 1 -------- #
if page == "🏠 Project Info":
    st.title("🏥 Healthcare Claim Prediction System")
    st.markdown("""
- Predicts claim **Approved / Risk / Denied**
- Uses **Machine Learning + Rule-based Logic**
- Helps reduce claim errors
""")

# -------- PAGE 2 -------- #
elif page == "📊 Prediction":

    st.title("📊 Claim Prediction")

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    extracted_text = ""
    auto_age, auto_diag, auto_proc = None, None, None

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        try:
            if IS_CLOUD:
                extracted_text = "⚠ OCR not supported in cloud deployment"
            else:
                extracted_text = pytesseract.image_to_string(image)
        except:
            extracted_text = "⚠ OCR Error"

        st.subheader("📜 Extracted Text")
        st.text(extracted_text)

        auto_age, auto_diag, auto_proc = extract_details(extracted_text)

    # -------- INPUT -------- #
    age = st.number_input("Age", 1, 120, value=auto_age if auto_age else 25)

    network = st.selectbox("In Network?", ["Yes", "No"])
    prior_auth = st.selectbox("Prior Authorization", ["Yes", "No"])

    billing = st.number_input("Billing Amount ₹", min_value=0.0)
    delay = st.number_input("Submission Delay", min_value=0)

    plan = st.selectbox("Insurance Type", list(plan_map.keys()))
    st.info(plan_description[plan])

    procedure = st.selectbox("Procedure", list(procedure_map.keys()))
    diagnosis = st.selectbox("Diagnosis", list(diagnosis_map.keys()))

    if st.button("Predict"):

        # -------- DATA PREP -------- #
        user_data = pd.DataFrame(0, index=[0], columns=columns)

        user_data.loc[0, 'patient_age_years'] = age
        user_data.loc[0, 'is_in_network'] = 1 if network == "Yes" else 0
        user_data.loc[0, 'prior_auth_required'] = 1 if prior_auth == "Yes" else 0
        user_data.loc[0, 'billed_amount_usd'] = billing
        user_data.loc[0, 'days_between_service_and_submission'] = delay

        user_data.loc[0, f"insurance_plan_type_{plan}"] = 1
        user_data.loc[0, f"procedure_code_cpt_{procedure}"] = 1
        user_data.loc[0, f"primary_diagnosis_code_icd10_{diagnosis}"] = 1

        # -------- FIX COLUMN ORDER -------- #
        model_columns = list(columns)

        for col in model_columns:
            if col not in user_data.columns:
                user_data[col] = 0

        user_data = user_data[model_columns]

        # -------- SCALING -------- #
        user_scaled = scaler.transform(user_data.values)

        # -------- PREDICTION -------- #
        prob = model.predict_proba(user_scaled)[0][1] * 100

        # -------- RULES -------- #
        reasons = []

        network_val = 1 if network == "Yes" else 0
        prior_auth_val = 1 if prior_auth == "Yes" else 0

        if network_val == 0:
            reasons.append("Out-of-network provider")

        if prior_auth_val == 0:
            reasons.append("Missing prior authorization")

        if billing > 100000:
            reasons.append("High billing amount")

        if delay > 30:
            reasons.append("Late claim submission")

        # -------- DECISION -------- #
        if len(reasons) >= 3:
            status = "DENIED"
        elif len(reasons) == 2:
            status = "RISK"
        else:
            status = "APPROVED"

        # -------- OUTPUT -------- #
        st.subheader("📊 Result")
        st.write("Claim Status:", status)
        st.write("Denial Probability:", round(prob, 2), "%")

        # -------- WHY -------- #
        st.subheader("📌 Why this prediction?")

        if reasons:
            st.write(f"The claim has **{round(prob, 2)}% denial probability** because:")
            for reason in reasons:
                st.write(f"🔹 {reason}")
        else:
            approval_reasons = []

            if network_val == 1:
                approval_reasons.append("Provider is in-network")

            if prior_auth_val == 1:
                approval_reasons.append("Prior authorization available")

            if billing <= 100000:
                approval_reasons.append("Normal billing amount")

            if delay <= 30:
                approval_reasons.append("Submitted on time")

            if age <= 75:
                approval_reasons.append("Normal patient profile")

            st.write(f"The claim has **low denial probability ({round(prob, 2)}%)** because:")
            for reason in approval_reasons:
                st.write(f"✅ {reason}")

        # -------- MEDICAL INFO -------- #
        st.subheader("🩺 Medical Info")
        st.write(f"{procedure} → {procedure_map[procedure]}")
        st.write(f"{diagnosis} → {diagnosis_map[diagnosis]}")

        # -------- SAVE HISTORY -------- #
        history_row = {
            "Age": age,
            "Plan": plan,
            "Procedure": procedure,
            "Diagnosis": diagnosis,
            "Probability %": round(prob, 2),
            "Status": status
        }

        st.session_state.prediction_history.append(history_row)

    # -------- HISTORY -------- #
    st.markdown("---")
    st.subheader("📁 Prediction History Report")

    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)

        with st.expander("📊 View Previous Predictions", expanded=True):
            st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Report",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

        if st.button("🗑 Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions made yet.")
