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
    "HMO": "Use only network hospitals/doctors and referral is usually needed for specialists.",
    "PPO": "Flexible plan allowing any doctor visit with lower cost for network providers.",
    "EPO": "Only network providers are covered except emergencies.",
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

        # -------- SAFE OCR -------- #
        try:
            if IS_CLOUD:
                extracted_text = "⚠ OCR not supported in cloud deployment"
            else:
                extracted_text = pytesseract.image_to_string(image)

        except Exception as e:
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

        user_data = pd.DataFrame(0, index=[0], columns=columns)

        user_data.loc[0, 'patient_age_years'] = age
        user_data.loc[0, 'is_in_network'] = 1 if network == "Yes" else 0
        user_data.loc[0, 'prior_auth_required'] = 1 if prior_auth == "Yes" else 0
        user_data.loc[0, 'billed_amount_usd'] = billing
        user_data.loc[0, 'days_between_service_and_submission'] = delay

        user_data.loc[0, f"insurance_plan_type_{plan}"] = 1
        user_data.loc[0, f"procedure_code_cpt_{procedure}"] = 1
        user_data.loc[0, f"primary_diagnosis_code_icd10_{diagnosis}"] = 1

        user_scaled = scaler.transform(user_data)
        prob = model.predict_proba(user_scaled)[0][1] * 100

        st.subheader("📊 Result")
        st.write("Probability:", round(prob, 2), "%")

        status = "APPROVED" if prob < 30 else "RISK" if prob < 70 else "DENIED"
        st.write("Status:", status)
