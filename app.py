import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import pytesseract
import re

# -------- TESSERACT PATH -------- #
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="Claim AI", layout="wide")

# -------- SESSION HISTORY -------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -------- LOAD FILES -------- #
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

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
    "EPO": "Only network providers are covered except emergencies, usually no referral needed.",
    "POS": "Hybrid plan with referral support and optional out-of-network coverage.",
    "HDHP": "Lower premium but higher deductible before insurance starts paying."
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
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Project Info", "📊 Prediction"]
)

# -------- PAGE 1 -------- #
if page == "🏠 Project Info":

    st.title("🏥 Healthcare Claim Prediction System")

    st.markdown("""
### 📌 Project Overview
- Predicts claim **Approved / Risk / Denied**
- Uses **Machine Learning + Rule-based Logic**
- Helps reduce claim errors
- Improves claim processing efficiency

### ⚠️ Disclaimer
- Model is NOT 100% accurate
- Based on historical claim patterns
- Final decision should be reviewed manually
""")

# -------- PAGE 2 -------- #
elif page == "📊 Prediction":

    st.title("📊 Claim Prediction")

    # -------- UPLOAD -------- #
    st.subheader("📄 Upload Prescription")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    extracted_text = ""
    auto_age, auto_diag, auto_proc = None, None, None

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        extracted_text = pytesseract.image_to_string(image)

        st.subheader("📜 PATIENT DETAILS")
        st.text(extracted_text)

        auto_age, auto_diag, auto_proc = extract_details(
            extracted_text
        )

    # -------- INPUT -------- #
    age = st.number_input(
        "Age (1–120)",
        min_value=1,
        max_value=120,
        value=auto_age if auto_age else 25
    )

    network = st.selectbox(
        "In Network?[Yes/No]",
        ["Yes", "No"]
    )

    prior_auth = st.selectbox(
        "Prior Authorization[Yes/NO]",
        ["Yes", "No"]
    )

    billing = st.number_input(
        "Billing Amount ₹",
        min_value=0.0
    )

    delay = st.number_input(
        "Submission Delay [Days]",
        min_value=0
    )

    plan = st.selectbox(
        "Insurance Type [HMO,PPO,EPO,POS,HDHP]",
        list(plan_map.keys())
    )

    st.info(f"ℹ️ {plan}: {plan_description[plan]}")

    procedure = st.selectbox(
        "Procedure Code[29881,36415,71045,93000,99213,99214,99283,G0439]",
        list(procedure_map.keys()),
        index=list(procedure_map.keys()).index(auto_proc)
        if auto_proc in procedure_map else 0
    )

    diagnosis = st.selectbox(
        "Diagnosis Code[E11.9,F32.9,I10,J45.909,M54.5,N39.0,R05,Z00.00]",
        list(diagnosis_map.keys()),
        index=list(diagnosis_map.keys()).index(auto_diag)
        if auto_diag in diagnosis_map else 0
    )

    # -------- PREDICT -------- #
    if st.button("Predict"):

        network_val = 1 if network == "Yes" else 0
        prior_auth_val = 1 if prior_auth == "Yes" else 0

        user_data = pd.DataFrame(
            0,
            index=[0],
            columns=columns
        )

        user_data.loc[0, 'patient_age_years'] = age
        user_data.loc[0, 'is_in_network'] = network_val
        user_data.loc[0, 'prior_auth_required'] = prior_auth_val
        user_data.loc[0, 'billed_amount_usd'] = billing
        user_data.loc[0, 'days_between_service_and_submission'] = delay

        # -------- ONE HOT -------- #
        plan_col = f"insurance_plan_type_{plan}"
        proc_col = f"procedure_code_cpt_{procedure}"
        diag_col = f"primary_diagnosis_code_icd10_{diagnosis}"

        if plan_col in user_data.columns:
            user_data.loc[0, plan_col] = 1

        if proc_col in user_data.columns:
            user_data.loc[0, proc_col] = 1

        if diag_col in user_data.columns:
            user_data.loc[0, diag_col] = 1

        # -------- SCALING -------- #
        user_scaled = scaler.transform(user_data)

        # -------- PREDICTION -------- #
        prob = model.predict_proba(user_scaled)[0][1] * 100

        # -------- RULES -------- #
        reasons = []

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

        # -------- WHY THIS PREDICTION -------- #
        st.subheader("📌 Why this prediction?")

        if reasons:
            st.write(
                f"The claim has **{round(prob, 2)}% denial probability** because:"
            )

            for reason in reasons:
                st.write(f"🔹 {reason}")

        else:
            approval_reasons = []

            if network_val == 1:
                approval_reasons.append(
                    "Provider is in-network, reducing denial risk"
                )

            if prior_auth_val == 1:
                approval_reasons.append(
                    "Prior authorization is available"
                )

            if billing <= 100000:
                approval_reasons.append(
                    "Billing amount is within normal expected range"
                )

            if delay <= 30:
                approval_reasons.append(
                    "Claim submitted within allowed time window"
                )

            if age <= 75:
                approval_reasons.append(
                    "Patient profile falls under normal verification criteria"
                )

            st.write(
                f"The claim has a **low denial probability of {round(prob, 2)}%** because:"
            )

            for reason in approval_reasons:
                st.write(f"✅ {reason}")

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

    # -------- HISTORY REPORT -------- #
    st.markdown("---")
    st.subheader("📁 Prediction History Report")

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        with st.expander(
            "📊 View Previous Predictions",
            expanded=True
        ):
            st.dataframe(
                history_df,
                use_container_width=True
            )

        # -------- DOWNLOAD REPORT -------- #
        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Report (CSV)",
            data=csv,
            file_name="claim_prediction_history_report.csv",
            mime="text/csv"
        )

        # -------- CLEAR HISTORY -------- #
        if st.button("🗑 Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

    else:
        st.info("No predictions made yet.")