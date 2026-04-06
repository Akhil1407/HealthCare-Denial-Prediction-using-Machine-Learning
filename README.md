# HealthCare-Denial-Prediction-using-Machine-Learning

## 📌 Project Overview

The **Insurance Claim Prediction System** is a Machine Learning-based web application that predicts whether a healthcare insurance claim will be **approved or denied** based on patient, provider, and claim details.

This project demonstrates the practical application of **data preprocessing, feature engineering, and classification algorithms** in a real-world healthcare domain.

---

## 🎯 Objectives

* Predict claim approval status using ML models
* Reduce manual claim verification effort
* Improve decision-making in insurance processing

---

## 🧠 Machine Learning Approach

* Data Cleaning & Preprocessing
* Handling Categorical Variables using Encoding
* Feature Scaling using StandardScaler
* Model Training using Classification Algorithms

### Algorithms Used:

* Logistic Regression
* Random Forest
* KNN
* Decision Tree

---

## 📊 Features Used

* Patient Age
* Gender
* Insurance Plan Type
* Network Status (In/Out Network)
* Procedure Code (CPT)
* Diagnosis Code (ICD-10)
* Provider Specialty
* Facility Type

---

## 🚀 Streamlit Web App

The project includes an interactive **Streamlit UI** where users can:

* Input claim details
* Get real-time prediction
* View prediction confidence

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Streamlit
* Pickle (Model Serialization)

---

## 📂 Project Structure

```
├── app.py                  # Streamlit application
├── model.pkl              # Trained ML model
├── scaler.pkl             # Scaler object
├── columns.pkl            # Feature columns
├── Insurance_Prediction.ipynb  # Model training notebook
├── dataset.csv            # Dataset (optional or sample)
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```
streamlit run app.py
```

---

## 📈 Future Improvements

* Add more advanced models (XGBoost, LightGBM)
* Improve UI/UX
* Deploy on cloud (Streamlit Cloud / AWS)
* Add explainability (SHAP / Feature Importance)

---

## 📌 Use Case

This system can be used by:

* Insurance companies
* Healthcare analytics teams
* Claim processing systems

---

## 👨‍💻 Author

** Akhil **

---

## ⭐ Acknowledgement

This project is built for learning and demonstration purposes in Machine Learning and Data Science.
