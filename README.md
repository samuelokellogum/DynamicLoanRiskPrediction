# 📊 Dynamic Loan Risk Prediction Using Temporal Data

This project presents a dynamic machine learning framework to predict loan default risk based on borrower behavior over time. It leverages real-world Lending Club data and integrates both classical and deep learning models for accurate, interpretable credit scoring.

## 🔍 Project Overview

Traditional credit scoring models often use static features and fail to adapt to changing borrower behavior. This project addresses that gap using:

- **Logistic Regression** – as a baseline model.
- **LSTM (Long Short-Term Memory)** – to model time-based features like payment delays.
- **SHAP (SHapley Additive exPlanations)** – to explain individual predictions.

---

## 📁 Dataset

- **Source**: [Lending Club Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Fields used**:
  - `loan_amnt`: Loan amount
  - `issue_d`: Loan issue date
  - `last_pymnt_d`: Last payment date
  - `loan_status`: Fully Paid / Charged Off (target)
  - Computed: `days_late` (repayment delay in days)

---

## 🧠 Models Used

| Model               | Description                             |
|---------------------|-----------------------------------------|
| Logistic Regression | Baseline binary classifier              |
| LSTM (Keras)        | Captures temporal payment patterns      |

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC Curve
- Confusion Matrix
- SHAP Feature Importance

---

## 📦 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- TensorFlow/Keras
- SHAP
- Matplotlib & Seaborn

---

## 🧪 How to Run

1. Clone this repository
2. Upload your `kaggle.json` API key
3. Run the notebook or script in Google Colab
4. Train and evaluate both models
5. Visualize SHAP explanations and ROC curves

---

## 👥 Authors

- **Samuel Okello**
- **Omollo Tanga**

---

## ✅ Future Work

- Deploy as a Streamlit app
- Integrate real-time data sources
- Extend to multi-class risk tiers

---

## 📄 License

This project is for academic and non-commercial use only. Please credit the authors when reusing or adapting the work.
