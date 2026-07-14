"""Streamlit demo: enter a customer's details and get a live loan-default risk prediction.

Run it with:   streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.joblib"


# @st.cache_resource tells Streamlit to load the model only ONCE and reuse it,
# instead of reloading from disk on every interaction (which would be slow).
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Customer Loan Risk Prediction")
st.write(
    "Enter an applicant's details below. The model predicts the probability that "
    "they will **default** on the loan, and gives an approve/reject decision based on "
    "the threshold you choose."
)
st.info(
    "Demo / educational project. Trained on the public Kaggle Credit Risk dataset with a "
    "tuned XGBoost model. Probabilities are risk scores (not calibrated), so treat them as "
    "relative rankings rather than exact percentages. Not financial advice.",
    icon="ℹ️",
)

# --- Input widgets ---
st.header("Applicant details")
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual income ($)", min_value=0, value=50000, step=1000)
    person_emp_length = st.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=5.0)
    loan_amnt = st.number_input("Loan amount ($)", min_value=500, value=10000, step=500)
    cb_person_cred_hist_length = st.number_input("Credit history length (years)", min_value=0, max_value=40, value=5)

with col2:
    person_home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox(
        "Loan purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_int_rate = st.number_input("Interest rate (%)", min_value=1.0, max_value=30.0, value=11.0)
    cb_person_default_on_file = st.selectbox("Previous default on file?", ["N", "Y"])

# loan_percent_income is a derived feature: loan amount as a fraction of income.
# We compute it automatically instead of asking the user (domain knowledge + fewer inputs).
loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
st.caption(f"Derived loan-to-income ratio: {loan_percent_income}")

# The business threshold dial: lower = catch more defaulters (stricter), higher = more lenient.
threshold = st.slider(
    "Decision threshold (probability of default above which we REJECT the loan)",
    min_value=0.05, max_value=0.95, value=0.50, step=0.01,
)

if st.button("Predict risk"):
    # Build a single-row DataFrame with the EXACT column names the pipeline expects.
    applicant = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file,
    }])

    # predict_proba returns [P(repay), P(default)]; we want column 1 = probability of default.
    default_prob = model.predict_proba(applicant)[0, 1]

    st.header("Result")
    st.metric("Probability of default", f"{default_prob:.1%}")

    if default_prob >= threshold:
        st.error(f"REJECT — default risk {default_prob:.1%} is at or above the {threshold:.0%} threshold.")
    else:
        st.success(f"APPROVE — default risk {default_prob:.1%} is below the {threshold:.0%} threshold.")
