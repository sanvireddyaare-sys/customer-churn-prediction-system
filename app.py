
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

@st.cache_resource
def load_artifacts():
    model = joblib.load(
        os.path.join(ARTIFACTS_DIR, "churn_model.pkl")
    )
    preprocessor = joblib.load(
        os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    )
    threshold = joblib.load(
        os.path.join(ARTIFACTS_DIR, "churn_threshold.pkl")
    )
    return model, preprocessor, threshold

st.title("📊 Customer Churn Prediction System")
st.write(
    "Estimate the probability that a telecom customer "
    "will leave the service."
)

try:
    model, preprocessor, threshold = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.info(
        "Run the notebook's model-saving cell first and "
        "ensure the artifacts folder is beside app.py."
    )
    st.stop()

st.subheader("Customer Information")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", 0, 72, 12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        )
        total = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    try:
        processed = preprocessor.transform(customer)
        probability = float(model.predict_proba(processed)[0, 1])

        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)
        c1.metric("Estimated Churn Probability", f"{probability:.1%}")
        c2.metric("Estimated Retention Probability", f"{1-probability:.1%}")

        st.progress(probability)

        if probability >= threshold:
            st.warning(
                f"Customer flagged for churn review "
                f"(threshold: {threshold:.0%})."
            )
        else:
            st.success(
                f"Customer is below the churn threshold "
                f"of {threshold:.0%}."
            )

        st.caption(
            "This is a model estimate, not a guarantee. "
            "Validate the threshold against business goals."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
