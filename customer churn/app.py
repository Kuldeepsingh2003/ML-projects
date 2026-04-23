import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model = pickle.load(open("customer_churn_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Dashboard")

# ---------- SIDEBAR ----------
st.sidebar.header("Customer Input")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.sidebar.slider("Monthly Charges", 0, 150, 70)
    total = st.sidebar.slider("Total Charges", 0, 10000, 2000)

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    return pd.DataFrame([data])

input_df = user_input()

# ---------- PREPROCESS ----------
def preprocess(df):
    df = pd.get_dummies(df)

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]
    return df

processed = preprocess(input_df)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["🔮 Prediction", "📂 Bulk Analysis"])

# ================= TAB 1 =================
with tab1:

    st.subheader("🔍 Input Data")
    st.dataframe(input_df)

    if st.button("Predict"):

        prediction = model.predict(processed)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed)[0][1]
        else:
            probability = 0

        # KPI
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "Churn" if prediction == 1 else "Stay")
        col2.metric("Churn Probability", f"{probability:.2f}")
        col3.metric("Tenure", f"{input_df['tenure'][0]} months")

        st.progress(float(probability))

        # Result
        if prediction == 1:
            st.error("⚠️ Customer likely to CHURN")
        else:
            st.success("✅ Customer likely to STAY")

        # ---------- FEATURE IMPORTANCE ----------
        st.subheader("🔍 Top Factors Affecting Churn")

        try:
            # If VotingClassifier
            rf_model = model.named_estimators_['rf']
            importance = pd.Series(rf_model.feature_importances_, index=columns)
        except:
            try:
                # If RandomForest directly
                importance = pd.Series(model.feature_importances_, index=columns)
            except:
                importance = None

        if importance is not None:
            top_features = importance.sort_values(ascending=False).head(10)
            st.bar_chart(top_features)
        else:
            st.info("Feature importance not available for this model")

# ================= TAB 2 =================
with tab2:

    st.subheader("📂 Upload CSV for Bulk Prediction")

    file = st.file_uploader("Upload file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        processed_data = preprocess(data)

        preds = model.predict(processed_data)
        data["Prediction"] = preds

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(processed_data)[:, 1]
            data["Churn_Probability"] = probs

        # Preview
        st.write("🔍 Preview")
        st.dataframe(data.head())

        # KPIs
        total_customers = len(data)
        churned = sum(preds)

        col1, col2 = st.columns(2)
        col1.metric("Total Customers", total_customers)
        col2.metric("Predicted Churn", churned)

        # Chart
        st.subheader("📊 Churn Distribution")
        st.bar_chart(data["Prediction"].value_counts())

        # Download
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Results", csv, "churn_predictions.csv")
