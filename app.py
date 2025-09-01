import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------
# Load the saved model & preprocessor
# ----------------------------
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("post_transplant_model.joblib")
    preprocessor = joblib.load("post_transplant_preprocessor.joblib")
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(
    page_title="Pediatric BMT Mortality Risk Calculator",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Pediatric Bone Marrow Transplant Mortality Risk Calculator")
st.markdown("""
This tool uses a **machine learning model** to predict the **risk of mortality** 
in pediatric patients after **bone marrow transplantation**.
""")

# ----------------------------
# Define user input fields
# ----------------------------
st.header("🔹 Enter Patient & Transplant Details")

recipient_gender = st.selectbox("Recipient Gender", ["Male", "Female"])
stem_cell_source = st.selectbox("Stem Cell Source", ["Bone Marrow", "Peripheral Blood"])
donor_age = st.number_input("Donor Age (Years)", min_value=0, max_value=80, value=30)
donor_age_35 = st.selectbox("Donor Age ≥ 35 Years", ["No", "Yes"])
IIIV = st.selectbox("Development of Acute GVHD Stage II–IV", ["No", "Yes"])
gender_match = st.selectbox("Gender Match Between Donor & Recipient", ["Matched", "Mismatched"])
donor_ABO = st.selectbox("Donor ABO Blood Group", ["A", "B", "AB", "O"])
recipient_ABO = st.selectbox("Recipient ABO Blood Group", ["A", "B", "AB", "O"])
recipient_Rh = st.selectbox("Recipient Rh Factor", ["Positive", "Negative"])
ABO_match = st.selectbox("ABO Match Between Donor & Recipient", ["Matched", "Mismatched"])
recipient_body_mass = st.number_input("Recipient Body Mass (kg)", min_value=5.0, max_value=100.0, value=20.0)
cd34_cells = st.number_input("CD34 Cells per kg × 10⁶", min_value=0.0, value=5.0)
extensive_cGVHD = st.selectbox("Extensive Chronic GVHD", ["No", "Yes"])
cd3_cd34_ratio = st.number_input("CD3/CD34 Cell Ratio", min_value=0.0, value=1.0)
cd3_cells = st.number_input("CD3 Cells per kg × 10⁸", min_value=0.0, value=1.0)
anc_recovery = st.number_input("ANC Recovery Time (Days)", min_value=0, value=14)
plt_recovery = st.number_input("Platelet Recovery Time (Days)", min_value=0, value=30)
time_to_agvhd = st.number_input("Time to Onset of Acute GVHD Stage III/IV (Days)", min_value=0, value=30)

# ----------------------------
# Prepare input data for prediction
# ----------------------------
input_data = pd.DataFrame({
    "Recipientgender": [recipient_gender],
    "Stemcellsource": [stem_cell_source],
    "Donorage": [donor_age],
    "Donorage35": [1 if donor_age_35 == "Yes" else 0],
    "IIIV": [1 if IIIV == "Yes" else 0],
    "Gendermatch": [gender_match],
    "DonorABO": [donor_ABO],
    "RecipientABO": [recipient_ABO],
    "RecipientRh": [recipient_Rh],
    "ABOmatch": [ABO_match],
    "Rbodymass": [recipient_body_mass],
    "CD34kgx10d6": [cd34_cells],
    "extcGvHD": [1 if extensive_cGVHD == "Yes" else 0],
    "CD3dCD34": [cd3_cd34_ratio],
    "CD3dkgx10d8": [cd3_cells],
    "ANCrecovery": [anc_recovery],
    "PLTrecovery": [plt_recovery],
    "time_to_aGvHD_III_IV": [time_to_agvhd]
})

# ----------------------------
# Predict & Display Results
# ----------------------------
if st.button("Predict Mortality Risk"):
    try:
        # Preprocess input data
        processed_input = preprocessor.transform(input_data)
        
        # Predict probability
        probability = model.predict_proba(processed_input)[0][1]
        
        # Determine risk classification
        risk_category = "High Risk" if probability >= 0.27 else "Low Risk"
        
        # Display results
        st.subheader("📊 Prediction Results")
        st.metric("Predicted Mortality Risk", f"{probability:.2%}")
        st.metric("Risk Category", risk_category)
        
        # Interpret results
        if risk_category == "High Risk":
            st.error("⚠️ This patient is at **high risk** of post-transplant mortality. Close monitoring and early interventions are advised.")
        else:
            st.success("✅ This patient is at **low risk** based on current model predictions.")
        
        st.caption("⚠️ Disclaimer: This tool is for research purposes only and should not replace clinical judgment.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
