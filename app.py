
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


model = tf.keras.models.load_model('breast_cancer_model.h5')
scaler = joblib.load('scaler.joblib')


st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Breast Cancer Detection ")
st.markdown("""
Predict whether a tumor is **Benign** or **Malignant** using an Artificial Neural Network.
Choose your input method below.
""")


option = st.radio("Select Input Method:", ("Single Patient", "Batch CSV"))

# ------------------ Single Patient Prediction ------------------
if option == "Single Patient":
    st.subheader("Enter Patient Features")
    
    feature_names = scaler.feature_names_in_  #
    patient_data = []
    
   
    for feature in feature_names:
        value = st.slider(
            label=f"{feature}",
            min_value=0.0, 
            max_value=10.0, 
            step=0.01,
            value=0.5
        )
        patient_data.append(value)
    
    if st.button("Predict"):
        patient_array = np.array(patient_data).reshape(1, -1)
        patient_scaled = scaler.transform(patient_array)
        pred = model.predict(patient_scaled)[0][0]
        result = "Benign" if pred > 0.5 else "Malignant"
        confidence = pred if pred > 0.5 else 1 - pred
        
       
        if result == "Benign":
            st.success(f"✅ Prediction: {result} ({confidence*100:.2f}% confidence)")
        else:
            st.error(f"⚠️ Prediction: {result} ({confidence*100:.2f}% confidence)")

# ------------------ Batch Prediction from CSV ------------------
else:
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        with st.spinner("Processing batch predictions..."):
            batch_data = pd.read_csv(uploaded_file)
            batch_scaled = scaler.transform(batch_data)
            predictions = model.predict(batch_scaled)
            labels = ['Benign' if p > 0.5 else 'Malignant' for p in predictions]
            confidence_scores = [p if p > 0.5 else 1 - p for p in predictions]
            
            batch_data['Prediction'] = labels
            batch_data['Confidence'] = [f"{c.item()*100:.2f}%" for c in confidence_scores]

            
          
            def highlight_predictions(row):
                color = 'background-color: #FFCCCC' if row['Prediction'] == 'Malignant' else 'background-color: #CCFFCC'
                return [color]*len(row)
            
            st.dataframe(batch_data.style.apply(highlight_predictions, axis=1))
        
     
        csv = batch_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

