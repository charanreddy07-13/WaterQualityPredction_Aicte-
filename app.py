# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the model and model columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #4FC3F7;
        }
        .subtitle {
            font-size: 18px;
            color: #D3D3D3;
            margin-bottom: 20px;
        }
        .result-section {
            padding: 20px;
            background-color: #1E1E1E;
            border-radius: 10px;
            margin-top: 20px;
        }
        .metric-box {
            text-align: center;
            padding: 10px;
            background-color: #262730;
            border-radius: 10px;
            margin: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-title">💧 Water Pollutants Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a year and station ID to predict water pollutant levels</div>', unsafe_allow_html=True)

# User Inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# Prediction section
if st.button("🔍 Predict"):
    if not station_id.strip():
        st.warning("⚠️ Please enter a valid Station ID.")
    else:
        # Prepare and encode input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict pollutants
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        pollutant_data = dict(zip(pollutants, predicted_pollutants))

        # Display output
        st.markdown(f"""
            <div class="result-section">
                <h3>📍 Predicted Pollutants at Station <code>{station_id}</code> for Year <code>{year_input}</code>:</h3>
            </div>
        """, unsafe_allow_html=True)

        # Create metric-style display in columns
        cols = st.columns(len(pollutants))
        for col, pollutant in zip(cols, pollutants):
            col.metric(label=pollutant, value=f"{pollutant_data[pollutant]:.2f}")

        # Optional: Show tabular version below
        df_result = pd.DataFrame([pollutant_data])
        st.markdown("### 📊 Tabular Results")
        st.dataframe(df_result, use_container_width=True)