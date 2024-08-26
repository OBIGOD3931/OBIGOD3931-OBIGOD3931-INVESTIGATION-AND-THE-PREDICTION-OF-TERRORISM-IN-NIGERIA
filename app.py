import streamlit as st
import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Load the encoders used during training
ordinal_encoder = joblib.load('re_ordinal_encoder.pkl')  # For year, month, and day
label_encoder = joblib.load('re_label_encoder.pkl')      # For State

# Streamlit App
st.title("Terrorist Attack Probability Predictor")

# Sidebar for user input features
st.sidebar.header("Input Features")

# Input fields for user data
mean_pct_read_seng15 = st.sidebar.number_input("Mean Percentage Read English (15+)", min_value=0.0, max_value=100.0, value=50.0)
avg_unemploy_state = st.sidebar.number_input("Mean Unemployment Rate by State", min_value=0.0, max_value=100.0, value=5.0)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2024)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=1)
state = st.sidebar.selectbox("State", options=label_encoder.classes_)  # Use label encoder classes for selection

# Create a DataFrame for the inputs
input_df = pd.DataFrame({
    'mean_pct_read_seng15': [mean_pct_read_seng15],
    'avg_unemploy_state': [avg_unemploy_state],
    'year': [year],
    'month': [month],
    'day': [day],
    'State': [state]
})

# Encode the categorical inputs
try:
    # Apply ordinal encoding
    input_df = ordinal_encoder.transform(input_df)
    # Apply label encoding
    input_df['State'] = label_encoder.transform(input_df['State'])
except ValueError as e:
    st.error(f"Error encoding inputs: {e}")
    st.stop()

# Add a button for making predictions
if st.button('Predict Probability'):
    # Make prediction
    probability = model.predict_proba(input_df)[0][1]  # Probability of terrorist attack

    # Display the result
    st.subheader("Prediction")
    st.write(f"The predicted probability of a terrorist attack is: {probability:.2%}")

# Additional information
st.info("This model is based on historical data and is intended for educational purposes only.")
