import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet

# Load the trained Prophet model
model = joblib.load('prophet.pkl')

# Title of the app
st.title('Optimizing eBay Sales with Predictive Analytics ')

# Instructions
st.write("""
         ## Predict Future Stock Prices
         Enter the date you want to predict the stock price for .
         """)

# Input for target date
target_date = st.date_input("Select a date", value=pd.to_datetime('2024-12-12'))

# Convert target date to datetime and set the time to 04:00:00
target_date = pd.to_datetime(target_date).replace(hour=4, minute=0, second=0)

# Define future dates
last_date_in_data = model.history['ds'].max()
future_periods = (target_date - last_date_in_data).days + 1

# Create future dataframe
future = model.make_future_dataframe(periods=future_periods, freq='D')

# Make predictions
forecast = model.predict(future)

# Display the prediction for the target date
selected_row = forecast[forecast['ds'] == target_date]

if not selected_row.empty:
    st.write(f"## Prediction for {target_date}")
    st.write(f"**Predicted closing price:** ${selected_row['yhat'].values[0]:.2f}")
    st.write(f"**Lower bound:** ${selected_row['yhat_lower'].values[0]:.2f}")
    st.write(f"**Upper bound:** ${selected_row['yhat_upper'].values[0]:.2f}")
else:
    st.write("No prediction available for the selected date.")
