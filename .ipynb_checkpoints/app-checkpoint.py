import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet
import time

# Multi-page layout
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Custom Prediction", "Upload Data"])

    if page == "Home":
        home_page()
    elif page == "Custom Prediction":
        custom_prediction_page()
    elif page == "Upload Data":
        upload_data_page()

def home_page():
    st.title('Optimizing eBay Sales with Predictive Analytics')
    st.write("""
    ## Welcome to the Sales Prediction App
    Use this app to predict future stock prices and optimize your sales on eBay.
    Navigate using the sidebar to explore more features.
    """)

def custom_prediction_page():
    st.title('Predict Future Stock Prices')

    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("Dark Mode")
    if dark_mode:
        st.markdown(
            """
            <style>
            body {
                background-color: #121212;
                color: #ffffff;
            }
            </style>
            """, unsafe_allow_html=True
        )

    st.write("## Enter the date you want to predict the stock price for.")

    # Input for target date
    target_date = st.date_input("Select a date", value=pd.to_datetime('2024-12-12'))

    # Loading indicator while processing
    with st.spinner('Calculating prediction...'):
        time.sleep(1)  # Simulate loading time

    # Convert target date to datetime and set the time to 04:00:00
    target_date = pd.to_datetime(target_date).replace(hour=4, minute=0, second=0)

    # Load the trained Prophet model
    model = joblib.load('prophet.pkl')

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

def upload_data_page():
    st.title('Upload Your Dataset for Custom Predictions')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset")
        st.write(data)
        
        # Option to run predictions on the uploaded data
        if st.button('Run Predictions'):
            # Placeholder for running predictions on the uploaded data
            st.write("Predictions will be implemented here.")

if __name__ == "__main__":
    main()
