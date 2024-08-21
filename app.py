import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet
import time

# Multi-page layout
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Custom Prediction", "Upload Data", "About"])

    if page == "Home":
        home_page()
    elif page == "Custom Prediction":
        custom_prediction_page()
    elif page == "Upload Data":
        upload_data_page()
    elif page == "About":
        about_page()

def home_page():
    st.title('Optimizing eBay Sales with Predictive Analytics')
    st.write("""
    ## Welcome to the Sales Prediction App
    Use this app to predict future stock prices and optimize your sales on eBay.
    Navigate using the sidebar to explore more features.
    """)
    
    # Display an image on the home page
    st.image("images/eBay-prediction.jpg", use_column_width=True)

def custom_prediction_page():
    st.title('Predict Future Stock Prices')

    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
    
    if dark_mode:
        st.markdown(
            """
            <style>
            body {
                background-color: #121212;
                color: #ffffff;
            }
            .stButton>button {
                background-color: #333;
                color: #ffffff;
            }
            .stTextInput>div>input {
                background-color: #333;
                color: #ffffff;
            }
            </style>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            .stButton>button {
                background-color: #ffffff;
                color: #000000;
            }
            .stTextInput>div>input {
                background-color: #ffffff;
                color: #000000;
            }
            </style>
            """, unsafe_allow_html=True
        )

    st.write("## Enter the date range for predictions")

    # Input for start and end dates
    start_date = st.date_input("Select start date", value=pd.to_datetime('2024-12-01'))
    end_date = st.date_input("Select end date", value=pd.to_datetime('2024-12-31'))

    # Loading indicator while processing
    with st.spinner('Calculating prediction...'):
        time.sleep(1)  # Simulate loading time

    # Load the trained Prophet model
    model = joblib.load('prophet.pkl')

    # Convert start_date and end_date to datetime and set the time to 04:00:00
    start_date = pd.to_datetime(start_date).replace(hour=4, minute=0, second=0)
    end_date = pd.to_datetime(end_date).replace(hour=4, minute=0, second=0)

    # Define future dates
    last_date_in_data = model.history['ds'].max()
    future_periods = (end_date - last_date_in_data).days + 1

    # Create future dataframe
    future = model.make_future_dataframe(periods=future_periods, freq='D')

    # Make predictions
    forecast = model.predict(future)

    # Filter predictions by date range
    date_range_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]

    # Display predictions for the selected date range
    if not date_range_forecast.empty:
        st.write(f"## Predictions from {start_date.date()} to {end_date.date()}")
        for _, row in date_range_forecast.iterrows():
            st.write(f"**Date:** {row['ds'].date()}")
            st.write(f"**Predicted closing price:** ${row['yhat']:.2f}")
            st.write(f"**Lower bound:** ${row['yhat_lower']:.2f}")
            st.write(f"**Upper bound:** ${row['yhat_upper']:.2f}")
            st.write("---")
    else:
        st.write("No predictions available for the selected date range.")


def about_page():
    st.title('About eBay')

    # Add CSS to set a blurred background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("images/about page.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            position: relative;
        }
        
        .blurred-background {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            filter: blur(8px);
            z-index: -1;
        }
        
        .content {
            position: relative;
            z-index: 1;
            color: #ffffff;
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Background blur image
    st.markdown('<div class="blurred-background"></div>', unsafe_allow_html=True)

    # Display content on top of the blurred image
    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    st.write("""
    eBay is an eCommerce platform that operates as an online marketplace where individuals and businesses can buy and sell a wide variety of products, including electronics, fashion, collectibles, home goods, and more.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
