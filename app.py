import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import yfinance as yf

# Load data from yfinance
@st.cache
def load_data():
    stock = yf.Ticker("EBAY")
    df = stock.history(start="2023-01-01", end="2024-01-01")
    df.reset_index(inplace=True)
    return df

def train_model(df):
    df_prophet = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

def forecast(model, future):
    forecast = model.predict(future)
    return forecast

def main():
    st.title('Optimizing eBay Sales with Predictive Analytics')

    # Load data
    df = load_data()
    
    # Train the model
    model = train_model(df)
    
    # User input for prediction
    st.sidebar.header('User Input')
    date_input = st.sidebar.date_input("Select a date:", min_value=df['Date'].max())
    
    # Prepare future dataframe
    future = pd.DataFrame({'ds': [date_input]})
    
    # Make prediction
    forecast = forecast(model, future)
    predicted_price = forecast['yhat'].values[0]
    
    # Display result
    st.write(f"Predicted closing price for {date_input}: ${predicted_price:.2f}")

    # Plot historical data and forecast
    fig, ax = plt.subplots()
    model.plot(model.predict(pd.DataFrame({'ds': df['Date']})), ax=ax)
    plt.title('Stock Price Forecast')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
