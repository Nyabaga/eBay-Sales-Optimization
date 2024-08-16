import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Set page layout
st.set_page_config(layout="wide")

# App title
st.title('Optimizing eBay Sales with Predictive Analytics')

# Ticker input
ticker = st.text_input('Enter stock ticker:', 'EBAY')

# Fetch historical data from Yahoo Finance
@st.cache
def load_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='max')
    data.reset_index(inplace=True)
    return data

data = load_data(ticker)

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Prepare data for Prophet
df = data[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Train the Prophet model
m = Prophet()
m.fit(df)

# Make future predictions
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Plot forecast
fig1 = m.plot(forecast)
st.pyplot(fig1)

# Plot components
st.subheader('Forecast components')
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Optionally download the forecast data
st.download_button(label='Download Forecast Data', data=forecast.to_csv(), file_name='forecast.csv', mime='text/csv')
