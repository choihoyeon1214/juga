import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def preprocess_data(df):
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(datetime.toordinal)
    df = df[['Date', 'Close']]
    df.dropna(inplace=True)
    return df

def train_model(df):
    X = df[['Date']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def predict_future_prices(model, days_ahead=30):
    last_date = datetime.now().toordinal()
    future_dates = np.array([last_date + i for i in range(1, days_ahead + 1)]).reshape(-1, 1)
    future_prices = model.predict(future_dates)
    return future_dates, future_prices

def plot_predictions(df, future_dates, future_prices):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Historical Prices')
    plt.plot(future_dates, future_prices, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

def main(ticker):
    df = fetch_stock_data(ticker)
    df = preprocess_data(df)
    model = train_model(df)
    future_dates, future_prices = predict_future_prices(model)
    plot_predictions(df, future_dates, future_prices)

# 종목 코드를 입력하여 실행
ticker = input("종목 코드를 입력하세요 (예: AAPL): ")
main(ticker)
