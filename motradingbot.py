import numpy as np
import pandas as pd
import yfinance as yf
import schedule
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import alpaca_trade_api as tradeapi
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
import warnings
import json
import os
import logging
import argparse
import yfinance as yf


# Initialize Alpaca API client
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = 'PK8MVRB9DERDHF08RFLD'
APCA_API_SECRET_KEY = 'cnsPQGJa4hV1GlNyqoxgHMlCtFfeztSmGJa2wkOd'
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning, message='All PyTorch model weights were used')

# Trading parameters
STOP_LOSS_PERCENTAGE = 1.0  # 1% stop loss
TAKE_PROFIT_PERCENTAGE = 1.0  # 1% take profit
TRADE_AMOUNT = 10000  # Fixed trade amount in USD
CHECK_INTERVAL = 5  # Interval to check and execute trading logic in seconds

allocated_budget = 500  # Allocated budget for trading

def fetch_stock_data(ticker, start_date='2020-01-01'):
    df = yf.download(ticker, start=start_date)
    
    # Print the columns to check their names
    print("Available columns in fetched data:", df.columns)
    
    # Add necessary indicatorss
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['Stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    
    # Ensure the columns used are consistent
    if not all(col in df.columns for col in ['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA']):
        print("Error: Missing required columns for feature extraction.")

    return df

def preprocess_data(df):
    df = df.dropna()
    features = df[['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA']].values
    target = np.sign(df['Close'].diff().shift(-1)).dropna().values
    target = np.where(target == -1, 0, 1)
    if len(features) != len(target):
        features = features[:len(target)]
    return features, target

def train_models(X, y):
    rf_model = RandomForestClassifier()
    xgb_model = xgb.XGBClassifier()
    rf_model.fit(X, y)
    xgb_model.fit(X, y)
    lstm_model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
    lstm_model.fit(X_lstm, y, epochs=10, batch_size=32, verbose=1)
    return rf_model, xgb_model, lstm_model

# Setup logging configuration
LOG_FILE = 'trading_bot.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Function to log the allocated budget and other activities
def log_message(message):
    logging.info(message)

# Function to download stock data from Yahoo Finance
def DownloadStockdatafromYfinance(ticker, start_date='2018-01-01'):
    df = yf.download(ticker, start=start_date)
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['Stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    
    # Save the stock data to a CSV file for future reference
    df.to_csv(f"{ticker}_sample_stock_data.csv")
    print(f"Downloaded and saved stock data for {ticker} to {ticker}_sample_stock_data.csv")
    return df

def load_sample_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        features = df[['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA']].values
        if features.shape[0] == 0 or features.shape[1] != 6:
            raise ValueError("The sample data from CSV is either empty or does not have the required columns.")
        return features
    except Exception as e:
        print(f"Error loading sample data from CSV: {e}")
        return None

def fetch_stock_data_with_volume_and_trends(ticker):
    # Fetch historical stock data
    stock_data = fetch_stock_data(ticker)
    
    # Print the columns to verify their names
    print("Available columns:", stock_data.columns)
    
    # Proceed if 'volume' is available, or use the correct column name
    if 'Volume' not in stock_data.columns:
        print("Error: 'Volume' column is not available in the stock data.")
        return stock_data
    
    # Create new features for volume and candlestick trend
    stock_data['Volume'] = stock_data['Volume']  # Ensure correct column name
    stock_data['Positive_Candle'] = (stock_data['Close'] > stock_data['Open']).astype(int)
    
    # Calculate a simple rolling trend: number of positive candles in the last 5 periods
    stock_data['Candle_Trend'] = stock_data['Positive_Candle'].rolling(window=5).sum()
    
    return stock_data


def fixed_position_sizing(current_price, trade_amount=TRADE_AMOUNT):
    """
    Determine the position size based on a fixed trade amount.
    """
    position_size = trade_amount // current_price
    return max(int(position_size), 1)

# File to store allocated budget
BUDGET_FILE = 'allocated_budget.json'

# Function to load the allocated budget from a file
def load_allocated_budget():
    if os.path.exists(BUDGET_FILE):
        with open(BUDGET_FILE, 'r') as file:
            data = json.load(file)
            return data.get('allocated_budget', 500)  # Default to 500 if file is empty or key doesn't exist
    return 500  # Default budget if file doesn't exist

# Function to save the allocated budget to a file
def save_allocated_budget():
    with open(BUDGET_FILE, 'w') as file:
        json.dump({'allocated_budget': allocated_budget}, file)

# Initialize allocated budget (will be loaded from the file if it exists)
allocated_budget = load_allocated_budget()

## Function to execute an order with proper bracket pricing and budget check (Buy only, no short selling)
def execute_order(ticker, position_size, limit_price, stop_price, take_profit_price, reason):
    global allocated_budget

    estimated_order_cost = limit_price * position_size

    # Log the decision-making process for placing the order
    log_message(f"Decision: BUY {position_size} shares of {ticker}. Reason: {reason}")

    # Check if the order exceeds the allocated budget for buy scenarios
    if estimated_order_cost > allocated_budget:
        log_message(f"Cannot place buy order for {ticker}. Estimated cost ${estimated_order_cost:.2f} exceeds allocated budget of ${allocated_budget:.2f}.")
        return False  # Order not placed

    # Adjust take-profit and stop-loss prices
    take_profit_price = max(take_profit_price, round(limit_price + 0.01, 2))  # Ensure it's above limit price
    stop_price = min(stop_price, round(limit_price - 0.01, 2))  # Ensure it's below limit price

    if take_profit_price <= stop_price:
        log_message(f"Error: Take profit price must be greater than stop loss price.")
        return False

    try:
        # Attempt to submit the bracket order
        order = api.submit_order(
            symbol=ticker,
            qty=position_size,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=round(limit_price, 2),
            order_class='bracket',
            stop_loss={'stop_price': round(stop_price, 2)},
            take_profit={'limit_price': round(take_profit_price, 2)}
        )
        
        log_message(f"Placed buy order for {position_size} shares of {ticker} at limit price ${limit_price:.2f} with stop loss ${stop_price:.2f} and take profit ${take_profit_price:.2f}.")

        # Adjust the allocated budget **only after** the order is successfully placed
        allocated_budget -= estimated_order_cost
        log_message(f"New allocated budget after buy: ${allocated_budget:.2f}")

        # Save the updated budget after a successful order
        save_allocated_budget()

        return True  # Order placed successfully

    except Exception as e:
        # Log the error and prevent budget changes if the order failed
        log_message(f"Error placing order for {ticker}: {e}")
        return False  # Order not placed

# Function to apply trading logic
def trading_logic(ticker):
    global allocated_budget

    # Fetch stock data and pre-process it
    stock_data = fetch_stock_data_with_volume_and_trends(ticker)
    X, y = preprocess_data(stock_data)

    if len(X) == 0 or len(y) == 0:
        print("No data available for training.")
        return

    # Train the models
    rf_model, xgb_model, lstm_model = train_models(X, y)

    # Load latest features for prediction
    sample_data_file = f'{ticker}_sample_stock_data.csv'
    latest_features = load_sample_data_from_csv(sample_data_file)

    if latest_features is None:
        print("No valid sample data available for prediction.")
        return

    # Predict with each model
    rf_prediction = rf_model.predict(latest_features)
    xgb_prediction = xgb_model.predict(latest_features)
    lstm_prediction = lstm_model.predict(latest_features.reshape((latest_features.shape[0], latest_features.shape[1], 1)))
    lstm_prediction = np.round(lstm_prediction).astype(int).flatten()

    print(f"Model Predictions - RF: {rf_prediction}, XGB: {xgb_prediction}, LSTM: {lstm_prediction}")

    # Weighted voting for final prediction
    weights = [0.4, 0.4, 0.2]
    weighted_predictions = np.bincount([rf_prediction[0], xgb_prediction[0], lstm_prediction[0]], weights=weights)
    final_prediction = np.argmax(weighted_predictions)

    print(f"Final Prediction: {'BUY' if final_prediction == 1 else 'NO ACTION'}")

    # Fetch the latest trade price
    try:
        last_trade = api.get_latest_trade(ticker)
        current_price = last_trade.price
    except Exception as e:
        print(f"Error fetching the current price for {ticker}: {e}")
        return

    position_size = fixed_position_sizing(current_price, trade_amount=TRADE_AMOUNT)

    # Fetch the account cash buying power
    account = api.get_account()
    allocated_budget = float(account.cash)  # Update the allocated budget with current cash

    if allocated_budget < current_price * position_size:
        print("Not enough buying power.")
        return

    # Determine stop-loss and take-profit prices
    volatility = np.std(stock_data['Close'].tail(20))  # Calculate recent volatility
    if final_prediction == 1:  # Buy Signal
        stop_loss_price = current_price * (1 - STOP_LOSS_PERCENTAGE / 100 * volatility)
        take_profit_price = current_price * (1 + TAKE_PROFIT_PERCENTAGE / 100 * volatility)

        if take_profit_price <= stop_loss_price:
            print("Error: Take profit price must be greater than stop loss price.")
            return

        # Execute buy order
        if execute_order(ticker, position_size, round(current_price * 1.01, 2), stop_loss_price, take_profit_price, "Model predicted BUY"):
            print(f"New allocated budget after buy: {allocated_budget:.2f}")


    # Handle all open orders to adjust or cancel as needed
    handle_open_orders()

def handle_open_orders():
    """
    Function to handle open orders (e.g., cancel, adjust) and log them.
    """
    try:
        open_orders = api.list_orders(status='open')
        if not open_orders:
            log_message("No open orders to handle.")
        for order in open_orders:
            log_message(f"Handling open order: {order.id}, symbol: {order.symbol}, qty: {order.qty}, side: {order.side}")
            # Example of cancellation logic (if required)
            # api.cancel_order(order.id)
            log_message(f"Order {order.id} handled.")
    except Exception as e:
        log_message(f"Error handling open orders: {e}")

# Main function
def main():
    # Argument parsing to pass the ticker symbol
    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol of the stock')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to download the stock data
    ticker = args.ticker
    print(f"Downloading stock data for {ticker}...")
    
    # Fetch stock data from Yahoo Finance for the given ticker
    stock_data = DownloadStockdatafromYfinance(ticker)
    
    # Proceed with your trading logic using the downloaded data
    # ...

    # Call trading logic or other functionality that uses the stock data
    trading_logic(ticker)

if __name__ == "__main__":
    main()

# Main function
def main():
    # Argument parsing to pass the ticker symbol
    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol of the stock')

    # Parse the arguments
    args = parser.parse_args()

    # Get the ticker from arguments
    ticker = args.ticker

    # Schedule the trading logic to run periodically with the ticker argument
    schedule.every(5).seconds.do(lambda: trading_logic(ticker))

    # Keep running the schedule
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
