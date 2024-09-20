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
import logging
import json
import os
import argparse
import forex_python.converter as fx  # Forex data for currency exchange rates

# Execute command : python moforextradingbot.py --pair EUR/USD

# Initialize Alpaca API client (assumes you are using a forex-capable API)
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = 'PK8MVRB9DERDHF08RFLD'
APCA_API_SECRET_KEY = 'cnsPQGJa4hV1GlNyqoxgHMlCtFfeztSmGJa2wkOd'
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

# Trading parameters (pip-based)
STOP_LOSS_PIPS = 20  # Stop loss of 20 pips
TAKE_PROFIT_PIPS = 40  # Take profit of 40 pips
TRADE_AMOUNT = 10000  # Fixed trade amount in USD
CHECK_INTERVAL = 5  # Interval to check and execute trading logic in seconds

allocated_budget = 500  # Allocated budget for trading

# Setup logging configuration
LOG_FILE = 'forex_trading_bot.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Function to log messages
def log_message(message):
    logging.info(message)


# Function to fetch forex data from Alpaca API (latest quote)
def fetch_forex_data(pair):
    try:
        # Fetch the latest forex quote using Alpaca's `get_latest_quote` method
        quote = api.get_latest_quote(pair.replace("/", ""))
        
        if quote is None:
            log_message(f"No data fetched for {pair}.")
            return None
        
        # Create a DataFrame with one row containing the latest bid/ask and other fields
        data = {
            'Time': [quote.timestamp],
            'Bid Price': [quote.bid_price],
            'Ask Price': [quote.ask_price],
            'Bid Size': [quote.bid_size],
            'Ask Size': [quote.ask_size]
        }
        df = pd.DataFrame(data)
        df.set_index('Time', inplace=True)
        
        # Use the midpoint of the bid and ask prices as a proxy for the closing price
        df['Close'] = (df['Bid Price'] + df['Ask Price']) / 2
        
        # Add technical indicators (you can use the close price here for technicals)
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['Stochastic'] = StochasticOscillator(df['Bid Price'], df['Ask Price'], df['Close']).stoch()
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        df['ATR'] = AverageTrueRange(df['Bid Price'], df['Ask Price'], df['Close']).average_true_range()

        return df.dropna()
    
    except Exception as e:
        log_message(f"Error fetching data for {pair}: {e}")
        return None


def preprocess_data(df):
    df = df.dropna()
    features = df[['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA', 'ATR']].values
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

# Forex-specific function to convert pips to price movement
def pips_to_price(pips, price, pair):
    # Example conversion: In EUR/USD, 1 pip is 0.0001
    pip_value = 0.0001 if 'USD' in pair else 0.01  # Adjust for JPY or other currency pairs
    return pips * pip_value

def fixed_position_sizing(current_price, trade_amount=TRADE_AMOUNT):
    position_size = trade_amount / current_price
    return max(int(position_size), 1)

# Function to execute an order with pip-based stop-loss and take-profit
def execute_order(pair, position_size, current_price, reason):
    global allocated_budget

    # Calculate stop-loss and take-profit levels in terms of pips
    stop_loss_price = current_price - pips_to_price(STOP_LOSS_PIPS, current_price, pair)
    take_profit_price = current_price + pips_to_price(TAKE_PROFIT_PIPS, current_price, pair)

    # Log decision-making
    log_message(f"Decision: BUY {position_size} units of {pair}. Reason: {reason}")

    # Check if estimated order cost exceeds allocated budget
    estimated_order_cost = current_price * position_size
    if estimated_order_cost > allocated_budget:
        log_message(f"Cannot place order for {pair}. Cost ${estimated_order_cost:.2f} exceeds allocated budget ${allocated_budget:.2f}.")
        return False

    # Execute buy order using Alpaca API or other forex trading API
    try:
        # Placeholder: Replace with actual order submission
        log_message(f"Placed order for {position_size} units of {pair} at {current_price} with SL {stop_loss_price} and TP {take_profit_price}.")
        allocated_budget -= estimated_order_cost
        save_allocated_budget()  # Save new budget
        return True
    except Exception as e:
        log_message(f"Error placing order for {pair}: {e}")
        return False

# Function to apply forex trading logic
def forex_trading_logic(pair):
    global allocated_budget

    # Fetch forex data and preprocess
    forex_data = fetch_forex_data(pair)

    if forex_data is None:  # Check if data is None
        log_message(f"No data fetched for {pair}. Skipping trading logic.")
        return
    
    # Preprocess the data
    X, y = preprocess_data(forex_data)

    if len(X) == 0 or len(y) == 0:  # Ensure there is enough data for training
        log_message("Not enough data available for training. Skipping.")
        return

    # Train models
    rf_model, xgb_model, lstm_model = train_models(X, y)

    # Predict with models (similar to your stock trading approach)
    latest_features = X[-1].reshape(1, -1)  # Use the latest available data point
    rf_prediction = rf_model.predict(latest_features)
    xgb_prediction = xgb_model.predict(latest_features)
    lstm_prediction = lstm_model.predict(latest_features.reshape((1, latest_features.shape[1], 1)))
    lstm_prediction = np.round(lstm_prediction).astype(int).flatten()

    # Weighted voting for final prediction
    weights = [0.4, 0.4, 0.2]
    final_prediction = np.argmax(np.bincount([rf_prediction[0], xgb_prediction[0], lstm_prediction[0]], weights=weights))

    # Fetch the current price for the currency pair
    current_price = forex_data['Close'].iloc[-1]  # Use the latest closing price

    if final_prediction == 1:  # Buy Signal
        position_size = fixed_position_sizing(current_price)
        execute_order(pair, position_size, current_price, "Model predicted BUY")

# Main function to run the trading bot
def main():
    # Argument parsing for forex pair
    parser = argparse.ArgumentParser(description="Forex Trading Bot")
    parser.add_argument('--pair', type=str, required=True, help='Currency pair to trade (e.g., EUR/USD)')
    args = parser.parse_args()

    # Get the currency pair from arguments
    pair = args.pair

    # Schedule the trading logic to run periodically
    schedule.every(5).seconds.do(lambda: forex_trading_logic(pair))

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
