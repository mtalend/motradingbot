import numpy as np
import pandas as pd
import yfinance as yf
import schedule
import time
from datetime import datetime, timedelta 
import pytz
import sys
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import alpaca_trade_api as tradeapi
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, EMAIndicator
import warnings
import json
import os
import logging
import argparse
import yfinance as yf
from ta.volatility import AverageTrueRange
from ta.volume import AccDistIndexIndicator
from ta.trend import PSARIndicator
import talib

#sh-3.2# cp /Users/mtalend/gitrepo/motradingbot/motradingbot/motradingbot.py .
#sh-3.2# python3 motradingbot.py --ticker AAPL


# Initialize Alpaca API client
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = 'PKJYJD4WF771JIO8C4YL'
APCA_API_SECRET_KEY = 'zjqZdbmNUqKvX0BDmrEZBje81pyTR4TyC72EJcgg'
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning, message='All PyTorch model weights were used')

# Trading parameters
STOP_LOSS_PERCENTAGE = 5.0  # 1% stop loss
TAKE_PROFIT_PERCENTAGE = 0.01  # 1% take profit
TRADE_AMOUNT = 200  # Fixed trade amount in USD
CHECK_INTERVAL = 15  # Interval to check and execute trading logic in seconds

#allocated_budget = 1000  # Allocated budget for trading

# Define trading hours (9:30 AM to 4:00 PM ET)
TRADING_START_TIME = timedelta(hours=9, minutes=30)
TRADING_END_TIME = timedelta(hours=16)

def check_market_hours():
    # Get the current time in Eastern Time (ET)
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern).time()

    # Convert the current time to a timedelta for comparison
    current_time_delta = timedelta(hours=current_time.hour, minutes=current_time.minute)

    # Check if the current time is outside of trading hours
    # if not (TRADING_START_TIME <= current_time_delta <= TRADING_END_TIME):
    if not (current_time_delta <= TRADING_END_TIME):
        print("Outside regular trading hours. Terminating script.")
        log_message("Script terminated: Outside regular trading hours.")
        sys.exit()  # Terminate the script

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
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['ROC'] = ROCIndicator(df['Close']).roc()
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['AccDist'] = AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    psar = PSARIndicator(df['High'], df['Low'], df['Close'])
    df['PSAR'] = psar.psar()
    
    # Ensure the columns used are consistent
    if not all(col in df.columns for col in ['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA', 'ATR', 'ROC', 'OBV', 'AccDist', 'PSAR']):
        print("Error: Missing required columns for feature extraction.")

    return df

def preprocess_data(df):
    df = df.dropna()
    features = df[['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA', 'ATR', 'ROC', 'OBV', 'AccDist', 'PSAR']].values
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
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['ROC'] = ROCIndicator(df['Close']).roc()
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['AccDist'] = AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    psar = PSARIndicator(df['High'], df['Low'], df['Close'])
    df['PSAR'] = psar.psar()

    # Save the stock data to a CSV file for future reference
    df.to_csv(f"{ticker}_sample_stock_data.csv")
    print(f"Downloaded and saved stock data for {ticker} to {ticker}_sample_stock_data.csv")
    return df

def load_sample_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        # Check if all required columns are present
        required_columns = ['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA', 'ATR', 'ROC', 'OBV', 'AccDist', 'PSAR']
        
        # Ensure the required columns exist in the CSV
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV data: {missing_columns}")
        
        # Extract features
        features = df[required_columns].values
        
        if features.shape[0] == 0:
            raise ValueError("The sample data from CSV is empty.")
        
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

BUDGET_FILE = 'allocated_budget.json'

# Define the maximum allocated budget
MAX_ALLOCATED_BUDGET = 1000.0
# Define the cash threshold and maximum amount to add
CASH_THRESHOLD = 25000
MAX_AMOUNT_TO_ADD = 1000

def save_allocated_budget(budget):
    """Save the allocated budget to a JSON file, ensuring it does not exceed the maximum limit."""
    budget = min(budget, MAX_ALLOCATED_BUDGET)  # Cap the budget to MAX_ALLOCATED_BUDGET
    with open(BUDGET_FILE, 'w') as f:
        json.dump({'allocated_budget': budget}, f)

def load_allocated_budget(default_budget):
    """Load the allocated budget from a JSON file, or use a default value if the file doesn't exist."""
    if os.path.exists(BUDGET_FILE):
        with open(BUDGET_FILE, 'r') as f:
            data = json.load(f)
            return min(data.get('allocated_budget', default_budget), MAX_ALLOCATED_BUDGET)
    return default_budget

# Initialize allocated budget with default or from saved file
default_budget = 1000  # Set your default trading budget here
allocated_budget = load_allocated_budget(default_budget)  # Allocated budget for trading
print(f"Allocated budget loaded: ${allocated_budget:.2f}")

def update_allocated_budget_with_excess(ticker):
    global allocated_budget

    try:
        # Fetch the account cash buying power (use only cash, not margin)
        try:
            account = api.get_account()
            cash_balance = float(account.cash)  # Get the cash balance as a float
        except Exception as e:
            print(f"Error fetching account details: {e}")
            return
        
        # Check if cash balance is above $25,000
        if cash_balance < CASH_THRESHOLD:
            print(f"Insufficient cash balance to trade. Current cash balance: ${cash_balance:.2f}")
            log_message(f"Trade not executed for {ticker}. Cash balance ${cash_balance:.2f} is below the $25,000 threshold.")
            return

        # Check if the cash balance exceeds the defined threshold
        if cash_balance > CASH_THRESHOLD:
            excess_amount = cash_balance - CASH_THRESHOLD

            # Cap the excess amount to be added
            #amount_to_add = min(excess_amount, MAX_AMOUNT_TO_ADD) # Delete this line,  if below line works better 
            amount_to_add = min(excess_amount, MAX_ALLOCATED_BUDGET - allocated_budget)

            # Update the allocated budget by adding the capped amount
            if allocated_budget + amount_to_add <= MAX_ALLOCATED_BUDGET:
                allocated_budget += amount_to_add
                log_message(f"Cash balance exceeded ${CASH_THRESHOLD}. Excess of ${amount_to_add:.2f} added to allocated budget.")
                log_message(f"Excess cash of ${amount_to_add:.2f} added to allocated budget. New budget: ${allocated_budget:.2f}")
                print(f"Excess of ${amount_to_add:.2f} added to allocated budget. New allocated budget: ${allocated_budget:.2f}")
            else:
                log_message(f"Adding excess amount would exceed max allocated budget. Keeping it at ${MAX_ALLOCATED_BUDGET:.2f}.")
                #allocated_budget = MAX_ALLOCATED_BUDGET # Delete this line,  if below line works better 

        # Save the updated allocated budget to a file
        save_allocated_budget(allocated_budget)

    except Exception as e:
        log_message(f"Error updating allocated budget with excess cash for {ticker}: {e}")
        print(f"Error updating allocated budget with excess cash for {ticker}: {e}")

# Function to execute an order with proper bracket pricing and budget check (Buy only, no short selling)
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

        # Fetch the account cash buying power (use only cash, not margin)
        account = api.get_account()
        cash_balance = float(account.cash)

        # Ensure that the cash balance does not drop below $25,000 after the buy order
        if cash_balance - estimated_order_cost < CASH_THRESHOLD:
            print(f"Cannot place buy order for {ticker}. Cash balance would drop below $25,000.")
            log_message(f"Buy order not placed for {ticker}. Cash balance would drop below $25,000.")
            return False
        
        # Submit the bracket order, using cash reserves only
        order = api.submit_order(
            symbol=ticker,
            qty=position_size,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=round(limit_price, 2),
            order_class='bracket',
            stop_loss={'stop_price': round(stop_price, 2)},
            take_profit={'limit_price': round(take_profit_price, 2)},
            extended_hours=False  # Ensure it's during regular hours
        )

        log_message(f"Placed buy order for {position_size} shares of {ticker} at limit price ${limit_price:.2f} with stop loss ${stop_price:.2f} and take profit ${take_profit_price:.2f}.")

        # Adjust the allocated budget **only after** the order is successfully placed
        allocated_budget -= estimated_order_cost
        log_message(f"New allocated budget after buy: ${allocated_budget:.2f}")

        # Save the updated allocated budget to a file
        save_allocated_budget(allocated_budget)
        log_message(f"Placed order for {ticker}, remaining budget: ${allocated_budget:.2f}")

        return True  # Order placed successfully

    except Exception as e:
        # Log the error and prevent budget changes if the order failed
        log_message(f"Error placing order for {ticker}: {e}")
        return False  # Order not placed

# Ensure eager execution
tf.config.run_functions_eagerly(True)

# Enable eager execution for tf.data functions specifically
tf.data.experimental.enable_debug_mode()

@tf.function(reduce_retracing=True)
def lstm_predict(model, data):
    # Check if 'data' is a symbolic tensor and convert to NumPy array if necessary
    if isinstance(data, tf.Tensor):
        data = data.numpy()  # Convert to NumPy array if it's a TensorFlow tensor
    elif not isinstance(data, np.ndarray):  # If not already a NumPy array
        data = np.array(data)  # Convert to NumPy array

    return model.predict(data)

# Function to apply trading logic
def trading_logic(ticker):
    global allocated_budget

    # Check if the current time is within trading hours
    check_market_hours()
    
    # Update the allocated budget if cash balance exceeds $25,000
    update_allocated_budget_with_excess(ticker)

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
    #below code is moved outside the trade logic 
    #lstm_prediction = np.round(lstm_model.predict(latest_features.reshape((latest_features.shape[0], latest_features.shape[1], 1)))).astype(int).flatten()

    # Ensure consistent input shape and convert to float32 for LSTM model predictions
    latest_features_lstm = latest_features.reshape((latest_features.shape[0], latest_features.shape[1], 1)).astype(np.float32)
    
    # Use the defined lstm_predict function to make predictions
    lstm_prediction = lstm_predict(lstm_model, latest_features_lstm)

    print(f"Model Predictions - RF: {rf_prediction}, XGB: {xgb_prediction}, LSTM: {lstm_prediction}")

    # Weighted voting for final prediction
    weights = [0.4, 0.4, 0.2]

    # Assuming rf_prediction, xgb_prediction, lstm_prediction, and weights are defined
    rf_value = rf_prediction[0] if np.isscalar(rf_prediction[0]) else rf_prediction[0].item()
    xgb_value = xgb_prediction[0] if np.isscalar(xgb_prediction[0]) else xgb_prediction[0].item()
    lstm_value = lstm_prediction[0] if np.isscalar(lstm_prediction[0]) else lstm_prediction[0].item()

    # weighted_predictions = np.bincount([rf_prediction[0], xgb_prediction[0], lstm_prediction[0]], weights=weights) 
    weighted_predictions = np.bincount(
        [rf_value, xgb_value, lstm_value],
        weights=weights
    )
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

    # Fetch the account cash buying power (use only cash, not margin)
    account = api.get_account()
    allocated_budget = float(account.cash)  # Use only the cash balance

    if allocated_budget < current_price * position_size:
        print("Not enough buying power with cash reserves.")
        return

    # Determine stop-loss and take-profit prices
    volatility = np.std(stock_data['Close'].tail(20))  # Calculate recent volatility
    if final_prediction == 1:  # Buy Signal
        stop_loss_price = current_price * (1 - STOP_LOSS_PERCENTAGE / 100 * volatility)
        take_profit_price = current_price * (1 + TAKE_PROFIT_PERCENTAGE / 100 * volatility)

        if take_profit_price <= stop_loss_price:
            print("Error: Take profit price must be greater than stop loss price.")
            return

        # Execute buy order using cash only
        if execute_order(ticker, position_size, round(current_price * 1.01, 2), stop_loss_price, take_profit_price, "Model predicted BUY"):
            print(f"New allocated budget after buy: {allocated_budget:.2f}")

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
    schedule.every(15).seconds.do(lambda: trading_logic(ticker))

    # Keep running the schedule
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
