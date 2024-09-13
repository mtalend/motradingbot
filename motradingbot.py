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

# Initialize Alpaca API client
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = 'PK09KDUAE68BYSEKGN7H'
APCA_API_SECRET_KEY = 'OVz8gLexQJb0DKgJPHWymr8i0s6by4hfdhWX3rjM'
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning, message='All PyTorch model weights were used')

# Trading parameters
STOP_LOSS_PERCENTAGE = 1  # 1% stop loss
TAKE_PROFIT_PERCENTAGE = 1  # 1% take profit
TRADE_AMOUNT = 100  # Fixed trade amount in USD
CHECK_INTERVAL = 15  # Interval to check and execute trading logic in seconds

allocated_budget = 500  # Allocated budget for trading

def fetch_stock_data(ticker, start_date='2020-01-01'):
    df = yf.download(ticker, start=start_date)
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['Stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
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

def fixed_position_sizing(current_price, trade_amount=TRADE_AMOUNT):
    """
    Determine the position size based on a fixed trade amount.
    """
    position_size = trade_amount // current_price
    return max(int(position_size), 1)

# Function to execute an order with proper bracket pricing and budget check
def execute_order(ticker, position_size, side, limit_price, stop_price, take_profit_price, is_short=False):
    global allocated_budget

    # Fetch account details and calculate buying power
    account = api.get_account()
    buying_power = float(account.cash)
    estimated_order_cost = limit_price * position_size

    # Ensure the trade is executed with cash only
    if estimated_order_cost > buying_power:
        print(f"Cannot place order for {ticker}. Estimated cost ${estimated_order_cost:.2f} exceeds available cash ${buying_power:.2f}.")
        return False  # Order not placed

    # Check if the order exceeds the allocated budget
    # Update the allocated budget after a successful trade
    if side == 'sell':  # This applies to both regular and short sells
        if is_short:  # Short Sale logic
            # Decrease allocated budget on short sell
            if estimated_order_cost > allocated_budget:
                print(f"Cannot short sell {ticker}. Estimated cost ${estimated_order_cost:.2f} exceeds allocated budget of ${allocated_budget:.2f}.")
                return False # Order not placed
            else:
                # For short sell, subtract the cost from allocated_budget
                allocated_budget -= estimated_order_cost
                print(f"New allocated budget after short sell: ${allocated_budget:.2f}")
        else:  # Regular Sell logic (closing a long position)
            # Increase allocated budget by the sale amount for regular sell
            allocated_budget += estimated_order_cost
            print(f"New allocated budget after regular sell: ${allocated_budget:.2f}")

    elif side == 'buy':  # This applies to both regular buy and buy to cover short
        if is_short:  # Buy to cover short position
            # Subtract the buy cost from allocated budget
            if estimated_order_cost > allocated_budget:
                print(f"Cannot cover short for {ticker}. Estimated cost ${estimated_order_cost:.2f} exceeds allocated budget of ${allocated_budget:.2f}.")
                return False
            else:
                allocated_budget -= estimated_order_cost
                print(f"New allocated budget after buying to cover short: ${allocated_budget:.2f}")
        else:  # Regular Buy logic
            # Subtract the cost of the buy from the allocated budget
            if estimated_order_cost > allocated_budget:
                print(f"Cannot buy {ticker}. Estimated cost ${estimated_order_cost:.2f} exceeds allocated budget of ${allocated_budget:.2f}.")
                return False
            else:
                allocated_budget -= estimated_order_cost
                print(f"New allocated budget after regular buy: ${allocated_budget:.2f}")


    try:
        # Adjust `take_profit_price` to ensure it meets Alpaca's requirements
        if side == 'buy':
            # Ensure `take_profit_price` is above `limit_price` and `stop_price` is below `limit_price`
            take_profit_price = max(take_profit_price, round(limit_price + 0.01, 2))
            stop_price = min(stop_price, round(limit_price - 0.01, 2))
        elif side == 'sell':
            # Ensure `take_profit_price` is below `limit_price` and `stop_price` is above `limit_price`
            take_profit_price = min(take_profit_price, round(limit_price - 0.01, 2))
            stop_price = max(stop_price, round(limit_price + 0.01, 2))

        if take_profit_price >= stop_price:
            print(f"Error: For a {side} order, take profit price must be {'less' if side == 'sell' else 'greater'} than stop loss price.")
            return False

        # Submit a bracket order
        order = api.submit_order(
            symbol=ticker,
            qty=position_size,
            side=side,
            type='limit',
            time_in_force='gtc',
            limit_price=round(limit_price, 2),
            order_class='bracket',
            stop_loss={'stop_price': round(stop_price, 2)},
            take_profit={'limit_price': round(take_profit_price, 2)}
        )
        print(f"Placed {side} order for {position_size} shares of {ticker} at limit price {limit_price} with stop loss {stop_price} and take profit {take_profit_price}.")
        return True  # Order placed successfully
    except Exception as e:
        print(f"Error placing order for {ticker}: {e}")
        return False  # Order not placed

# Function to apply trading logic
def trading_logic():
    global allocated_budget

    ticker = 'T'

    stock_data = fetch_stock_data(ticker)
    X, y = preprocess_data(stock_data)

    if len(X) == 0 or len(y) == 0:
        print("No data available for training.")
        return

    rf_model, xgb_model, lstm_model = train_models(X, y)

    sample_data_file = f'{ticker}_sample_stock_data.csv'
    latest_features = load_sample_data_from_csv(sample_data_file)

    if latest_features is None:
        print("No valid sample data available for prediction.")
        return

    latest_features = latest_features.reshape(-1, 6)
    latest_features_lstm = latest_features.reshape((latest_features.shape[0], latest_features.shape[1], 1))

    rf_prediction = rf_model.predict(latest_features)
    xgb_prediction = xgb_model.predict(latest_features)
    lstm_prediction = lstm_model.predict(latest_features_lstm)
    lstm_prediction = np.round(lstm_prediction).astype(int).flatten()

    predictions = [rf_prediction[0], xgb_prediction[0], lstm_prediction[0]]
    final_prediction = np.bincount(predictions).argmax()

    try:
        # Fetch the latest trade price
        last_trade = api.get_latest_trade(ticker)
        current_price = last_trade.price
    except Exception as e:
        print(f"Error fetching the current price for {ticker}: {e}")
        return

    position_size = fixed_position_sizing(current_price, trade_amount=TRADE_AMOUNT)
    account = api.get_account()
    buying_power = float(account.cash)

    if buying_power < current_price * position_size:
        print("Not enough buying power.")
        return

    # Check if the allocated budget is enough for the trade
    print(f"Checking allocated budget: ${allocated_budget:.2f}, Required: ${TRADE_AMOUNT:.2f}")
    if allocated_budget < TRADE_AMOUNT:
        print("Allocated budget exceeded. Cannot place more trades.")
        return

    # Determine the stop-loss and take-profit prices based on the type of trade
    if final_prediction == 1:  # Buy Signal
        stop_loss_price = current_price * (1 - STOP_LOSS_PERCENTAGE / 100)
        take_profit_price = current_price * (1 + TAKE_PROFIT_PERCENTAGE / 100)

        if take_profit_price <= stop_loss_price:
            print("Error: For a buy order, take profit price must be greater than stop loss price.")
            return

        # Execute buy order and update budget only if successful
        if execute_order(ticker, position_size, 'buy', round(current_price * 1.01, 2), stop_loss_price, take_profit_price):
            allocated_budget -= TRADE_AMOUNT
            print(f"New allocated budget after buy: {allocated_budget:.2f}")

    elif final_prediction == 0:  # Sell Signal
        stop_loss_price = current_price * (1 + STOP_LOSS_PERCENTAGE / 100)
        take_profit_price = current_price * (1 - TAKE_PROFIT_PERCENTAGE / 100)

        if take_profit_price >= stop_loss_price:
            print("Error: For a sell order, take profit price must be less than stop loss price.")
            return

        # Execute sell order and update budget only if successful
        if execute_order(ticker, position_size, 'sell', round(current_price * 0.99, 2), stop_loss_price, take_profit_price):
            allocated_budget -= TRADE_AMOUNT
            print(f"New allocated budget after sell: {allocated_budget:.2f}")

    # Handle all open orders to adjust or cancel as needed
    handle_open_orders()

def handle_open_orders():
    """
    Function to handle open orders (e.g., cancel, adjust).
    """
    open_orders = api.list_orders(status='open')
    for order in open_orders:
        print(f"Handling open order: {order.id}")
        # Logic to cancel or adjust open orders as needed
        # Example: api.cancel_order(order.id)

# Schedule trading logic to run every CHECK_INTERVAL seconds
schedule.every(CHECK_INTERVAL).seconds.do(trading_logic)

while True:
    schedule.run_pending()
    time.sleep(1)
