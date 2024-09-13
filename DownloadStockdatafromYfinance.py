import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator

def download_and_process_stock_data(ticker, start_date='2020-01-01', output_file='sample_stock_data.csv'):
    # Download historical stock data from Yahoo Finance
    df = yf.download(ticker, start=start_date)

    # Calculate RSI
    df['RSI'] = RSIIndicator(df['Close']).rsi()

    # Calculate Stochastic Oscillator
    stochastic = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stochastic'] = stochastic.stoch()

    # Calculate MACD and MACD Signal
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Calculate EMA
    df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()

    # Drop any rows with missing values
    df = df.dropna()

    # Select relevant columns for prediction
    df = df[['Close', 'RSI', 'Stochastic', 'MACD', 'MACD_Signal', 'EMA']]

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample stock data saved to {output_file}")

# Example usage for Apple stock (AAPL)
download_and_process_stock_data('T', output_file='T_sample_stock_data.csv')
