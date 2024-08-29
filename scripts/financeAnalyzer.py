import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


class FinancialAnalyzer:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def retrieve_stock_data(self):
        return yf.download(self.ticker, start=self.start_date, end=self.end_date)

    def calculate_moving_average(self, data, window_size):
        return ta.SMA(data, timeperiod=window_size)

    def calculate_technical_indicators(self, data):
        # Calculate various technical indicators
        data['SMA'] = self.calculate_moving_average(data['Close'], 20)
        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
        data['EMA'] = ta.EMA(data['Close'], timeperiod=20)
        macd, macd_signal, _ = ta.MACD(data['Close'])
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        # Add more indicators as needed
        return data

    def plot_stock_data(self, data):
        close_prices = data['Close']
        sma_values = data['SMA']

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Close price
        ax.plot(data.index, close_prices, label='Close')

        # Plot SMA
        ax.plot(data.index, sma_values, label='SMA', linestyle='dashed')

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title('Stock Price with Moving Average')

        # Add legend
        ax.legend()

        # Rotate x-axis labels for readability (optional)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_rsi(self, data):
        rsi_values = data['RSI']

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot RSI
        ax.plot(data.index, rsi_values, label='RSI')

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('RSI')
        ax.set_title('Relative Strength Index (RSI)')

        # Add y-axis limits (optional)
        ax.set_ylim(0, 100)  # Assuming RSI values are between 0 and 100

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_ema(self, data):
        close_prices = data['Close']
        ema_values = data['EMA']

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Close price
        ax.plot(data.index, close_prices, label='Close')

        # Plot EMA
        ax.plot(data.index, ema_values, label='EMA', linestyle='dashed')

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title('Stock Price with Exponential Moving Average')

        # Add legend
        ax.legend()

        # Rotate x-axis labels for readability (optional)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_macd(self, data):
        macd_values = data['MACD']
        macd_signal_values = data['MACD_Signal']

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot MACD line (blue)
        ax.plot(data.index, macd_values, label='MACD', color='blue')

        # Plot MACD Signal line (green)
        ax.plot(data.index, macd_signal_values, label='MACD Signal', color='green')

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('MACD Value')
        ax.set_title('Moving Average Convergence Divergence (MACD)')

        # Add legend
        ax.legend()

        # Rotate x-axis labels for readability (optional)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def calculate_portfolio_weights(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        weights = dict(zip(tickers, weights.values()))
        return weights

    def calculate_portfolio_performance(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
        return portfolio_return, portfolio_volatility, sharpe_ratio