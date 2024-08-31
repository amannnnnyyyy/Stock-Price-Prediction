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

    def plot_stock_data(self, data, ax=None):
        close_prices = data['Close']
        sma_values = data['SMA']

        # If no ax is provided, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Close price
        ax.plot(data.index, close_prices, label='Close')

        # Plot SMA
        ax.plot(data.index, sma_values, label='SMA', linestyle='dashed')

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title(f'{self.ticker} Price SMA')

        # Add legend
        ax.legend()

        # Rotate x-axis labels for readability (optional)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Show the plot only if ax is None (for standalone plotting)
        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_rsi(self, data, ax=None):
        rsi_values = data['RSI']

        # Use the provided ax if given; otherwise, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot RSI
        ax.plot(data.index, rsi_values, label='RSI')

        # Set labels and title using self.ticker if available
        ax.set_xlabel('Date')
        ax.set_ylabel('RSI')
        ax.set_title(f'{self.ticker} RSI')

        # Add y-axis limits (optional)
        ax.set_ylim(0, 100)  # Assuming RSI values are between 0 and 100

        # Add legend
        ax.legend()

        # Rotate x-axis labels for readability (optional)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Show the plot only if ax is None (for standalone plotting)
        if ax is None:
            plt.tight_layout()
            plt.show()


    def plot_ema(self, data, ax=None):
        close_prices = data['Close']
        ema_values = data['EMA']

        # Use the provided ax if given; otherwise, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Close price
        ax.plot(data.index, close_prices, label='Close')

        # Plot EMA
        ax.plot(data.index, ema_values, label='EMA', linestyle='dashed')

        # Set labels and use stock name for the subplot title
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title(f'{self.ticker} Price with EMA')

        # Add legend
        ax.legend()

        # Rotate x-axis labels for readability (optional)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Adjust layout if ax is not provided (for standalone plotting)
        if ax is None:
            plt.tight_layout()
            plt.show()



    def plot_macd(self, data, ax=None):
        macd_values = data['MACD']
        macd_signal_values = data['MACD_Signal']

        # Use the provided ax if given; otherwise, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot MACD line (blue)
        ax.plot(data.index, macd_values, label='MACD', color='blue')

        # Plot MACD Signal line (green)
        ax.plot(data.index, macd_signal_values, label='MACD Signal', color='green')

        # Set labels and use stock name for the subplot title
        ax.set_xlabel('Date')
        ax.set_ylabel('MACD Value')
        ax.set_title(f'{self.ticker} MACD')

        # Add legend
        ax.legend()

        # Rotate x-axis labels for readability (optional)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Adjust layout if ax is not provided (for standalone plotting)
        if ax is None:
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