"""Market data and portfolio calculations for the FinAgent dashboard."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yfinance as yf


PERIODS = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}

NEWS_TOPIC_KEYWORDS = {
    "Earnings": ("earnings", "quarterly", "revenue", "profit", "guidance", "results"),
    "Products & technology": ("product", "launch", "technology", "artificial intelligence"),
    "M&A & deals": ("acquisition", "acquire", "merger", "deal"),
    "Regulation & legal": (
        "regulation",
        "regulator",
        "antitrust",
        "lawsuit",
        "court",
        "sec",
        "ftc",
    ),
    "Markets & analysts": ("stock", "shares", "market", "analyst", "price target", "rating"),
}


def classify_news_topic(text: str) -> str:
    """Assign a transparent keyword-based topic label to a headline."""
    normalized = text.casefold()
    for topic, keywords in NEWS_TOPIC_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in keywords):
            return topic
    return "Other"


def fetch_history(symbol: str, period: str) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV history, raising clear errors for bad data."""
    ticker = symbol.strip().upper()
    if not ticker:
        raise ValueError("Enter a stock ticker.")
    if period not in PERIODS.values():
        raise ValueError(f"Unsupported history period: {period}")

    history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history:
        raise ValueError(f"No price history was returned for {ticker}.")
    history = history.dropna(subset=["Close"])
    if len(history) < 2:
        raise ValueError(f"At least two price observations are required for {ticker}.")
    return history


def fetch_fundamentals(symbol: str) -> dict[str, object]:
    """Return available company fundamentals without treating missing metrics as zero."""
    ticker = symbol.strip().upper()
    if not ticker:
        raise ValueError("Enter a stock ticker.")

    info = yf.Ticker(ticker).get_info()
    if not info:
        raise ValueError(f"No company fundamentals were returned for {ticker}.")
    return {
        "Company": info.get("longName") or info.get("shortName") or ticker,
        "Sector": info.get("sector") or "Not reported",
        "Currency": info.get("currency") or "",
        "Market cap": info.get("marketCap"),
        "P/E (trailing)": info.get("trailingPE"),
        "P/E (forward)": info.get("forwardPE"),
        "Revenue growth": info.get("revenueGrowth"),
        "Profit margin": info.get("profitMargins"),
        "Dividend yield": info.get("dividendYield"),
        "52-week high": info.get("fiftyTwoWeekHigh"),
        "52-week low": info.get("fiftyTwoWeekLow"),
    }


def performance_metrics(prices: pd.Series) -> dict[str, float]:
    """Calculate cumulative return, annualized volatility, Sharpe, and drawdown."""
    clean_prices = prices.dropna()
    if len(clean_prices) < 2 or (clean_prices <= 0).any():
        raise ValueError("At least two positive price observations are needed.")

    returns = clean_prices.pct_change().dropna()
    cumulative_return = float(clean_prices.iloc[-1] / clean_prices.iloc[0] - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    daily_std = returns.std(ddof=1)
    sharpe = float(returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    drawdown = clean_prices / clean_prices.cummax() - 1
    return {
        "cumulative_return": cumulative_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def moving_average_backtest(
    prices: pd.Series,
    fast_window: int = 50,
    slow_window: int = 200,
    fee_rate: float = 0.001,
) -> pd.DataFrame:
    """Backtest long/cash moving-average crossover with lagged signals and fees."""
    if fast_window < 1 or slow_window <= fast_window:
        raise ValueError("Use positive windows with the slow window larger than the fast window.")
    if not 0 <= fee_rate < 1:
        raise ValueError("The transaction fee must be between 0% and 100%.")

    clean_prices = prices.dropna()
    if len(clean_prices) <= slow_window:
        raise ValueError(f"At least {slow_window + 1} daily observations are required.")

    daily_returns = clean_prices.pct_change().fillna(0.0)
    fast_average = clean_prices.rolling(fast_window, min_periods=fast_window).mean()
    slow_average = clean_prices.rolling(slow_window, min_periods=slow_window).mean()
    signal = (fast_average > slow_average).astype(float)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position)
    strategy_returns = position * daily_returns - turnover * fee_rate

    return pd.DataFrame(
        {
            "Strategy": (1 + strategy_returns).cumprod(),
            "Buy and hold": (1 + daily_returns).cumprod(),
            "Position": position,
        },
        index=clean_prices.index,
    )


def portfolio_returns(
    histories: dict[str, pd.DataFrame], weights: dict[str, float]
) -> pd.Series:
    """Combine daily security returns using the requested long-only target weights."""
    if not histories:
        raise ValueError("Select at least one ticker for the portfolio.")
    if set(histories) != set(weights):
        raise ValueError("Each portfolio ticker must have exactly one allocation.")

    raw_weights = pd.Series(weights, dtype=float)
    if raw_weights.isna().any() or (raw_weights < 0).any():
        raise ValueError("Portfolio allocations must be non-negative numbers.")
    total_weight = float(raw_weights.sum())
    if total_weight <= 0:
        raise ValueError("Portfolio allocations must sum to more than zero.")
    normalized_weights = raw_weights / total_weight

    close_prices = pd.concat(
        {symbol: history["Close"] for symbol, history in histories.items()},
        axis=1,
    ).dropna(how="any")
    if len(close_prices) < 2:
        raise ValueError("There is not enough overlapping portfolio price history.")
    returns = close_prices.pct_change().fillna(0.0)
    return returns.mul(normalized_weights, axis="columns").sum(axis=1)
