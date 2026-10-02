"""Interactive stock research and portfolio dashboard."""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from groq import Groq, GroqError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yfinance.exceptions import YFException

from finagent import (
    NEWS_TOPIC_KEYWORDS,
    PERIODS,
    classify_news_topic,
    fetch_fundamentals,
    fetch_history,
    moving_average_backtest,
    performance_metrics,
    portfolio_returns,
)

st.set_page_config(
    page_title="Market Lens | Stock Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #101c2a; border: 1px solid #24364a;
        padding: 16px 18px; border-radius: 12px;}
    [data-testid="stMetricLabel"] p {color: #a9bbcf;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(symbol: str, period: str) -> pd.DataFrame:
    return fetch_history(symbol, period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(symbol: str) -> dict[str, object]:
    return fetch_fundamentals(symbol)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(company: str, api_key: str) -> list[dict[str, object]]:
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": company,
            "sortBy": "publishedAt",
            "from": (date.today() - timedelta(days=30)).isoformat(),
            "pageSize": 10,
            "language": "en",
        },
        headers={"X-Api-Key": api_key},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("NewsAPI returned an unexpected response format.")
    if payload.get("status") != "ok":
        raise ValueError(payload.get("message", "NewsAPI returned an unsuccessful response."))
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("NewsAPI returned an invalid article list.")
    return articles


def configured_secret(name: str) -> str:
    """Read a secret from Streamlit Cloud/local secrets, then environment."""
    try:
        secret = st.secrets.get(name, "")
    except FileNotFoundError:
        secret = ""
    return str(secret or os.getenv(name, "")).strip()


def format_number(value: object, style: str = "number", currency: str = "") -> str:
    if value is None or pd.isna(value):
        return "Not reported"
    number = float(value)
    if style == "currency":
        prefix = f"{currency} " if currency else ""
        if abs(number) >= 1e12:
            return f"{prefix}{number / 1e12:,.2f}T"
        if abs(number) >= 1e9:
            return f"{prefix}{number / 1e9:,.2f}B"
        if abs(number) >= 1e6:
            return f"{prefix}{number / 1e6:,.2f}M"
        return f"{prefix}{number:,.2f}"
    if style == "percent":
        return f"{number:.2%}"
    return f"{number:,.2f}"


def chart_layout(fig: go.Figure, title: str, y_title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=440,
        margin=dict(l=15, r=15, t=65, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title=y_title, gridcolor="#24364a")
    fig.update_xaxes(gridcolor="#1b2a3a")
    return fig


st.title("📈 Market Lens")
st.caption(
    "Company fundamentals, market risk, transparent strategy testing, portfolio tracking, "
    "and optional news summaries."
)

with st.sidebar:
    st.header("Research settings")
    symbol = st.text_input("Primary ticker", value="AAPL", max_chars=12).strip().upper()
    selected_period = st.selectbox("History window", list(PERIODS), index=1)
    period = PERIODS[selected_period]
    benchmark = st.text_input("Benchmark ticker", value="SPY", max_chars=12).strip().upper()
    st.divider()
    st.caption("Yahoo Finance supplies market and fundamentals data. Provider availability may vary.")
    with st.expander("Optional integrations"):
        st.caption("For news and AI summaries, set keys in Streamlit Cloud → App settings → Secrets.")
        st.code(
            'NEWS_API_KEY = "your-newsapi-key"\n'
            'GROQ_API_KEY = "your-groq-key"\n'
            'GROQ_MODEL = "openai/gpt-oss-120b"',
            language="toml",
        )
        st.caption("You can also set NEWS_API_KEY and GROQ_API_KEY as environment variables locally.")

if not symbol:
    st.info("Enter a stock ticker in the sidebar to get started.")
    st.stop()

try:
    with st.spinner(f"Loading {symbol} market data…"):
        history = cached_history(symbol, period)
except (ValueError, YFException, requests.RequestException) as exc:
    st.error(f"Market data provider error for {symbol}: {exc}")
    st.stop()

latest_close = float(history["Close"].iloc[-1])
previous_close = float(history["Close"].iloc[-2]) if len(history) > 1 else latest_close
day_change = latest_close / previous_close - 1 if previous_close else 0.0
market_metrics = performance_metrics(history["Close"])
quote_cols = st.columns(4)
quote_cols[0].metric(f"{symbol} latest close", f"{latest_close:,.2f}", f"{day_change:.2%} daily")
quote_cols[1].metric("Period return", format_number(market_metrics["cumulative_return"], "percent"))
quote_cols[2].metric("Annualized volatility", format_number(market_metrics["annualized_volatility"], "percent"))
quote_cols[3].metric("Maximum drawdown", format_number(market_metrics["max_drawdown"], "percent"))
st.caption("Prices use the data provider's quoted currency; availability and currency vary by ticker.")

overview_tab, risk_tab, backtest_tab, portfolio_tab, news_tab = st.tabs(
    ["Company overview", "Risk & benchmark", "Strategy backtest", "Portfolio", "News & AI"]
)

with overview_tab:
    st.subheader("Price history")
    chart_data = history.copy()
    chart_data["MA50"] = chart_data["Close"].rolling(50).mean()
    chart_data["MA200"] = chart_data["Close"].rolling(200).mean()
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], name="Adjusted close"))
    price_fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MA50"], name="50-day average"))
    price_fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MA200"], name="200-day average"))
    st.plotly_chart(chart_layout(price_fig, f"{symbol} adjusted price"), use_container_width=True)

    volume_fig = go.Figure(go.Bar(x=history.index, y=history["Volume"], name="Volume"))
    st.plotly_chart(
        chart_layout(volume_fig, f"{symbol} daily trading volume", "Shares"),
        use_container_width=True,
    )

    st.subheader("Company fundamentals")
    try:
        fundamentals = cached_fundamentals(symbol)
    except (ValueError, YFException) as exc:
        st.warning(f"Company profile data is unavailable: {exc}")
    else:
        st.markdown(f"**{fundamentals['Company']}** · {fundamentals['Sector']}")
        cards = st.columns(5)
        currency = str(fundamentals["Currency"])
        cards[0].metric("Market cap", format_number(fundamentals["Market cap"], "currency", currency))
        cards[1].metric("P/E (trailing)", format_number(fundamentals["P/E (trailing)"]))
        cards[2].metric("P/E (forward)", format_number(fundamentals["P/E (forward)"]))
        cards[3].metric("Revenue growth", format_number(fundamentals["Revenue growth"], "percent"))
        cards[4].metric("Profit margin", format_number(fundamentals["Profit margin"], "percent"))
        cards = st.columns(2)
        cards[0].metric("Dividend yield", format_number(fundamentals["Dividend yield"], "percent"))
        cards[1].metric(
            "52-week range",
            f"{format_number(fundamentals['52-week low'], 'currency', currency)} – "
            f"{format_number(fundamentals['52-week high'], 'currency', currency)}",
        )
    st.caption("Fundamentals are reported by the data provider and may be delayed, incomplete, or unavailable.")

with risk_tab:
    st.subheader("Risk and benchmark comparison")
    try:
        if benchmark == symbol:
            raise ValueError("Choose a benchmark different from the primary ticker.")
        benchmark_history = cached_history(benchmark, period)
        aligned_prices = pd.concat(
            [history["Close"].rename(symbol), benchmark_history["Close"].rename(benchmark)],
            axis=1,
        ).dropna()
        if len(aligned_prices) < 2:
            raise ValueError("There is not enough overlapping history to compare these securities.")
        normalized = aligned_prices.div(aligned_prices.iloc[0]).mul(100)
        compare_fig = go.Figure()
        for ticker in normalized:
            compare_fig.add_trace(
                go.Scatter(x=normalized.index, y=normalized[ticker], name=ticker)
            )
        st.plotly_chart(
            chart_layout(compare_fig, "Relative performance (rebased to 100)", "Value"),
            use_container_width=True,
        )
        returns = aligned_prices.pct_change().dropna()
        beta = (
            float(returns[symbol].cov(returns[benchmark]) / returns[benchmark].var())
            if returns[benchmark].var() > 0
            else float("nan")
        )
        risk_stats = performance_metrics(aligned_prices[symbol])
        benchmark_stats = performance_metrics(aligned_prices[benchmark])
        columns = st.columns(5)
        columns[0].metric("Beta vs benchmark", format_number(beta))
        columns[1].metric("Stock Sharpe* ", format_number(risk_stats["sharpe_ratio"]))
        columns[2].metric("Benchmark Sharpe*", format_number(benchmark_stats["sharpe_ratio"]))
        columns[3].metric("Stock volatility", format_number(risk_stats["annualized_volatility"], "percent"))
        columns[4].metric("Benchmark volatility", format_number(benchmark_stats["annualized_volatility"], "percent"))
        st.caption("*Sharpe ratio uses a zero risk-free rate. Returns and risk use overlapping daily observations.")
    except (ValueError, YFException) as exc:
        st.error(f"Could not compare against {benchmark}: {exc}")

with backtest_tab:
    st.subheader("50/200-day moving-average crossover")
    st.write(
        "Long the stock when its fast average is above its slow average; otherwise hold cash. "
        "Signals take effect the following trading day to avoid look-ahead bias."
    )
    control_cols = st.columns(3)
    fast_window = control_cols[0].number_input("Fast moving average (days)", 5, 150, 50)
    slow_window = control_cols[1].number_input("Slow moving average (days)", 20, 300, 200)
    fee_percent = control_cols[2].number_input(
        "Transaction fee per trade (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05
    )
    try:
        if benchmark == symbol:
            raise ValueError("Choose a benchmark different from the primary ticker.")
        results = moving_average_backtest(
            history["Close"], int(fast_window), int(slow_window), fee_percent / 100
        )
        benchmark_history = cached_history(benchmark, period)
        comparison = results[["Strategy", "Buy and hold"]].join(
            benchmark_history["Close"].rename("Benchmark"), how="inner"
        ).dropna()
        if len(comparison) < 2:
            raise ValueError("There is not enough overlapping history for the benchmark comparison.")
        strategy_curve = comparison["Strategy"] / comparison["Strategy"].iloc[0]
        hold_curve = comparison["Buy and hold"] / comparison["Buy and hold"].iloc[0]
        benchmark_curve = comparison["Benchmark"] / comparison["Benchmark"].iloc[0]
        strategy_metrics = performance_metrics(strategy_curve)
        hold_metrics = performance_metrics(hold_curve)
        benchmark_backtest_metrics = performance_metrics(benchmark_curve)
    except (ValueError, YFException) as exc:
        st.warning(str(exc))
    else:
        result_cols = st.columns(5)
        result_cols[0].metric("Strategy total return", format_number(strategy_metrics["cumulative_return"], "percent"))
        result_cols[1].metric("Buy-and-hold return", format_number(hold_metrics["cumulative_return"], "percent"))
        result_cols[2].metric(f"{benchmark} return", format_number(benchmark_backtest_metrics["cumulative_return"], "percent"))
        result_cols[3].metric("Strategy max drawdown", format_number(strategy_metrics["max_drawdown"], "percent"))
        result_cols[4].metric("Strategy annualized volatility", format_number(strategy_metrics["annualized_volatility"], "percent"))
        backtest_fig = go.Figure()
        backtest_fig.add_trace(go.Scatter(x=comparison.index, y=strategy_curve * 100, name="Strategy"))
        backtest_fig.add_trace(
            go.Scatter(x=comparison.index, y=hold_curve * 100, name="Buy and hold")
        )
        backtest_fig.add_trace(
            go.Scatter(x=comparison.index, y=benchmark_curve * 100, name=benchmark)
        )
        st.plotly_chart(
            chart_layout(backtest_fig, "Strategy vs buy-and-hold vs benchmark (base 100)", "Index value"),
            use_container_width=True,
        )
        st.caption(
            "Historical simulation only; excludes taxes, slippage, survivorship bias, and dividends "
            "not represented by adjusted close. Results do not predict future performance."
        )

with portfolio_tab:
    st.subheader("Portfolio allocation and performance")
    default_tickers = f"{symbol}, MSFT, NVDA"
    raw_symbols = st.text_input(
        "Portfolio tickers (comma-separated)", value=default_tickers, key="portfolio_tickers"
    )
    portfolio_symbols = list(dict.fromkeys(
        ticker.strip().upper() for ticker in raw_symbols.split(",") if ticker.strip()
    ))
    if len(portfolio_symbols) > 10:
        st.error("Use no more than 10 tickers so the dashboard can load portfolio data reliably.")
        portfolio_symbols = []
    if not portfolio_symbols:
        st.info("Add at least one ticker to build a portfolio.")
    else:
        default_weight = 100.0 / len(portfolio_symbols)
        weight_cols = st.columns(min(len(portfolio_symbols), 4))
        weights: dict[str, float] = {}
        for index, ticker in enumerate(portfolio_symbols):
            with weight_cols[index % len(weight_cols)]:
                weights[ticker] = st.number_input(
                    f"{ticker} allocation (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(default_weight, 2),
                    step=1.0,
                    key=f"weight_{ticker}",
                )
        total_weight = sum(weights.values())
        st.caption(f"Entered allocation: {total_weight:.2f}%. Non-zero weights are normalized to 100%.")
        try:
            portfolio_histories = {
                ticker: cached_history(ticker, period) for ticker in portfolio_symbols
            }
            combined_returns = portfolio_returns(portfolio_histories, weights)
            portfolio_value = (1 + combined_returns).cumprod()
            portfolio_stats = performance_metrics(portfolio_value)
        except (ValueError, YFException) as exc:
            st.error(f"Could not calculate portfolio performance: {exc}")
        else:
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Portfolio return", format_number(portfolio_stats["cumulative_return"], "percent"))
            metrics_cols[1].metric("Annualized volatility", format_number(portfolio_stats["annualized_volatility"], "percent"))
            metrics_cols[2].metric("Maximum drawdown", format_number(portfolio_stats["max_drawdown"], "percent"))
            metrics_cols[3].metric("Sharpe ratio*", format_number(portfolio_stats["sharpe_ratio"]))
            portfolio_fig = go.Figure(
                go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value * 100,
                    name="Portfolio",
                    fill="tozeroy",
                )
            )
            st.plotly_chart(
                chart_layout(portfolio_fig, "Portfolio performance (base 100)", "Index value"),
                use_container_width=True,
            )
            normalized_weights = pd.Series(weights, dtype=float) / total_weight if total_weight else pd.Series(dtype=float)
            allocation_fig = go.Figure(
                go.Pie(
                    labels=list(normalized_weights.index),
                    values=list(normalized_weights.values),
                    hole=0.55,
                )
            )
            allocation_fig.update_layout(
                title="Normalized target allocation", template="plotly_dark", height=380
            )
            st.plotly_chart(allocation_fig, use_container_width=True)
            st.caption(
                "Equal weights are prefilled and editable. Daily-rebalanced, long-only model; "
                "historical illustration, not a live holdings or execution system. Sharpe assumes zero risk-free rate."
            )

with news_tab:
    st.subheader("Recent company news")
    news_company = st.text_input("Company or topic to search", value=symbol, key="news_query").strip()
    news_key = configured_secret("NEWS_API_KEY")
    if not news_key:
        st.info("Add NEWS_API_KEY in Streamlit Cloud Secrets (or your local environment) to enable NewsAPI.")
    elif st.button("Load recent news", type="primary"):
        try:
            with st.spinner("Fetching recent headlines…"):
                st.session_state["news_articles"] = fetch_news(news_company or symbol, news_key)
        except (requests.RequestException, ValueError) as exc:
            st.session_state["news_articles"] = []
            st.error(f"News request failed: {exc}")

    current_articles = st.session_state.get("news_articles", [])
    groq_key = configured_secret("GROQ_API_KEY")
    if current_articles:
        analyzer = SentimentIntensityAnalyzer()
        news_rows = []
        for article in current_articles:
            headline = str(article.get("title") or "Untitled article")
            description = str(article.get("description") or "")
            score = analyzer.polarity_scores(f"{headline}. {description}")["compound"]
            sentiment = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
            news_rows.append(
                {
                    "article": article,
                    "headline": headline,
                    "description": description,
                    "score": score,
                    "sentiment": sentiment,
                    "topic": classify_news_topic(f"{headline} {description}"),
                }
            )

        st.caption("Topic labels use simple headline keywords; sentiment scores are a rough text signal.")
        topic_choices = ["All topics", *NEWS_TOPIC_KEYWORDS.keys(), "Other"]
        selected_topic = st.selectbox("Group/filter headlines by topic", topic_choices)
        visible_rows = (
            news_rows
            if selected_topic == "All topics"
            else [row for row in news_rows if row["topic"] == selected_topic]
        )
        sentiment_counts = {
            sentiment: sum(row["sentiment"] == sentiment for row in visible_rows)
            for sentiment in ("Positive", "Neutral", "Negative")
        }
        sentiment_fig = go.Figure(
            go.Bar(
                x=list(sentiment_counts),
                y=list(sentiment_counts.values()),
                marker_color=["#43C6AC", "#8191a5", "#ef6a6a"],
            )
        )
        st.plotly_chart(
            chart_layout(sentiment_fig, "Headline sentiment counts", "Articles"),
            use_container_width=True,
        )
        for row in visible_rows:
            article = row["article"]
            published = str(article.get("publishedAt") or "")[:10]
            source = article.get("source", {})
            source_name = source.get("name", "Unknown source") if isinstance(source, dict) else "Unknown source"
            with st.container(border=True):
                st.markdown(f"**{row['headline']}**")
                st.caption(
                    f"{published} · {source_name} · {row['topic']} · "
                    f"{row['sentiment']} ({row['score']:+.2f})"
                )
                if row["description"]:
                    st.write(row["description"])
                url = article.get("url")
                if isinstance(url, str) and url.startswith("https://"):
                    st.link_button("Read article", url)

        st.divider()
        st.subheader("AI news briefing")
        if not groq_key:
            st.info("Add GROQ_API_KEY to Streamlit Cloud Secrets (or your local environment) to enable AI summaries.")
        elif st.button("Summarize headlines with Groq"):
            model_name = configured_secret("GROQ_MODEL") or "openai/gpt-oss-120b"
            briefing_items = [
                {
                    "title": row["headline"],
                    "description": row["description"],
                    "published": str(row["article"].get("publishedAt") or ""),
                }
                for row in visible_rows
            ]
            try:
                response = Groq(api_key=groq_key, timeout=30, max_retries=2).chat.completions.create(
                    model=model_name,
                    temperature=0.2,
                    max_tokens=500,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Summarize only the supplied financial news. Separate reported facts "
                                "from interpretation, mention conflicting/uncertain signals, and do "
                                "not forecast prices or give investment recommendations."
                            ),
                        },
                        {"role": "user", "content": str(briefing_items)},
                    ],
                )
            except GroqError as exc:
                st.error(f"Groq summary failed: {exc}")
            else:
                st.markdown(response.choices[0].message.content or "The model returned an empty summary.")

st.divider()
st.caption(
    "Market data can be delayed or incomplete. This dashboard is for education and research, "
    "not financial advice. Verify figures with official filings before making decisions."
)
