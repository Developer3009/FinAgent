# Market Lens

Market Lens is a Streamlit stock research dashboard. It combines Yahoo Finance market data and company fundamentals with risk/benchmark comparisons, a transparent moving-average backtest, portfolio allocation/performance, and optional NewsAPI headlines with Groq-powered summaries.

It is an educational research tool, not investment advice. It does not predict future prices or place trades.

## Run locally

Use Python 3.10 or newer. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Yahoo Finance price data and company fundamentals do not require an API key. News and AI summaries are optional.

To enable optional features, set keys in the terminal before starting Streamlit:

```powershell
$env:NEWS_API_KEY = "your-newsapi-key"
$env:GROQ_API_KEY = "your-groq-key"
streamlit run app.py
```

Alternatively, create `.streamlit/secrets.toml` locally (it is git-ignored):

```toml
NEWS_API_KEY = "your-newsapi-key"
GROQ_API_KEY = "your-groq-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Never commit real API keys or paste them into source code. The optional NewsAPI key enables headlines and sentiment; the optional Groq key enables the AI news briefing.

## Dashboard features

- Company profile, market cap, valuation, revenue growth, profitability, dividend yield, and 52-week range.
- Adjusted close and volume charts with 50- and 200-day moving averages.
- Return, annualized volatility, maximum drawdown, Sharpe ratio (zero risk-free rate), and beta versus an editable benchmark.
- Long/cash fast/slow moving-average crossover backtest. Positions use prior-day signals; transaction fees are configurable. Backtests are historical illustrations, not predictions.
- Multi-ticker portfolio with equal-weight defaults, editable target allocations, and normalized historical return/risk.
- Up to 10 recent NewsAPI headlines with VADER sentiment, publication date, source, and article links; optional Groq summary grounded only in retrieved headlines.
- Short-lived caching for prices, fundamentals, and news.

## Run tests

From the repository root:

```powershell
python -m unittest discover -s tests -v
```

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository you control. `requirements.txt` and `app.py` are at the repository root.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Before or after deployment, open **App settings → Secrets** and add the keys in TOML format:

   ```toml
   NEWS_API_KEY = "your-newsapi-key"
   GROQ_API_KEY = "your-groq-key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```

   Omit either optional key to disable its feature. Never commit a `secrets.toml` containing real credentials.
5. Deploy the app. Community Cloud installs dependencies from `requirements.txt`; future commits to the selected branch trigger redeployment.

The workspace has no Git remote configured, so deployment cannot be published directly from this checkout until you connect or push it to GitHub.
