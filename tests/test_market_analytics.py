import pandas as pd
import unittest

from finagent import (
    classify_news_topic,
    moving_average_backtest,
    performance_metrics,
    portfolio_returns,
)


class MarketAnalyticsTests(unittest.TestCase):
    def test_performance_metrics_for_constant_growth(self):
        prices = pd.Series([100.0, 110.0, 121.0])
        metrics = performance_metrics(prices)
        self.assertAlmostEqual(metrics["cumulative_return"], 0.21)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.0)
        self.assertAlmostEqual(metrics["annualized_volatility"], 0.0)
        self.assertAlmostEqual(metrics["sharpe_ratio"], 0.0)

    def test_moving_average_backtest_lags_signal_and_subtracts_cost(self):
        index = pd.date_range("2025-01-01", periods=8, freq="D")
        prices = pd.Series([10, 9, 8, 7, 6, 7, 8, 9], index=index, dtype=float)
        result = moving_average_backtest(prices, fast_window=2, slow_window=3, fee_rate=0.01)
        self.assertAlmostEqual(result.loc[index[0], "Strategy"], 1.0)
        self.assertAlmostEqual(result.loc[index[7], "Strategy"], 1.115)
        self.assertTrue(result["Strategy"].notna().all())
        self.assertTrue(result["Position"].isin([0.0, 1.0]).all())

    def test_backtest_rejects_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "observations"):
            moving_average_backtest(pd.Series(range(10, 20), dtype=float), 2, 5)

    def test_portfolio_normalizes_weights_and_includes_initial_value(self):
        index = pd.date_range("2025-01-01", periods=3, freq="D")
        histories = {
            "AAA": pd.DataFrame({"Close": [100.0, 110.0, 121.0]}, index=index),
            "BBB": pd.DataFrame({"Close": [100.0, 100.0, 100.0]}, index=index),
        }
        result = portfolio_returns(histories, {"AAA": 1.0, "BBB": 1.0})
        expected = pd.Series([0.0, 0.05, 0.05], index=index, dtype=float)
        pd.testing.assert_series_equal(result, expected)

    def test_portfolio_rejects_zero_weights(self):
        histories = {"AAA": pd.DataFrame({"Close": [10.0, 11.0]})}
        with self.assertRaisesRegex(ValueError, "more than zero"):
            portfolio_returns(histories, {"AAA": 0.0})

    def test_news_topics_use_keyword_boundaries(self):
        self.assertEqual(classify_news_topic("Company reports quarterly earnings"), "Earnings")
        self.assertEqual(classify_news_topic("A growing sector adds jobs"), "Other")


if __name__ == "__main__":
    unittest.main()
