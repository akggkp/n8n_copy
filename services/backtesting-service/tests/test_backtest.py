# services/backtesting-service/tests/test_backtest.py
# Unit tests for backtesting service

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test backtesting service health"""

    def test_health_check(self):
        """Health endpoint should return service info"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "strategies_count" in data
        assert "backtests_count" in data


class TestStrategyCreation:
    """Test strategy creation endpoint"""

    def test_create_valid_strategy(self):
        """Should create strategy with valid parameters"""
        response = client.post(
            "/strategies",
            json={
                "name": "Test_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert "strategy_id" in data
        assert data["name"] == "Test_Strategy"

    def test_create_strategy_missing_fields(self):
        """Should reject strategy with missing fields"""
        response = client.post(
            "/strategies",
            json={
                "name": "Incomplete_Strategy"
            }
        )
        assert response.status_code == 422  # Validation error

    def test_get_strategy(self):
        """Should retrieve created strategy"""
        # Create strategy first
        create_response = client.post(
            "/strategies",
            json={
                "name": "Test_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]

        # Get strategy
        response = client.get(f"/strategies/{strategy_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == strategy_id
        assert data["name"] == "Test_Strategy"


class TestBacktestExecution:
    """Test backtest execution"""

    def test_run_backtest(self):
        """Should execute backtest for valid strategy"""
        # Create strategy
        create_response = client.post(
            "/strategies",
            json={
                "name": "Backtest_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]

        # Run backtest
        response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "backtest_id" in data
        assert "metrics" in data
        assert data["strategy_id"] == strategy_id

        # Verify metrics structure
        metrics = data["metrics"]
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert "max_drawdown" in metrics
        assert "num_trades" in metrics

    def test_backtest_invalid_strategy(self):
        """Should reject backtest for non-existent strategy"""
        response = client.post(
            "/backtest",
            json={
                "strategy_id": 99999,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        assert response.status_code == 404


class TestBacktestResults:
    """Test backtest results endpoints"""

    def test_get_metrics(self):
        """Should retrieve backtest metrics"""
        # Create and run backtest
        create_response = client.post(
            "/strategies",
            json={
                "name": "Metrics_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]

        backtest_response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        backtest_id = backtest_response.json()["backtest_id"]

        # Get metrics
        response = client.get(f"/backtest/{backtest_id}/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "sharpe_ratio" in data
        assert "win_rate" in data

    def test_get_trades(self):
        """Should retrieve trade history"""
        # Create and run backtest
        create_response = client.post(
            "/strategies",
            json={
                "name": "Trades_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]

        backtest_response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        backtest_id = backtest_response.json()["backtest_id"]

        # Get trades
        response = client.get(f"/backtest/{backtest_id}/trades")
        assert response.status_code == 200

        data = response.json()
        assert "trades" in data
        assert "num_trades" in data

    def test_get_equity_curve(self):
        """Should retrieve equity curve"""
        # Create and run backtest
        create_response = client.post(
            "/strategies",
            json={
                "name": "Equity_Strategy",
                "entry_rules": ["RSI < 30"],
                "exit_rules": ["RSI > 70"],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.06
                }
            }
        )
        strategy_id = create_response.json()["strategy_id"]

        backtest_response = client.post(
            "/backtest",
            json={
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0
            }
        )
        backtest_id = backtest_response.json()["backtest_id"]

        # Get equity curve
        response = client.get(f"/backtest/{backtest_id}/equity")
        assert response.status_code == 200

        data = response.json()
        assert "equity_curve" in data
        assert "initial_capital" in data
        assert "final_capital" in data


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
