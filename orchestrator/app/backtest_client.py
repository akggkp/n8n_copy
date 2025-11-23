# orchestrator/app/backtest_client.py
# Client for interacting with backtesting service

import requests
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BacktestClient:
    """Client for backtesting trading strategies"""

    def __init__(self, backtest_url: str = "http://localhost:8001"):
        """
        Initialize backtesting client

        Args:
            backtest_url: Base URL for backtesting service
        """
        self.backtest_url = backtest_url

    def create_strategy(
        self,
        strategy_name: str,
        entry_rules: List[str],
        exit_rules: List[str],
        risk_params: Dict
    ) -> Optional[Dict]:
        """
        Create a new strategy configuration

        Args:
            strategy_name: Name of strategy
            entry_rules: List of entry condition strings
            exit_rules: List of exit condition strings
            risk_params: Dict with position_size, stop_loss, take_profit

        Returns:
            Strategy creation response with strategy_id
        """
        try:
            response = requests.post(
                f"{self.backtest_url}/strategies",
                json={
                    "name": strategy_name,
                    "entry_rules": entry_rules,
                    "exit_rules": exit_rules,
                    "risk_management": risk_params
                },
                timeout=30
            )

            if response.status_code != 201:
                logger.error(f"Strategy creation failed: {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            return None

    def run_backtest(
        self,
        strategy_id: int,
        symbol: str = "BTCUSDT",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = 10000.0
    ) -> Optional[Dict]:
        """
        Run backtest for a strategy

        Args:
            strategy_id: ID of strategy to backtest
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital in USD

        Returns:
            Backtest results with performance metrics
        """
        try:
            response = requests.post(
                f"{self.backtest_url}/backtest",
                json={
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "initial_capital": initial_capital
                },
                timeout=300  # 5 min timeout for backtest
            )

            if response.status_code != 200:
                logger.error(f"Backtest failed: {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return None

    def get_performance_metrics(self, backtest_id: int) -> Optional[Dict]:
        """
        Get detailed performance metrics for a backtest

        Args:
            backtest_id: ID of backtest run

        Returns:
            Performance metrics dict
        """
        try:
            response = requests.get(
                f"{self.backtest_url}/backtest/{backtest_id}/metrics",
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Failed to get metrics: {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return None

    def validate_strategy(
        self,
        performance_metrics: Dict,
        min_sharpe_ratio: float = 1.0,
        min_win_rate: float = 0.55,
        max_drawdown: float = 0.25
    ) -> bool:
        """
        Validate if strategy meets performance thresholds

        Args:
            performance_metrics: Dict with sharpe_ratio, win_rate, max_drawdown
            min_sharpe_ratio: Minimum Sharpe ratio (default: 1.0)
            min_win_rate: Minimum win rate (default: 55%)
            max_drawdown: Maximum acceptable drawdown (default: 25%)

        Returns:
            True if strategy passes all thresholds
        """
        try:
            sharpe = performance_metrics.get('sharpe_ratio', 0.0)
            win_rate = performance_metrics.get('win_rate', 0.0)
            drawdown = abs(performance_metrics.get('max_drawdown', 1.0))

            passed = (
                sharpe >= min_sharpe_ratio and
                win_rate >= min_win_rate and
                drawdown <= max_drawdown
            )

            if passed:
                logger.info(
                    f"Strategy validation PASSED: Sharpe={sharpe:.2f}, WinRate={win_rate:.2%}, Drawdown={drawdown:.2%}")
            else:
                logger.warning(
                    f"Strategy validation FAILED: Sharpe={sharpe:.2f}, WinRate={win_rate:.2%}, Drawdown={drawdown:.2%}")

            return passed

        except Exception as e:
            logger.error(f"Error validating strategy: {str(e)}")
            return False

    def health_check(self) -> bool:
        """Check if backtesting service is available"""
        try:
            response = requests.get(f"{self.backtest_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Backtest service health check failed: {str(e)}")
            return False
