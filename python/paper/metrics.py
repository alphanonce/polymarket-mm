"""
Paper Trading Metrics

Calculates performance metrics: Sharpe ratio, max drawdown, win rate, etc.
"""

import math
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class MetricsState:
    """Current metrics state."""

    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_equity: float = 0.0
    current_drawdown: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades


class MetricsCalculator:
    """Calculates trading performance metrics."""

    def __init__(self, initial_equity: float = 10000.0, risk_free_rate: float = 0.0):
        self.initial_equity = initial_equity
        self.risk_free_rate = risk_free_rate

        self.state = MetricsState()
        self.state.max_equity = initial_equity

        # Track returns for Sharpe calculation
        self._equity_history: list[float] = [initial_equity]
        self._returns: list[float] = []
        self._pnl_history: list[float] = []

        self._logger = logger.bind(component="metrics_calculator")

    def record_trade(self, pnl: float) -> None:
        """Record a trade result."""
        self.state.total_trades += 1
        self._pnl_history.append(pnl)

        if pnl > 0:
            self.state.win_count += 1
        elif pnl < 0:
            self.state.loss_count += 1

    def update_equity(self, equity: float, realized_pnl: float, unrealized_pnl: float) -> None:
        """Update equity and recalculate metrics."""
        # Update PnL
        self.state.realized_pnl = realized_pnl
        self.state.unrealized_pnl = unrealized_pnl
        self.state.total_pnl = realized_pnl + unrealized_pnl

        # Update equity history
        if len(self._equity_history) > 0:
            prev_equity = self._equity_history[-1]
            if prev_equity > 0:
                ret = (equity - prev_equity) / prev_equity
                self._returns.append(ret)

        self._equity_history.append(equity)

        # Keep history bounded
        if len(self._equity_history) > 10000:
            self._equity_history = self._equity_history[-5000:]
            self._returns = self._returns[-5000:]

        # Update max equity and drawdown
        if equity > self.state.max_equity:
            self.state.max_equity = equity

        if self.state.max_equity > 0:
            self.state.current_drawdown = (self.state.max_equity - equity) / self.state.max_equity
            self.state.max_drawdown = max(self.state.max_drawdown, self.state.current_drawdown)

        # Recalculate Sharpe
        self._update_sharpe()

    def _update_sharpe(self) -> None:
        """Calculate annualized Sharpe ratio."""
        if len(self._returns) < 2:
            self.state.sharpe_ratio = 0.0
            return

        returns = self._returns

        # Calculate mean and std of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance) if variance > 0 else 0

        if std_return == 0:
            self.state.sharpe_ratio = 0.0
            return

        # Assume daily returns, annualize
        # For intraday (e.g., 15m markets), we have ~35,000 periods/year
        # Simplified: use periods per year = 252 * trading hours * periods/hour
        # For 15m markets: ~35,000 periods/year
        periods_per_year = 35000  # Approximate for 15m markets

        # Annualized Sharpe = (mean - risk_free) / std * sqrt(periods)
        excess_return = mean_return - (self.risk_free_rate / periods_per_year)
        self.state.sharpe_ratio = (excess_return / std_return) * math.sqrt(periods_per_year)

    def get_state(self) -> MetricsState:
        """Get current metrics state."""
        return self.state

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary as dict."""
        return {
            "total_pnl": self.state.total_pnl,
            "realized_pnl": self.state.realized_pnl,
            "unrealized_pnl": self.state.unrealized_pnl,
            "total_trades": self.state.total_trades,
            "win_count": self.state.win_count,
            "win_rate": self.state.win_rate,
            "sharpe_ratio": self.state.sharpe_ratio,
            "max_drawdown": self.state.max_drawdown,
            "current_drawdown": self.state.current_drawdown,
        }

    def reset(self, initial_equity: float | None = None) -> None:
        """Reset metrics."""
        if initial_equity is not None:
            self.initial_equity = initial_equity

        self.state = MetricsState()
        self.state.max_equity = self.initial_equity
        self._equity_history = [self.initial_equity]
        self._returns = []
        self._pnl_history = []
