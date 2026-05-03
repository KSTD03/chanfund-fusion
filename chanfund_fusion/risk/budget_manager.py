"""
动态风险预算管理 — RiskBudgetManager
=============================
管理每日风险预算的分配、使用和释放。
"""

from __future__ import annotations

from datetime import date
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RiskBudgetManager:
    """动态风险预算管理器

    关键参数（来自config.yaml）：
        daily_risk_budget_pct: 日风险预算（总权益的%）
        risk_per_trade_pct: 单笔风险（总权益的%）
        single_stock_max_pct: 单只股票最大仓位
        sector_max_pct: 单行业最大暴露
    """

    def __init__(self, config: dict):
        self.config = config
        self.position_config = config.get("position", {})
        self.fusion_config = config.get("fusion", {})

        self.daily_budget_pct = self.fusion_config.get("daily_risk_budget_pct", 0.03)
        self.risk_per_trade = self.position_config.get("risk_per_trade_pct", 0.005)

        # 当前状态
        self.equity: float = 1_000_000.0  # 当前总权益
        self.daily_budget: float = 0.0    # 今日总预算
        self.used_budget: float = 0.0     # 已使用预算（比例形式）
        self.current_date: Optional[date] = None

    def new_day(self, trade_date: date, equity: float):
        """新的一天，重置预算

        Args:
            trade_date: 交易日
            equity: 当前总权益
        """
        self.current_date = trade_date
        self.equity = equity
        self.daily_budget = equity * self.daily_budget_pct
        self.used_budget = 0.0
        logger.debug(f"[Budget] New day {trade_date}: equity={equity:.2f}, "
                     f"budget={self.daily_budget:.2f}")

    def can_open(self, trade_date: date, target_weight: float) -> bool:
        """检查是否有足够预算开仓

        Args:
            trade_date: 交易日
            target_weight: 目标仓位比例（占总权益）

        Returns:
            can_open: 是否允许开仓
        """
        # 计算此次开仓所需预算
        required = target_weight * self.equity

        # 总预算检查
        if self.daily_budget <= 0:
            return False

        total_used = self.used_budget * self.equity + required
        if total_used > self.daily_budget:
            return False

        return True

    def allocate(self, trade_date: date, weight: float):
        """分配预算

        Args:
            trade_date: 交易日
            weight: 仓位比例
        """
        cost = weight * self.equity
        self.used_budget += weight
        logger.debug(f"[Budget] Allocated {cost:.2f} ({weight:.4%}), "
                     f"total used: {self.used_budget:.4%}")

    def release(self, trade_date: date, weight: float):
        """释放预算（平仓时调用）

        Args:
            trade_date: 交易日
            weight: 释放的仓位比例
        """
        self.used_budget = max(0.0, self.used_budget - weight)
        logger.debug(f"[Budget] Released {weight:.4%}, "
                     f"total used: {self.used_budget:.4%}")

    def is_exhausted(self) -> bool:
        """预算是否已耗尽"""
        return self.used_budget >= (self.daily_budget / self.equity if self.equity > 0 else 0)

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.daily_budget - self.used_budget * self.equity)

    def reset(self):
        self.equity = 1_000_000.0
        self.daily_budget = 0.0
        self.used_budget = 0.0
        self.current_date = None
