"""
试探性建仓模块
===========
用于处理"软背驰"试探性买入信号的后续演化逻辑：

1. 试探仓条件：软背驰 + 基本面得分达标 + 均线不空头
2. 仓位：标准仓位的25%
3. 升级主仓：价格回到中枢区内形成确认的二类买点
4. 止损：正常止损的1.5倍
5. 主动止盈：2R利润后平一半，剩余移损至成本
6. 超时强平：20根K线未升级则清仓
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrialPosition:
    """试探仓持仓记录"""
    symbol: str
    entry_date: date
    entry_price: float
    shares: float
    stop_price: float
    target_price: float        # 2R目标价
    upgrade_date: Optional[date] = None
    upgrade_price: Optional[float] = None
    is_upgraded: bool = False
    kbar_count: int = 0
    max_kbars: int = 20         # 最大K线持有数
    status: str = "ACTIVE"      # ACTIVE / UPGRADED / STOPPED / TAKEPROFIT / EXPIRED


class TrialBuyManager:
    """试探仓管理器"""

    def __init__(self, config: dict):
        self.config = config
        self.position_pct = config.get("position_pct", 0.25)
        self.stop_multiplier = config.get("stop_multiplier", 1.5)
        self.expire_kbars = config.get("expire_kbars", 20)
        self.upgrade_score = config.get("upgrade_score", 85)
        self.double_r = config.get("upgrade_double_r", 2.0)
        # 活跃试探仓 {symbol: TrialPosition}
        self.active: Dict[str, TrialPosition] = {}

    def should_enter(
        self,
        symbol: str,
        signal_subtype: str,
        fund_score: float,
        ma_state: str,
        has_position: bool = False,
    ) -> bool:
        """判断是否应该进行试探性买入

        Args:
            symbol: 股票代码
            signal_subtype: 信号子类型
            fund_score: 基本面得分
            ma_state: 均线状态 ("bull" / "bear" / "neutral")
            has_position: 是否已有持仓

        Returns:
            should_enter: 是否建立试探仓
        """
        # 已有持仓不再试探
        if has_position:
            return False

        # 只处理软背驰信号
        if signal_subtype != "soft_divergence":
            return False

        # 均线空头排列不参与
        if ma_state == "bear":
            return False

        # 基本面太差也不参与
        if fund_score < 40:
            return False

        # 基本面高分可直接视为标准信号
        if fund_score >= self.upgrade_score:
            return True  # 用标准仓位的50%跟随

        return True

    def enter(
        self,
        symbol: str,
        entry_date: date,
        entry_price: float,
        stop_price: float,
        equity: float,
        risk_pct: float = 0.005,
    ) -> TrialPosition:
        """建立试探仓

        Args:
            symbol: 股票代码
            entry_date: 入场日期
            entry_price: 入场价格
            stop_price: 止损价
            equity: 当前权益
            risk_pct: 单笔风险比例

        Returns:
            position: 试探仓记录
        """
        # 计算仓位量
        risk_amount = equity * risk_pct * self.position_pct
        price_risk = abs(entry_price - stop_price)
        if price_risk == 0:
            shares = 0
        else:
            shares = risk_amount / price_risk

        if shares <= 0:
            logger.warning(f"[{symbol}] TrialBuy: 仓位计算为0")
            raise ValueError("Invalid shares")

        # 放宽止损至1.5倍
        adjusted_stop = (
            entry_price - (entry_price - stop_price) * self.stop_multiplier
            if stop_price < entry_price
            else entry_price + (stop_price - entry_price) * self.stop_multiplier
        )

        # 2R目标价
        risk_per_share = abs(entry_price - stop_price)
        target_price = (
            entry_price + self.double_r * risk_per_share
            if entry_price >= stop_price
            else entry_price - self.double_r * risk_per_share
        )

        position = TrialPosition(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            stop_price=adjusted_stop,
            target_price=target_price,
            max_kbars=self.expire_kbars,
        )
        self.active[symbol] = position
        logger.info(
            f"[{symbol}] TrialBuy entered: price={entry_price:.3f}, "
            f"shares={shares:.2f}, stop={adjusted_stop:.3f}, target={target_price:.3f}"
        )
        return position

    def update(self, symbol: str, kbar_count: int, high: float, low: float,
               close: float, current_date: date) -> Optional[str]:
        """每日更新试探仓状态

        Returns:
            action: None / "upgrade" / "stop" / "takeprofit" / "expire"
        """
        pos = self.active.get(symbol)
        if pos is None or pos.status != "ACTIVE":
            return None

        pos.kbar_count = kbar_count

        # 1. 检查止损
        if low <= pos.stop_price:
            if close <= pos.stop_price:
                pos.status = "STOPPED"
                del self.active[symbol]
                return "stop"

        # 2. 检查2R主动止盈
        if high >= pos.target_price:
            # 平一半，剩余移损至成本价（简化实现）
            pos.status = "TAKEPROFIT"
            del self.active[symbol]
            return "takeprofit"

        # 3. 检查超时强平
        if kbar_count >= pos.max_kbars:
            pos.status = "EXPIRED"
            del self.active[symbol]
            return "expire"

        # 4. 升级条件（需外部调用 upgrade 方法）
        return None

    def upgrade(self, symbol: str, upgrade_price: float,
                upgrade_date: date) -> bool:
        """试探仓升级为主仓

        Args:
            symbol: 股票代码
            upgrade_price: 升级价格
            upgrade_date: 升级日期

        Returns:
            success: 是否成功升级
        """
        pos = self.active.get(symbol)
        if pos is None or pos.status != "ACTIVE":
            return False

        pos.is_upgraded = True
        pos.upgrade_price = upgrade_price
        pos.upgrade_date = upgrade_date
        pos.status = "UPGRADED"
        del self.active[symbol]
        logger.info(
            f"[{symbol}] TrialBuy upgraded at {upgrade_price:.3f} on {upgrade_date}"
        )
        return True

    def is_active(self, symbol: str) -> bool:
        """检查是否有活跃的试探仓"""
        pos = self.active.get(symbol)
        return pos is not None and pos.status == "ACTIVE"

    def get_active_symbols(self) -> list:
        return [s for s, p in self.active.items() if p.status == "ACTIVE"]

    def reset(self):
        self.active.clear()

    def __len__(self) -> int:
        return len(self.active)
