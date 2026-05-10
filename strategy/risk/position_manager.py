"""
仓位计算与冷却期管理 — PositionManager
===============================
- 最终仓位计算（单只上限、行业上限）
- 冷却期管理（连续止损后暂停）
- 卖出逻辑
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionManager:
    """仓位与冷却期管理器"""

    def __init__(self, config: dict, budget_manager):
        self.config = config.get("position", {})
        self.fusion_config = config.get("fusion", {})
        self.budget_manager = budget_manager

        # 仓位参数
        self.single_stock_max_pct = self.config.get("single_stock_max_pct", 0.08)
        self.sector_max_pct = self.config.get("sector_max_pct", 0.25)
        self.slippage = self.config.get("slippage", 0.0015)

        # 冷却期参数
        self.cooling_period = self.fusion_config.get("cooling_period_kbars", 10)
        self.cooling_stop_count = self.fusion_config.get("cooling_stop_count", 2)

        # 当前持仓 {symbol: holding_info}
        self.holdings: Dict[str, dict] = {}

        # 冷却期管理 {symbol: expiry_date}
        self.cooling_symbols: Dict[str, date] = {}

        # 连续止损计数 {symbol: consecutive_stops}
        self.consecutive_stops: Dict[str, int] = defaultdict(int)

        # 行业映射
        self.symbol_to_sector: Dict[str, str] = {}

        # 交易记录
        self.trade_log: List[dict] = []

    # ------------------------------------------------------------------
    # 仓位计算
    # ------------------------------------------------------------------
    def calc_positions(
        self,
        fused_decisions: List[Tuple[str, float, str]],
        trade_date: date,
    ) -> Dict[str, float]:
        """生成最终目标权重

        Args:
            fused_decisions: [(symbol, target_weight, reason)]
            trade_date: 交易日

        Returns:
            target_weights: {symbol: weight}
        """
        targets: Dict[str, float] = {}

        for symbol, raw_weight, reason in fused_decisions:
            if symbol in self.cooling_symbols:
                expiry = self.cooling_symbols[symbol]
                if trade_date < expiry:
                    continue
                else:
                    del self.cooling_symbols[symbol]

            # 单只上限
            weight = min(raw_weight, self.single_stock_max_pct)

            # 行业上限
            sector = self.symbol_to_sector.get(symbol)
            if sector:
                sector_exposure = self._get_sector_exposure(sector, targets)
                available = self.sector_max_pct - sector_exposure
                weight = min(weight, max(0.0, available))

            # 预算约束
            if not self.budget_manager.can_open(trade_date, weight):
                weight = 0.0

            if weight > 0:
                targets[symbol] = weight
                self.budget_manager.allocate(trade_date, weight)

        return targets

    def _get_sector_exposure(self, sector: str, new_weights: dict) -> float:
        """计算某个行业的总暴露"""
        exposure = 0.0
        for sym, info in self.holdings.items():
            if self.symbol_to_sector.get(sym) == sector:
                exposure += info.get("weight", 0.0)
        for sym, w in new_weights.items():
            if self.symbol_to_sector.get(sym) == sector:
                exposure += w
        return exposure

    def exceeds_sector_limit(self, symbol: str, weight: float) -> bool:
        """检查新开仓是否会超过行业上限"""
        sector = self.symbol_to_sector.get(symbol)
        if not sector:
            return False
        current = 0.0
        for sym, info in self.holdings.items():
            if self.symbol_to_sector.get(sym) == sector:
                current += info.get("weight", 0.0)
        return (current + weight) > self.sector_max_pct

    # ------------------------------------------------------------------
    # 冷却期管理
    # ------------------------------------------------------------------
    def record_stop_loss(self, symbol: str, trade_date: date):
        """记录止损事件

        当连续止损次数达到阈值，触发冷却期
        """
        self.consecutive_stops[symbol] += 1
        count = self.consecutive_stops[symbol]

        if count >= self.cooling_stop_count:
            from datetime import timedelta
            expiry = trade_date + timedelta(days=self.cooling_period)
            self.cooling_symbols[symbol] = expiry
            logger.info(
                f"[{symbol}] Cooling triggered: {count} consecutive stops, "
                f"cooling until {expiry}"
            )
        else:
            logger.info(f"[{symbol}] Stop loss #{count} recorded")

    def is_in_cooling(self, symbol: str) -> bool:
        """检查是否处于冷却期"""
        return symbol in self.cooling_symbols

    def clear_cooling(self, symbol: str):
        """主动清除冷却期"""
        if symbol in self.cooling_symbols:
            del self.cooling_symbols[symbol]
        self.consecutive_stops[symbol] = 0

    # ------------------------------------------------------------------
    # 持仓管理
    # ------------------------------------------------------------------
    def open_position(self, symbol: str, price: float, weight: float,
                      trade_date: date, reason: str) -> dict:
        """开仓记录"""
        position = {
            "symbol": symbol,
            "entry_date": trade_date,
            "entry_price": price,
            "weight": weight,
            "shares": 0,
            "current_price": price,
            "pnl": 0.0,
            "reason": reason,
            "status": "open",
        }
        self.holdings[symbol] = position
        self.trade_log.append({
            "date": trade_date,
            "symbol": symbol,
            "action": "buy",
            "price": price,
            "weight": weight,
            "reason": reason,
        })
        logger.info(f"[{symbol}] Opened: price={price:.3f}, weight={weight:.4%}")
        return position

    def get_stop_price(self, symbol: str, current_price: float) -> Optional[float]:
        """获取保护性止损价

        原则（来自顶层设计V3.0 2.5节）：
        - 上涨趋势中：最近一个已确认笔的低点
        - 若中枢形成后上涨离开：中枢上沿ZG
        - 需由技术引擎传入结构信息
        """
        pos = self.holdings.get(symbol)
        if pos is None:
            return None
        return pos.get("stop_price")

    def set_stop_price(self, symbol: str, stop_price: float):
        """设置（更新）止损价"""
        pos = self.holdings.get(symbol)
        if pos:
            pos["stop_price"] = stop_price

    def close_position(self, symbol: str, price: float, trade_date: date,
                       reason: str) -> Optional[dict]:
        """平仓"""
        pos = self.holdings.pop(symbol, None)
        if pos is None:
            return None

        pos["exit_date"] = trade_date
        pos["exit_price"] = price
        pos["pnl"] = (price - pos["entry_price"]) / pos["entry_price"]
        pos["status"] = "closed"

        self.trade_log.append({
            "date": trade_date,
            "symbol": symbol,
            "action": "sell",
            "price": price,
            "weight": pos["weight"],
            "reason": reason,
            "pnl": pos["pnl"],
        })

        # 释放预算
        self.budget_manager.release(trade_date, pos["weight"])

        logger.info(
            f"[{symbol}] Closed: exit={price:.3f}, pnl={pos['pnl']:.2%}, reason={reason}"
        )

        # 止损则记录
        if reason == "stop_loss":
            self.record_stop_loss(symbol, trade_date)

        return pos

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_holding_weight(self, symbol: str) -> float:
        """获取某只股票的持仓权重"""
        pos = self.holdings.get(symbol)
        return pos["weight"] if pos else 0.0

    def get_holdings(self) -> List[dict]:
        return list(self.holdings.values())

    def total_exposure(self) -> float:
        return sum(pos["weight"] for pos in self.holdings.values())

    def has_position(self, symbol: str) -> bool:
        return symbol in self.holdings

    def set_sector_map(self, sector_map: Dict[str, str]):
        self.symbol_to_sector = sector_map

    def get_trade_log(self) -> List[dict]:
        return self.trade_log

    def reset(self):
        self.holdings.clear()
        self.cooling_symbols.clear()
        self.consecutive_stops.clear()
        self.trade_log.clear()
