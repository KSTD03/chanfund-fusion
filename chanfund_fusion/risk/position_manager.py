"""
仓位计算、冷却期管理与保护性移动止损 — PositionManager
================================================
- 最终仓位计算（单只上限、行业上限）
- 冷却期管理（连续止损后暂停）
- 保护性移动止损（缠绕理论驱动）
- 动态风险预算仓位 = (权益 × risk_per_trade%) / |入场价 - 止损价|
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple
import logging
import numpy as np

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
        self.risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.01)

        # 止损参数
        self.stop_config = self.config.get("stop_loss", {})
        self.stop_mode = self.stop_config.get("mode", "trailing")
        self.initial_stop_atr_mult = self.stop_config.get("initial_stop_atr_mult", 3.0)  # v1.1: 从2.0放宽至3.0
        self.trail_bi_low = self.stop_config.get("trail_bi_low", True)
        self.trail_zs_zg = self.stop_config.get("trail_zs_zg", True)

        # 【v1.1】止盈参数
        self.tp_config = self.config.get("take_profit", {})
        self.tp1_pct = self.tp_config.get("tp1_pct", 0.10)            # +10% 第一止盈
        self.tp1_reduce = self.tp_config.get("tp1_reduce", 0.30)      # 减30%
        self.tp2_pct = self.tp_config.get("tp2_pct", 0.20)            # +20% 第二止盈
        self.tp2_reduce = self.tp_config.get("tp2_reduce", 0.50)      # 减50%
        self.tp_trail_retrace = self.tp_config.get("tp_trail_retrace", 0.90)  # 峰值回落10%清仓
        self.tp_breakeven_lock = self.tp_config.get("tp_breakeven_lock", 0.05) # +5%移损至成本

        # 冷却期参数
        self.cooling_period = self.fusion_config.get("cooling_period_kbars", 10)
        self.cooling_stop_count = self.fusion_config.get("cooling_stop_count", 2)

        # 【v1.1】长持仓时滞保护
        self.long_hold_days = self.tp_config.get("long_hold_days", 60)       # 持仓≥60天
        self.long_hold_cancel_stop = self.tp_config.get("long_hold_cancel_stop", True)  # 取消结构止损
        self.long_hold_wide_stop_pct = self.tp_config.get("long_hold_wide_stop_pct", 0.15)  # 改用15%宽止损

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
    # 仓位计算（含动态风险预算）
    # ------------------------------------------------------------------
    def calc_positions(
        self,
        fused_decisions: List[Tuple[str, float, str]],
        trade_date: date,
        entry_prices: Dict[str, float] = None,
        chan_stops: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """生成最终目标权重

        Phase 3 实现：动态风险预算仓位
        仓位(股数) = (总权益 × risk_per_trade_pct) / |入场价 - 止损价|
        再乘以信号类型系数

        Args:
            fused_decisions: [(symbol, signal_coeff, subtype)]
            trade_date: 交易日
            entry_prices: {symbol: entry_price} (可选)
            chan_stops: {symbol: stop_price} (可选)

        Returns:
            target_weights: {symbol: weight}
        """
        if entry_prices is None:
            entry_prices = {}
        if chan_stops is None:
            chan_stops = {}

        equity = self.budget_manager.equity
        risk_amount_per_trade = equity * self.risk_per_trade_pct

        targets: Dict[str, float] = {}

        for symbol, signal_coeff, subtype in fused_decisions:
            # 冷却期检查
            if symbol in self.cooling_symbols:
                if trade_date < self.cooling_symbols[symbol]:
                    continue
                else:
                    del self.cooling_symbols[symbol]

            # 获取入场价和止损价
            entry_price = entry_prices.get(symbol)
            stop_price = chan_stops.get(symbol)

            if entry_price is None or stop_price is None or entry_price == stop_price:
                # 无法计算动态仓位，使用信号系数直接计算
                raw_weight = signal_coeff * 0.02  # 默认2%标准仓位
            else:
                # 动态风险预算仓位计算
                price_risk = abs(entry_price - stop_price)
                risk_shares_value = risk_amount_per_trade / price_risk

                # 转化为权重
                raw_weight = (risk_shares_value * entry_price) / max(equity, 1)

                # 乘以信号类型系数
                raw_weight *= signal_coeff

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
    # 保护性移动止损（Phase 1 核心）
    # ------------------------------------------------------------------
    def calc_stop_price(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        entry_date: date,
        current_date: date,
        chan_state: dict = None,
    ) -> float:
        """计算保护性止损价（v1.0.1 优化）

        Task 1: 初始止损 = max(entry_price - ATR×2.0, latest_bi_low)
        确保入场即受结构保护，避免固定ATR止损在宽幅震荡中暴露过大风险。

        Phase 1 实现：
        - 移除固定5%止损
        - 趋势中：最近一个已确认笔的低点
        - 中枢向上离开后：中枢上沿 ZG
        - 初始止损：max(ATR×倍数, 最近笔低点)

        Args:
            symbol: 股票代码
            entry_price: 入场价格
            current_price: 当前价格
            entry_date: 入场日期
            current_date: 当前日期
            chan_state: 缠论结构状态（由技术引擎提供）
                       包含 latest_bi_low, latest_zs_zg, atr 等

        Returns:
            stop_price: 保护性止损价
        """
        if chan_state is None:
            chan_state = {}

        atr = chan_state.get("atr", 0.0)

        # --- Task 1: 初始止损 = max(ATR×倍数, 最近笔低点) ---
        latest_bi_low = chan_state.get("latest_bi_low")

        if atr > 0:
            atr_stop = entry_price - (atr * self.initial_stop_atr_mult)
        else:
            atr_stop = entry_price * 0.95  # 无ATR时5%后备

        if latest_bi_low is not None:
            # 取两者较大值，确保结构保护优先
            initial_stop = max(atr_stop, latest_bi_low)
        else:
            initial_stop = atr_stop

        # 检查是否已有持仓止损需要更新
        pos = self.holdings.get(symbol)
        if pos is None:
            return initial_stop

        current_stop = pos.get("stop_price", initial_stop)

        # --- 保护性移动止损逻辑 ---
        new_stop = current_stop

        if self.trail_bi_low:
            # 最近一笔的低点（向上趋势中）
            if latest_bi_low is not None and current_price > entry_price:
                # 价格正在上涨 → 移动止损至最近笔低点
                bi_stop = latest_bi_low
                if bi_stop > new_stop:
                    new_stop = bi_stop
                    logger.debug(
                        f"[{symbol}] Trail stop moved to bi_low={bi_stop:.3f}"
                    )

        if self.trail_zs_zg:
            # 中枢向上离开 → 移损至中枢上沿 ZG
            zs_zg = chan_state.get("latest_zs_zg")
            zs_zd = chan_state.get("latest_zs_zd")
            if zs_zg is not None and zs_zd is not None:
                if current_price > zs_zg:
                    # 价格已离开中枢上沿
                    zg_stop = zs_zg
                    if zg_stop > new_stop:
                        new_stop = zg_stop
                        logger.debug(
                            f"[{symbol}] Trail stop moved to zs_zg={zg_stop:.3f}"
                        )

        # --- 【v1.1】长持仓时滞保护 ---
        hold_days = (current_date - entry_date).days if isinstance(current_date, date) and isinstance(entry_date, date) else 0
        if hold_days >= self.long_hold_days and self.long_hold_cancel_stop:
            pnl_pct = (current_price - entry_price) / entry_price
            # 持仓超60天 且 接近保本（±5%）→ 放宽止损至15%
            if abs(pnl_pct) < 0.05:
                wide_stop = entry_price * (1 - self.long_hold_wide_stop_pct)
                if wide_stop < new_stop:  # 宽止损低于当前止损
                    new_stop = wide_stop
                    logger.debug(f"[{symbol}] Long-hold({hold_days}d) override: wide stop={wide_stop:.3f}")

        # --- 【v1.1】保本移损 ---
        if current_price >= entry_price * (1 + self.tp_breakeven_lock):
            if new_stop < entry_price:
                # 涨幅超过5%，止损移至成本
                new_stop = max(new_stop, entry_price)
                logger.debug(f"[{symbol}] Breakeven lock: stop moved to {entry_price:.3f}")

        # 止损只能上移（做多），不能下移
        if new_stop > current_stop:
            return new_stop
        return current_stop

    # ------------------------------------------------------------------
    # 【v1.1】多级止盈检查
    # ------------------------------------------------------------------
    def check_take_profit(self, symbol: str, current_price: float,
                          current_date: date) -> Optional[dict]:
        """多级止盈检查

        Returns:
            None / {"action": str, "reason": str, "reduce_pct": float}
        """
        pos = self.holdings.get(symbol)
        if not pos or pos.get("status") != "open":
            return None

        entry_price = pos["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price

        # 更新峰值
        peak = pos.get("peak_price", entry_price)
        if current_price > peak:
            pos["peak_price"] = current_price
            peak = current_price
        peak_pnl = (peak - entry_price) / entry_price

        # Level 3: 峰值回落止盈（已获利的持仓）
        if peak_pnl >= self.tp_breakeven_lock:
            retrace_ratio = current_price / peak
            if retrace_ratio <= self.tp_trail_retrace:
                return {
                    "action": "close",
                    "reason": f"峰值回落{(1-self.tp_trail_retrace)*100:.0f}%止盈(峰值={peak:.2f})",
                    "reduce_pct": 1.0,
                    "pnl_pct": round(pnl_pct * 100, 2),
                }

        # Level 2: 第二止盈 +20%
        if pnl_pct >= self.tp2_pct and not pos.get("tp2_triggered", False):
            pos["tp2_triggered"] = True
            return {
                "action": "reduce",
                "reason": f"第二止盈+{self.tp2_pct*100:.0f}%减{self.tp2_reduce*100:.0f}%",
                "reduce_pct": self.tp2_reduce,
                "pnl_pct": round(pnl_pct * 100, 2),
            }

        # Level 1: 第一止盈 +10%
        if pnl_pct >= self.tp1_pct and not pos.get("tp1_triggered", False):
            pos["tp1_triggered"] = True
            return {
                "action": "reduce",
                "reason": f"第一止盈+{self.tp1_pct*100:.0f}%减{self.tp1_reduce*100:.0f}%",
                "reduce_pct": self.tp1_reduce,
                "pnl_pct": round(pnl_pct * 100, 2),
            }

        return None

    def reduce_position(self, symbol: str, price: float, trade_date: date,
                        reason: str, reduce_pct: float) -> Optional[float]:
        """减仓（部分止盈）

        Args:
            symbol: 股票代码
            price: 减仓价格
            trade_date: 日期
            reason: 原因
            reduce_pct: 减仓比例 (0~1)

        Returns:
            realized_pnl: 已实现盈亏比例
        """
        pos = self.holdings.get(symbol)
        if not pos:
            return None

        original_weight = pos["weight"]
        reduce_weight = original_weight * reduce_pct
        remaining_weight = original_weight - reduce_weight

        # 更新持仓权重
        pos["weight"] = remaining_weight

        # 估算已实现盈亏
        realized_pnl = reduce_weight * (price - pos["entry_price"]) / pos["entry_price"]

        self.trade_log.append({
            "date": trade_date,
            "symbol": symbol,
            "action": "sell",
            "price": price,
            "weight": reduce_weight,
            "reason": reason,
            "pnl": realized_pnl,
            "trade_type": "take_profit",
        })

        # 释放部分预算
        self.budget_manager.release(trade_date, reduce_weight)

        logger.info(
            f"[{symbol}] Reduce: price={price:.3f}, reduce={reduce_pct*100:.0f}%, "
            f"weight: {original_weight:.4%} → {remaining_weight:.4%}, reason={reason}"
        )

        return realized_pnl

    # ------------------------------------------------------------------
    # 冷却期管理（Phase 2）
    # ------------------------------------------------------------------
    def record_stop_loss(self, symbol: str, trade_date: date):
        """记录止损事件

        Phase 2：同一只股票连续2次被止损 → 10个交易日冷却期
        """
        self.consecutive_stops[symbol] += 1
        count = self.consecutive_stops[symbol]

        if count >= self.cooling_stop_count:
            expiry = trade_date + timedelta(days=self.cooling_period)
            self.cooling_symbols[symbol] = expiry
            logger.info(
                f"[{symbol}] Cooling triggered: {count} consecutive stops, "
                f"cooling until {expiry} (停止期)"
            )
        else:
            logger.info(f"[{symbol}] Stop loss #{count} recorded")

    def is_in_cooling(self, symbol: str) -> bool:
        """检查是否处于冷却期"""
        return symbol in self.cooling_symbols

    def clear_cooling(self, symbol: str):
        """主动清除冷却期（冷却期结束后计数器清零）"""
        if symbol in self.cooling_symbols:
            del self.cooling_symbols[symbol]
        self.consecutive_stops[symbol] = 0

    # ------------------------------------------------------------------
    # 持仓管理
    # ------------------------------------------------------------------
    def open_position(self, symbol: str, price: float, weight: float,
                      trade_date: date, reason: str,
                      stop_price: float = None) -> dict:
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
            "stop_price": stop_price,
        }
        self.holdings[symbol] = position
        self.trade_log.append({
            "date": trade_date,
            "symbol": symbol,
            "action": "buy",
            "price": price,
            "weight": weight,
            "stop_price": stop_price,
            "reason": reason,
        })
        logger.info(
            f"[{symbol}] Opened: price={price:.3f}, weight={weight:.4%}, "
            f"stop={stop_price:.3f}"
        )
        return position

    def get_stop_price(self, symbol: str) -> Optional[float]:
        """获取当前保护性止损价"""
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
                       reason: str, skip_cooling: bool = False) -> Optional[dict]:
        """平仓

        Args:
            symbol: 股票代码
            price: 平仓价格
            trade_date: 日期
            reason: 原因
            skip_cooling: 是否跳过冷却期（止盈清仓时不触发冷却）
        """
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

        # Task 3: 盈利平仓 → 重置连续止损计数器
        if pos["pnl"] > 0:
            self.consecutive_stops[symbol] = 0
            logger.debug(f"[{symbol}] Profitable close: reset consecutive_stops to 0")

        # 止损则记录冷却期（止盈清仓不触发冷却）
        if reason == "stop_loss" and not skip_cooling:
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
