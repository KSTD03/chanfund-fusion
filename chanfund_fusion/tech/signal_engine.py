"""
技术信号引擎 — TechSignalEngine
===========================
核心调度器：管理所有股票的增量缠论计算，
输出已确认的技术信号（T+1生效）。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple
import logging

from .chan_objects import Signal, SignalStatus
from .chan_structure import ChanStructureState, BiConfirmedEvent, ZSUpdatedEvent, SignalGenEvent
from .noise import NoiseEstimator, ActiveStockFilter

logger = logging.getLogger(__name__)


class TechSignalEngine:
    """技术信号引擎

    职责：
    1. 管理全市场股票的 ChanStructureState
    2. 每根新K线增量更新
    3. 管理 pending_signals（待执行信号）列表
    4. 主动失效被后续结构破坏的信号
    5. 预热（不生成信号）
    """

    def __init__(self, config: dict):
        self.config = config
        # {symbol: ChanStructureState}
        self.chan_cache: Dict[str, ChanStructureState] = {}
        # {symbol: [Signal]}
        self.pending_signals: Dict[str, List[Signal]] = defaultdict(list)
        # 当日生成的信号（未确认）
        self._todays_new_signals: List[Signal] = []
        # {symbol: NoiseEstimator}
        self.noise_estimators: Dict[str, NoiseEstimator] = {}
        # 当前交易日（由外部设置）
        self.current_trade_date: Optional[date] = None

        # 30分钟级别活跃股票筛选器
        self.active_filter = ActiveStockFilter(
            max_active=config.get("active_max", 300),
            top_fund_count=config.get("active_top_fund", 100),
        )

    # ------------------------------------------------------------------
    # 核心更新方法
    # ------------------------------------------------------------------
    def update(self, symbol: str, kbar: dict, emit_signals: bool = True) -> None:
        """增量处理一根新K线

        Args:
            symbol: 股票代码
            kbar: K线数据 {open, high, low, close, volume, time, ...}
            emit_signals: 是否生成信号（预热=False）
        """
        # 获取或创建状态
        state = self.chan_cache.setdefault(symbol, ChanStructureState(symbol))

        # 处理K线，获取事件
        events = state.process_kbar(kbar, emit_signals=emit_signals)

        # 处理事件
        new_tech_signals = []
        for event in events:
            if isinstance(event, SignalGenEvent):
                sig = event.signal
                sig.confirmed_time = self._next_trade_date(kbar)
                sig.status = SignalStatus.PENDING
                new_tech_signals.append(sig)

            elif isinstance(event, BiConfirmedEvent) or isinstance(event, ZSUpdatedEvent):
                # 新结构事件可能导致pending信号失效
                self._invalidate_pending_signals(symbol, event)

        # 加入pending队列
        for sig in new_tech_signals:
            self.pending_signals[symbol].append(sig)
            self._todays_new_signals.append(sig)

        # 更新噪声估计（使用收盘价序列）
        if "close" in kbar:
            self._update_noise(symbol, kbar["close"])

    def _invalidate_pending_signals(self, symbol: str, event) -> None:
        """Task 6: 基于结构变化的信号失效机制

        检查条件：
        1. 信号依赖的笔→新笔破坏了该笔的极值（笔低点被跌破）
        2. 信号依赖的中枢→中枢状态变为DESTROYED

        Args:
            symbol: 股票代码
            event: 新触发的结构事件 (BiConfirmedEvent | ZSUpdatedEvent)

        注意：失效的信号仍保留在 pending_signals 列表中但标记为 INVALIDATED，
        get_confirmed_signals() 会跳过非 PENDING 状态的信号，不影响后续流程。
        """
        if symbol not in self.pending_signals:
            return

        state = self.chan_cache.get(symbol)

        for sig in self.pending_signals[symbol]:
            if sig.status != SignalStatus.PENDING:
                continue

            details = sig.details or {}

            # --- 检查1: 信号依赖的笔被破坏 ---
            ref_bi_low = details.get("ref_bi_low")
            if ref_bi_low is not None and state is not None:
                latest_bi = state.get_latest_bi()
                if latest_bi is not None and latest_bi.low < ref_bi_low:
                    sig.status = SignalStatus.INVALIDATED
                    logger.debug(
                        f"[{symbol}] Signal {sig.signal_subtype} invalidated: "
                        f"bi low {ref_bi_low:.2f} broken by {latest_bi.low:.2f}"
                    )
                    continue

            # --- 检查2: 信号依赖的中枢被破坏 ---
            ref_zs_zg = details.get("ref_zs_zg")
            if ref_zs_zg is not None and state is not None:
                latest_zs = state.get_latest_zs()
                if latest_zs is not None and latest_zs.status == "DESTROYED":
                    sig.status = SignalStatus.INVALIDATED
                    logger.debug(
                        f"[{symbol}] Signal {sig.signal_subtype} invalidated: "
                        f"referenced ZS destroyed"
                    )

    def _update_noise(self, symbol: str, close_price: float) -> None:
        """更新噪声估计"""
        if symbol not in self.noise_estimators:
            self.noise_estimators[symbol] = NoiseEstimator(
                er_period=self.config.get("noise", {}).get("er_period", 20),
                hist_window=self.config.get("noise", {}).get("er_hist_window", 500),
                low_percentile=self.config.get("noise", {}).get("er_low_percentile", 0.30),
                high_percentile=self.config.get("noise", {}).get("er_high_percentile", 0.80),
            )

    def update_noise_series(self, symbol: str, close_series) -> None:
        """用完整序列更新噪声估计"""
        if symbol in self.noise_estimators:
            self.noise_estimators[symbol].update(close_series)

    # ------------------------------------------------------------------
    # 信号查询
    # ------------------------------------------------------------------
    def get_confirmed_signals(self, trade_date: date) -> List[Signal]:
        """获取指定交易日的所有已确认可执行信号

        规则：confirmed_time <= trade_date 且 状态为PENDING

        Returns:
            signals: 可执行的信号列表（仅买入信号）
        """
        confirmed: List[Signal] = []
        for symbol, sigs in self.pending_signals.items():
            for sig in sigs:
                if (sig.status == SignalStatus.PENDING
                        and sig.confirmed_time is not None
                        and sig.confirmed_time <= trade_date
                        and sig.signal_type == "buy"):
                    sig.status = SignalStatus.CONFIRMED
                    confirmed.append(sig)
        return confirmed

    def get_pending_signals(self, symbol: str, status: SignalStatus = SignalStatus.PENDING) -> List[Signal]:
        """获取指定股票的所有待处理信号"""
        return [s for s in self.pending_signals.get(symbol, []) if s.status == status]

    def clear_executed_signals(self, symbol: str) -> None:
        """清除已执行的信号（保持pending队列精简）"""
        if symbol in self.pending_signals:
            self.pending_signals[symbol] = [
                s for s in self.pending_signals[symbol]
                if s.status not in (SignalStatus.EXECUTED, SignalStatus.INVALIDATED, SignalStatus.EXPIRED)
            ]

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------
    def warm_up(self, symbol: str, history_kbars: List[dict]) -> None:
        """回测前对单只股票进行预热

        回放历史K线建立缠论结构，但不生成交易信号。
        """
        state = self.chan_cache.setdefault(symbol, ChanStructureState(symbol))
        for kbar in history_kbars:
            state.process_kbar(kbar, emit_signals=False)
        logger.debug(f"[{symbol}] Warm-up complete: {len(state.confirmed_bis)} bis confirmed")

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def get_noise_ratio(self, symbol: str) -> Optional[float]:
        """获取当前噪声分位"""
        estimator = self.noise_estimators.get(symbol)
        return estimator.get_noise_ratio() if estimator else None

    def get_structure_state(self, symbol: str) -> Optional[ChanStructureState]:
        """获取股票当前的缠论状态"""
        return self.chan_cache.get(symbol)

    def get_active_stocks(self, fund_scores) -> List[str]:
        """获取活跃股票列表（30分钟级别筛选）"""
        return self.active_filter.filter(self.chan_cache, fund_scores)

    def _next_trade_date(self, kbar: dict) -> date:
        """获取下一个交易日（简化：K线时间+1天）"""
        from datetime import timedelta
        kbar_time = kbar.get("time", date.min)
        if isinstance(kbar_time, date):
            return kbar_time  # T+0: 当天确认，当天不执行
        return date.min

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------
    def get_snapshot(self) -> dict:
        """获取引擎快照（用于断点续跑）"""
        return {
            "chan_cache": self.chan_cache,
            "pending_signals": dict(self.pending_signals),
            "current_trade_date": self.current_trade_date,
        }

    def restore_snapshot(self, snapshot: dict) -> None:
        """恢复引擎状态"""
        self.chan_cache = snapshot.get("chan_cache", {})
        self.pending_signals = defaultdict(list, snapshot.get("pending_signals", {}))
        self.current_trade_date = snapshot.get("current_trade_date")

    def reset(self) -> None:
        """重置引擎"""
        self.chan_cache.clear()
        self.pending_signals.clear()
        self._todays_new_signals.clear()
        self.noise_estimators.clear()
