"""
ChanFund Fusion 主策略 — Strategy
=============================
继承Qlib的BaseStrategy，整合技术子系统、基本面子系统、融合引擎和风控模块。

v1.0 优化（2026-05-03）：
  Phase 1: 保护性移动止损（替代固定5%止损）
  Phase 2: ER噪声过滤 + 冷却期机制
  Phase 3: 分级仓位 + 动态风险预算
  Phase 4: 趋势方向过滤器（EMA12/26，250MA）
  Phase 5: 红黄牌机制（接入 RedFlagDetector）

回测流程（每日）：
    1. 获取技术信号（已确认，T+1）
    2. 获取最新基本面得分
    3. ER噪声过滤 → 方向过滤 → 红黄牌过滤
    4. 信号融合（相位优先级）
    5. 动态风险预算仓位计算
    6. 保护性止损更新
    7. 执行交易决策
    8. 更新持仓状态
"""

from __future__ import annotations

from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir

from .config_schema import load_config, apply_overrides
from .tech.signal_engine import TechSignalEngine
from .tech.trial_buy import TrialBuyManager
from .fund.factor_engine import FundFactorEngine
from .fund.redflag import RedFlagDetector, RedFlagResult
from .fusion.rules import FusionRules
from .risk.budget_manager import RiskBudgetManager
from .risk.position_manager import PositionManager
from .utils.logger import DecisionLogger, RejectReason
from .utils.snapshot import SnapshotManager
from .data.market_data import MarketDataProvider
from .data.pit_data import PITDataPipeline

logger = logging.getLogger(__name__)


class Strategy(BaseStrategy):
    """ChanFund Fusion 双系统融合策略（缠论+基本面）

    核心流程：
        generate_trade_decision 每日被Qlib框架调用一次，
        返回目标权重字典。
    """

    def __init__(self, config_path: str = None, overrides: dict = None):
        super().__init__()

        # 加载配置
        self.config = load_config(config_path)
        if overrides:
            self.config = apply_overrides(self.config, overrides)

        self.strategy_name = "ChanFund Fusion"
        self.strategy_version = "1.0.0"

        # 初始化各子系统
        self.tech_engine = TechSignalEngine(self.config.get("tech", {}))
        self.fund_engine = FundFactorEngine(self.config.get("fund", {}))
        self.budget_manager = RiskBudgetManager(self.config)
        self.pos_manager = PositionManager(self.config, self.budget_manager)
        self.decision_logger = DecisionLogger()
        self.trial_manager = TrialBuyManager(
            self.config.get("tech", {}).get("trial", {})
        )

        # 【Phase 5】排雷检测器
        self.redflag_detector = RedFlagDetector(self.config.get("fund", {}))

        # 融合引擎 (依赖其他模块，最后初始化)
        self.fusion_rules = FusionRules(
            self.config,
            self.tech_engine,
            self.fund_engine,
            self.budget_manager,
            self.pos_manager,
            self.decision_logger,
        )

        # 数据层
        self.market_data = MarketDataProvider(self.config.get("data", {}))
        self.pit_data = PITDataPipeline()

        # 快照管理
        self.snapshot_manager = SnapshotManager(
            base_dir="snapshots",
            experiment_id="default",
        )

        # 回测状态
        self.current_date: Optional[date] = None
        self._universe: List[str] = []
        self._is_warmed_up: bool = False
        self._fund_scores: Dict[str, float] = {}
        self._noise_ratios: Dict[str, float] = {}
        self._ma_states: Dict[str, str] = {}     # 【Phase 4】均线状态
        self._redflag_results: Dict[str, RedFlagResult] = {}  # 【Phase 5】排雷结果
        self._chan_states: Dict[str, dict] = {}  # 缠论结构状态缓存（用于止损）

        # 数据接口
        self.data_provider = None

        logger.info(f"ChanFund Fusion v{self.strategy_version} initialized")

    # ------------------------------------------------------------------
    # Qlib策略接口
    # ------------------------------------------------------------------
    def generate_trade_decision(self, trade_date, universe, **kwargs) -> Dict:
        """生成当日的交易决策（每日被Qlib调用一次）

        Args:
            trade_date: 交易日
            universe: 当日可交易股票列表

        Returns:
            target_weights: {symbol: weight}
        """
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        elif isinstance(trade_date, pd.Timestamp):
            trade_date = trade_date.date()

        self.current_date = trade_date
        self._universe = list(universe) if universe is not None else []

        # 如果没有预热，先做一次完整预热
        if not self._is_warmed_up:
            self._warm_up(trade_date)

        # 1. 获取已确认的技术信号
        tech_signals = self.tech_engine.get_confirmed_signals(trade_date)

        # 2. 更新基本面子系统
        self.fund_engine.update_daily(trade_date, self._universe)
        self._fund_scores = self.fund_engine.get_latest_scores(
            trade_date, self._universe
        ).to_dict()

        # 3. 获取噪声评估（Phase 2）
        self._update_noise_ratios(trade_date)

        # 4. 【Phase 4】更新均线状态
        self._update_ma_states(trade_date)

        # 5. 【Phase 5】更新排雷结果
        self._update_redflag_results(trade_date)

        # 6. 【Phase 1】更新缠论结构状态（用于止损计算）
        self._update_chan_states(trade_date)

        # 7. 重置每日预算
        equity = self._get_equity()
        self.budget_manager.new_day(trade_date, equity)

        # 8. 【Phase 1】更新持仓保护性止损
        self._update_protective_stops(trade_date)

        # 9. 信号融合（含 Phase 2-5 过滤）
        decisions = self.fusion_rules.filter_and_merge(
            tech_signals,
            self._fund_scores,
            trade_date,
            noise_ratios=self._noise_ratios,
            ma_states=self._ma_states,
            redflag_results=self._redflag_results,
        )

        # 10. 动态仓位计算（含 Phase 3）
        entry_prices = self._get_entry_prices(decisions, trade_date)
        stop_prices = self._get_stop_prices(decisions)
        target_weights = self.pos_manager.calc_positions(
            decisions, trade_date, entry_prices, stop_prices
        )

        # 11. 更新持仓（止损检查等）
        self._update_holdings(trade_date)

        logger.debug(
            f"[{trade_date}] Tech signals: {len(tech_signals)}, "
            f"Decisions: {len(decisions)}, "
            f"Targets: {len(target_weights)}"
        )

        return target_weights

    # ------------------------------------------------------------------
    # 其他Qlib回调
    # ------------------------------------------------------------------
    def on_start(self):
        """回测开始时的初始化"""
        logger.info(f"ChanFund Fusion v{self.strategy_version} backtest started")

    def on_end(self):
        """回测结束时的清理"""
        self.save_snapshot()
        summary = self.decision_logger.summary()
        logger.info(f"Backtest ended.\n{summary}")

    # ------------------------------------------------------------------
    # Phase 1: 保护性移动止损更新
    # ------------------------------------------------------------------
    def _update_protective_stops(self, trade_date: date):
        """每日更新所有持仓的保护性移动止损

        基于缠论结构（最近笔低点、中枢ZG）持续上移止损位
        """
        for pos in self.pos_manager.get_holdings():
            symbol = pos["symbol"]
            current_price = self._get_current_price(symbol, trade_date)
            if current_price is None:
                continue

            # 获取该股票的缠论状态
            chan_state = self._chan_states.get(symbol, {})

            # 计算新的止损价
            new_stop = self.pos_manager.calc_stop_price(
                symbol=symbol,
                entry_price=pos["entry_price"],
                current_price=current_price,
                entry_date=pos["entry_date"],
                current_date=trade_date,
                chan_state=chan_state,
            )

            # 更新止损位
            self.pos_manager.set_stop_price(symbol, new_stop)

    # ------------------------------------------------------------------
    # Phase 4: 均线方向状态更新
    # ------------------------------------------------------------------
    def _update_ma_states(self, trade_date: date):
        """更新全市场股票的均线状态（v1.0.1: 支持自适应模式）

        判断标准（受 trend_filter.mode 影响）：
        - strict模式: EMA12 > EMA26 (+250MA)
        - adaptive模式: 按场景选用 EMA50/100 或 价格>MA200
        - off模式: 只计算bull/bear基础状态

        返回:
            "bull": 多头排列
            "bear": 空头排列
            "neutral": 中性
        """
        self._ma_states = {}

        # 根据模式选择均线参数
        tf_mode = self.fusion_rules.trend_filter_mode
        if tf_mode == "adaptive":
            ema_fast = self.fusion_rules.tf_adaptive_fast    # 50
            ema_slow = self.fusion_rules.tf_adaptive_slow    # 100
            price_ma = self.fusion_rules.tf_price_ma_period  # 200
        elif tf_mode == "strict":
            ema_fast = self.fusion_rules.tf_strict_fast      # 12
            ema_slow = self.fusion_rules.tf_strict_slow      # 26
            price_ma = self.fusion_rules.ma_long             # 120
        else:  # off
            ema_fast = self.fusion_rules.ema_fast
            ema_slow = self.fusion_rules.ema_slow
            price_ma = 0

        for symbol in self._universe:
            try:
                start = (trade_date - timedelta(days=max(500, price_ma + 50))).isoformat()
                end = trade_date.isoformat()
                df = self._get_kbars(symbol, start, end)
                if df.empty or len(df) < ema_slow + 5:
                    continue

                closes = df["close"].values
                ema_fast_val = self._ema(closes, ema_fast)[-1]
                ema_slow_val = self._ema(closes, ema_slow)[-1]
                current_price = closes[-1]

                if ema_fast_val > ema_slow_val:
                    # 多头排列 → 检查年线
                    if price_ma > 0 and len(closes) >= price_ma:
                        ma_long_val = np.mean(closes[-price_ma:])
                        if current_price >= ma_long_val:
                            self._ma_states[symbol] = "bull"
                        else:
                            self._ma_states[symbol] = "neutral"
                    else:
                        self._ma_states[symbol] = "bull"
                else:
                    self._ma_states[symbol] = "bear"

            except Exception as e:
                logger.debug(f"[{symbol}] MA state calc failed: {e}")

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        """计算EMA"""
        if len(values) < period:
            return values
        alpha = 2.0 / (period + 1.0)
        result = np.zeros_like(values)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result

    # ------------------------------------------------------------------
    # Phase 5: 排雷结果更新
    # ------------------------------------------------------------------
    def _update_redflag_results(self, trade_date: date):
        """更新全市场排雷结果"""
        self._redflag_results = {}
        for symbol in self._universe:
            # 从fund_engine缓存中提取财务数据
            cache = self.fund_engine.cache.get(symbol, {})
            fin_data = {
                "ar_ttm": cache.get("ar_ttm"),
                "revenue_ttm": cache.get("revenue_ttm"),
                "ocf_ttm": cache.get("ocf_ttm"),
                "inventory": cache.get("inventory"),
                "cost_of_sales": cache.get("cost_of_sales"),
                "cash_assets": cache.get("cash_assets"),
                "total_assets": cache.get("total_assets"),
                "interest_bearing_debt": cache.get("interest_bearing_debt"),
                # 红牌字段（需外部数据源提供）
                "is_st": cache.get("is_st", False),
                "is_investigation": cache.get("is_investigation", False),
                "audit_opinion": cache.get("audit_opinion", ""),
                "pledge_ratio": cache.get("pledge_ratio", 0.0),
                "goodwill_ratio": cache.get("goodwill_ratio", 0.0),
            }
            result = self.redflag_detector.check(symbol, fin_data, trade_date=trade_date)
            if result.has_flag:
                self._redflag_results[symbol] = result

    # ------------------------------------------------------------------
    # 缠论结构状态更新（用于止损计算）
    # ------------------------------------------------------------------
    def _update_chan_states(self, trade_date: date):
        """提取缠论结构状态用于止损计算"""
        self._chan_states = {}
        for symbol in self._universe:
            if not self.pos_manager.has_position(symbol):
                continue  # 仅对持仓股票更新时间
            state = self.tech_engine.get_structure_state(symbol)
            if state is None:
                continue

            # 提取关键价格位
            chan_state = {}
            latest_bi = state.get_latest_bi()
            if latest_bi:
                chan_state["latest_bi_low"] = latest_bi.low
                chan_state["latest_bi_high"] = latest_bi.high

            latest_zs = state.get_latest_zs()
            if latest_zs:
                chan_state["latest_zs_zg"] = latest_zs.zg
                chan_state["latest_zs_zd"] = latest_zs.zd

            # ATR（简化计算）
            try:
                start = (trade_date - timedelta(days=50)).isoformat()
                end = trade_date.isoformat()
                df = self._get_kbars(symbol, start, end)
                if not df.empty and len(df) >= 20:
                    high, low, close = df["high"].values, df["low"].values, df["close"].values
                    tr = np.maximum(
                        high[1:] - low[1:],
                        np.maximum(
                            np.abs(high[1:] - close[:-1]),
                            np.abs(low[1:] - close[:-1])
                        )
                    )
                    chan_state["atr"] = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            except Exception:
                pass

            self._chan_states[symbol] = chan_state

    # ------------------------------------------------------------------
    # 数据准备与预热
    # ------------------------------------------------------------------
    def set_data_provider(self, provider):
        """设置Qlib数据提供者（由外部传入）"""
        self.data_provider = provider
        if self.market_data:
            self.market_data.init_from_qlib(provider)

    def set_pit_data(self, pit_df: pd.DataFrame):
        """设置PIT财报数据"""
        self.pit_data.load_financial_reports(pit_df)
        self.fund_engine.pit_data = self.pit_data

    def _warm_up(self, current_date: date):
        """预热：回放历史K线建立缠论结构，不生成信号（优化1：流式处理+内存释放）"""
        logger.info(f"Starting ChanFund Fusion warm-up (until {current_date})...")

        start_date = self._get_warmup_start()

        # 内存标记
        import psutil
        mem_before = psutil.Process().memory_info().rss / 1024**3
        logger.warning(f"[MEM] Warm-up start: {mem_before:.2f} GB")

        if self.market_data and self._universe:
            warmup_limit = min(len(self._universe), 500)
            for symbol in self._universe[:warmup_limit]:
                if symbol in self.tech_engine.chan_cache:
                    continue
                try:
                    kbars = self._get_kbars(symbol, start_date, current_date)
                    if kbars.empty:
                        continue

                    kbar_list = self._df_to_kbar_list(kbars)
                    self.tech_engine.warm_up(symbol, kbar_list)

                    if "close" in kbars.columns:
                        close_series = kbars["close"].values
                        self.tech_engine.update_noise_series(symbol, close_series)

                    # 优化1：循环末尾释放中间数据
                    del kbars
                    del kbar_list
                    if 'close_series' in dir():
                        del close_series

                except Exception as e:
                    logger.warning(f"[{symbol}] Warm-up failed: {e}")

        self._is_warmed_up = True
        import gc
        gc.collect()
        mem_after = psutil.Process().memory_info().rss / 1024**3
        logger.warning(f"[MEM] Warm-up complete: {mem_after:.2f} GB (freed {mem_before-mem_after:.2f} GB)")

    # ------------------------------------------------------------------
    # 数据获取
    # ------------------------------------------------------------------
    def _get_kbars(self, symbol: str, start, end, freq: str = "day"):
        """获取K线数据"""
        start_str = start.isoformat() if hasattr(start, "isoformat") else str(start)
        end_str = end.isoformat() if hasattr(end, "isoformat") else str(end)

        if self.market_data:
            return self.market_data.get_kbars(symbol, start_str, end_str, freq)

        return pd.DataFrame()

    def _df_to_kbar_list(self, df: pd.DataFrame) -> List[dict]:
        """将DataFrame转为K线字典列表"""
        kbars = []
        for idx, row in df.iterrows():
            t = idx.to_pydatetime().date() if hasattr(idx, "to_pydatetime") else idx
            kbar = {
                "time": t,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            }
            kbars.append(kbar)
        return kbars

    # ------------------------------------------------------------------
    # 噪声更新
    # ------------------------------------------------------------------
    def _update_noise_ratios(self, trade_date: date):
        """更新全市场噪声评估"""
        self._noise_ratios = {}
        for symbol in self._universe:
            ratio = self.tech_engine.get_noise_ratio(symbol)
            if ratio is not None:
                self._noise_ratios[symbol] = ratio

    # ------------------------------------------------------------------
    # 入场价和止损价辅助
    # ------------------------------------------------------------------
    def _get_entry_prices(self, decisions: List[Tuple], trade_date: date) -> Dict[str, float]:
        """获取决策股票的入场价格"""
        prices = {}
        for symbol, _, _ in decisions:
            price = self._get_current_price(symbol, trade_date)
            if price is not None:
                prices[symbol] = price
        return prices

    def _get_stop_prices(self, decisions: List[Tuple]) -> Dict[str, float]:
        """获取决策股票的初始止损价格"""
        prices = {}
        for symbol, _, _ in decisions:
            chan_state = self._chan_states.get(symbol, {})
            entry_price = self._get_current_price(symbol, self.current_date)
            if entry_price is not None:
                stop = self.pos_manager.calc_stop_price(
                    symbol=symbol,
                    entry_price=entry_price,
                    current_price=entry_price,
                    entry_date=self.current_date,
                    current_date=self.current_date,
                    chan_state=chan_state,
                )
                prices[symbol] = stop
        return prices

    # ------------------------------------------------------------------
    # 持仓管理
    # ------------------------------------------------------------------
    def _update_holdings(self, trade_date: date):
        """每日更新持仓状态

        - 检查保护性止损（Phase 1）
        - 【v1.1】多级止盈检查
        - 基本面恶化检查
        - 试探仓状态更新
        """
        for pos in list(self.pos_manager.get_holdings()):
            symbol = pos["symbol"]
            current_price = self._get_current_price(symbol, trade_date)
            if current_price is None:
                continue

            stop_price = pos.get("stop_price")

            # 1. 【v1.1】先检查止盈（优先级高于止损）
            tp_action = self.pos_manager.check_take_profit(
                symbol, current_price, trade_date
            )
            if tp_action:
                action = tp_action["action"]
                if action == "close":
                    self.pos_manager.close_position(
                        symbol, current_price, trade_date,
                        tp_action["reason"], skip_cooling=True
                    )
                    continue
                elif action == "reduce":
                    self.pos_manager.reduce_position(
                        symbol, current_price, trade_date,
                        tp_action["reason"], tp_action["reduce_pct"]
                    )
                    # 减仓后继续检查止损

            # 2. 止损检查（保护性移动止损）
            if stop_price is not None:
                # 做多：价格 ≤ 止损位 → 止损
                if current_price <= stop_price and pos["entry_price"] > stop_price:
                    self.pos_manager.close_position(
                        symbol, current_price, trade_date, "stop_loss"
                    )
                    continue

            # 3. 基本面恶化检查
            fund_score = self._fund_scores.get(symbol, 50)
            if fund_score < 40:
                # 低于40分，次日减半仓
                new_weight = pos["weight"] * 0.5
                if new_weight > 0:
                    self.pos_manager.holdings[symbol]["weight"] = new_weight
                    logger.info(
                        f"[{symbol}] Fund score {fund_score:.1f} < 40, "
                        f"half position"
                    )
                else:
                    self.pos_manager.close_position(
                        symbol, current_price, trade_date, "fund_deterioration"
                    )

        # 4. 试探仓检查
        for symbol in self.trial_manager.get_active_symbols():
            kbar, _ = self._get_latest_kbar(symbol, trade_date)
            if kbar:
                action = self.trial_manager.update(
                    symbol, 0, kbar["high"], kbar["low"],
                    kbar["close"], trade_date
                )
                if action in ("stop", "expire", "takeprofit_half"):
                    current_price = kbar.get("close", kbar["high"])
                    self.pos_manager.close_position(
                        symbol, current_price, trade_date, f"trial_{action}"
                    )
                elif action == "takeprofit":
                    current_price = kbar.get("close", kbar["high"])
                    pos = self.pos_manager.holdings.get(symbol)
                    if pos:
                        half_weight = pos["weight"] * 0.5
                        pos["weight"] = half_weight
                        logger.info(
                            f"[{symbol}] TrialBuy halved at 2R, remaining trailing cost"
                        )

    def _get_current_price(self, symbol: str, trade_date: date) -> Optional[float]:
        """获取股票当前价格"""
        kbar, _ = self._get_latest_kbar(symbol, trade_date)
        if kbar:
            return kbar.get("close", kbar.get("high"))
        return None

    def _get_latest_kbar(self, symbol: str, trade_date: date) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
        """获取最新K线"""
        if not self.market_data:
            return None, None

        start = (trade_date - timedelta(days=10)).isoformat()
        end = trade_date.isoformat()

        df = self.market_data.get_kbars(symbol, start, end)
        if df.empty:
            return None, None

        last_row = df.iloc[-1]
        kbar = {
            "time": last_row.name.date() if hasattr(last_row.name, "date") else last_row.name,
            "open": float(last_row.get("open", 0)),
            "high": float(last_row.get("high", 0)),
            "low": float(last_row.get("low", 0)),
            "close": float(last_row.get("close", 0)),
            "volume": float(last_row.get("volume", 0)),
        }
        return kbar, df

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _get_equity(self) -> float:
        """获取当前总权益（由外部注入或使用初始值）"""
        return getattr(self, '_current_equity', self.config.get("backtest", {}).get("initial_cash", 1_000_000))

    def set_equity(self, equity: float):
        self._current_equity = equity

    def _get_warmup_start(self) -> date:
        """获取预热起始日期"""
        data_config = self.config.get("data", {})
        start_str = data_config.get("start_date", "2017-01-01")
        return datetime.strptime(start_str, "%Y-%m-%d").date()

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------
    def save_snapshot(self):
        """保存当前回测快照"""
        if not self.current_date:
            return
        self.snapshot_manager.save_snapshot(
            self.tech_engine.get_snapshot(),
            self.fund_engine.get_snapshot(),
            self.decision_logger,
            self.current_date,
        )

    def load_snapshot(self, target_date: date) -> bool:
        """加载快照恢复状态"""
        snapshot = self.snapshot_manager.load_snapshot(target_date)
        if snapshot is None:
            return False
        self.tech_engine.restore_snapshot(snapshot.get("tech_engine", {}))
        self.fund_engine.restore_snapshot(snapshot.get("fund_engine", {}))
        self._is_warmed_up = True
        return True

    # ------------------------------------------------------------------
    # 外部设置
    # ------------------------------------------------------------------
    def set_industry_map(self, industry_map: dict):
        """设置行业映射"""
        self.fund_engine.set_industry_map(industry_map)
        self.pos_manager.set_sector_map(industry_map)

    def set_experiment_id(self, experiment_id: str):
        """设置实验ID（按参数隔离快照）"""
        self.snapshot_manager = SnapshotManager(
            base_dir="snapshots",
            experiment_id=experiment_id,
        )

    def get_decision_log(self) -> list:
        return self.decision_logger.to_dict()

    def get_trade_log(self) -> list:
        return self.pos_manager.get_trade_log()
