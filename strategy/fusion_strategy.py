"""
主策略 — FusionStrategy
==================
继承Qlib的BaseStrategy，整合技术子系统、基本面子系统、融合引擎和风控模块。

回测流程（每日）：
    1. 获取技术信号（已确认，T+1）
    2. 获取最新基本面得分
    3. 信号融合过滤
    4. 仓位计算（风控约束）
    5. 执行交易决策
    6. 更新持仓状态
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd

from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir

from .config_schema import load_config, apply_overrides
from .tech.signal_engine import TechSignalEngine
from .tech.trial_buy import TrialBuyManager
from .fund.factor_engine import FundFactorEngine
from .fusion.rules import FusionRules
from .risk.budget_manager import RiskBudgetManager
from .risk.position_manager import PositionManager
from .utils.logger import DecisionLogger, RejectReason
from .utils.snapshot import SnapshotManager
from .data.market_data import MarketDataProvider
from .data.pit_data import PITDataPipeline

logger = logging.getLogger(__name__)


class FusionStrategy(BaseStrategy):
    """双系统融合策略（技术+基本面）

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

        # 初始化各子系统
        self.tech_engine = TechSignalEngine(self.config.get("tech", {}))
        self.fund_engine = FundFactorEngine(self.config.get("fund", {}))
        self.budget_manager = RiskBudgetManager(self.config)
        self.pos_manager = PositionManager(self.config, self.budget_manager)
        self.decision_logger = DecisionLogger()
        self.trial_manager = TrialBuyManager(
            self.config.get("tech", {}).get("trial", {})
        )

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

        # 数据接口
        self.market_data = None
        self.data_provider = None

        logger.info("FusionStrategy initialized")

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
            from datetime import datetime
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

        # 3. 获取噪声评估
        self._update_noise_ratios(trade_date)

        # 4. 重置每日预算
        equity = self._get_equity()
        self.budget_manager.new_day(trade_date, equity)

        # 5. 信号融合
        decisions = self.fusion_rules.filter_and_merge(
            tech_signals,
            self._fund_scores,
            trade_date,
            self._noise_ratios,
        )

        # 6. 计算目标仓位
        target_weights = self.pos_manager.calc_positions(decisions, trade_date)

        # 7. 更新持仓（止损、移损等）
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
        logger.info("Backtest started")

    def on_end(self):
        """回测结束时的清理"""
        self.save_snapshot()
        summary = self.decision_logger.summary()
        logger.info(f"Backtest ended.\n{summary}")

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
        """预热：回放历史K线建立缠论结构，不生成信号"""
        logger.info(f"Starting warm-up (until {current_date})...")

        start_date = self._get_warmup_start()

        if self.market_data and self._universe:
            for symbol in self._universe[:100]:  # 限制预热数量
                try:
                    kbars = self._get_kbars(symbol, start_date, current_date)
                    if kbars.empty:
                        continue

                    # 转为dict列表
                    kbar_list = self._df_to_kbar_list(kbars)

                    # 预热技术引擎
                    self.tech_engine.warm_up(symbol, kbar_list)

                    # 预热噪声估计
                    if "close" in kbars.columns:
                        close_series = kbars["close"].values
                        self.tech_engine.update_noise_series(symbol, close_series)

                except Exception as e:
                    logger.warning(f"[{symbol}] Warm-up failed: {e}")

        self._is_warmed_up = True
        logger.info("Warm-up complete")

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
    # 持仓管理
    # ------------------------------------------------------------------
    def _update_holdings(self, trade_date: date):
        """每日更新持仓状态

        - 检查保护性止损
        - 基本面恶化检查
        - 试探仓状态更新
        """
        # 1. 止损检查
        for pos in self.pos_manager.get_holdings():
            symbol = pos["symbol"]
            # 获取当前最新价格
            current_price = self._get_current_price(symbol, trade_date)
            if current_price is None:
                continue

            stop_price = pos.get("stop_price")
            if stop_price is not None:
                if (current_price <= stop_price and pos["entry_price"] > stop_price) or \
                   (current_price >= stop_price and pos["entry_price"] < stop_price):
                    self.pos_manager.close_position(
                        symbol, current_price, trade_date, "stop_loss"
                    )

            # 2. 基本面恶化检查
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

        # 3. 试探仓检查
        for symbol in self.trial_manager.get_active_symbols():
            kbar, _ = self._get_latest_kbar(symbol, trade_date)
            if kbar:
                action = self.trial_manager.update(
                    symbol, 0, kbar["high"], kbar["low"],
                    kbar["close"], trade_date
                )
                if action in ("stop", "expire", "takeprofit"):
                    current_price = kbar.get("close", kbar["high"])
                    self.pos_manager.close_position(
                        symbol, current_price, trade_date, f"trial_{action}"
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
        from datetime import datetime
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
