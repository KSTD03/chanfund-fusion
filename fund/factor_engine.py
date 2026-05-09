"""
基本面因子引擎 — FundFactorEngine
============================
分层更新策略：
- 实时因子 → 每日更新
- 财报因子 → 仅在披露日更新（PIT数据管道）
- 平滑过渡：新财报前10个交易日新旧值加权混合

数据流：
    PIT数据库 -> PITDataPipeline -> factor_engine -> 标准化 -> 融合
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

from .preprocess import standardize_pipeline
from .scene_detector import SceneDetector, MacroScene
from .redflag import RedFlagDetector, RedFlagResult

logger = logging.getLogger(__name__)


class FactorFunc:
    """因子计算函数包装器"""
    def __init__(self, name: str, calc_func: Callable, factor_type: str = "real_time",
                 weight: float = 1.0):
        self.name = name
        self.calc_func = calc_func
        self.factor_type = factor_type  # "real_time" or "financial"
        self.weight = weight

    def calc(self, symbol: str, trade_date: date, data_provider=None):
        return self.calc_func(symbol, trade_date, data_provider)


class FundFactorEngine:
    """基本面因子引擎

    职责：
    1. 管理实时因子和财报因子的计算与缓存
    2. PIT数据管道查询（防未来）
    3. 财报平滑过渡
    4. 跨截面标准化
    5. IC半衰期加权动态权重更新
    """

    def __init__(self, config: dict):
        self.config = config
        # 因子函数列表
        self.real_time_factors: List[FactorFunc] = []
        self.fin_factors: List[FactorFunc] = []
        # {symbol: {factor_name: value}}
        self.cache: Dict[str, dict] = defaultdict(dict)
        # {symbol: last_fin_period_end}
        self.last_fin_update: Dict[str, date] = {}
        # {symbol: fin_mix_start_date}
        self.fin_mix_starts: Dict[str, date] = {}
        # 因子权重衰减
        self.factor_weights: Dict[str, float] = {}
        self.ic_history: Dict[str, List[float]] = defaultdict(list)
        self.ic_half_life = 11  # 半衰期约11天: lambda = 0.94
        self.ic_lambda = 0.94
        # 单日权重变化限制
        self.max_weight_change = 0.20
        # EMA平滑窗口
        self.ema_window = 5
        # PIT数据管道
        self.pit_data = None
        # 市场数据提供者
        self.market_data = None

        # 场景检测器
        self.scene_detector = SceneDetector(config.get("scene_params", {}))
        self.current_scene: MacroScene = MacroScene.BALANCED

        # 排雷检测器
        self.redflag_detector = RedFlagDetector(config)

        # 平滑参数
        self.fin_mix_days = config.get("fin_mix_days", 10)
        self.fin_mix_initial_ratio = config.get("fin_mix_ratio", 0.75)

        # 估值因子动态调整
        self.market_pe_percentile = 0.5  # 全市场PE分位

    def register_factor(self, factor: FactorFunc):
        """注册因子"""
        if factor.factor_type == "real_time":
            self.real_time_factors.append(factor)
        else:
            self.fin_factors.append(factor)
        self.factor_weights[factor.name] = factor.weight

    def set_data_providers(self, pit_data=None, market_data=None):
        """设置数据提供者"""
        self.pit_data = pit_data
        self.market_data = market_data

    # ------------------------------------------------------------------
    # 每日更新
    # ------------------------------------------------------------------
    def update_daily(self, trade_date: date, universe: List[str]) -> None:
        """每日更新所有股票的基本面因子

        Args:
            trade_date: 当前交易日
            universe: 股票池
        """
        for symbol in universe:
            if symbol not in self.cache:
                self.cache[symbol] = {}

            # 1. 实时因子（每天）
            for factor in self.real_time_factors:
                try:
                    val = factor.calc(symbol, trade_date, self.market_data)
                    self.cache[symbol][factor.name] = val
                except Exception as e:
                    logger.warning(f"[{symbol}] Factor {factor.name} calc failed: {e}")

            # 2. 财报因子（仅在披露日更新）
            self._check_and_update_financials(symbol, trade_date)

        # 3. 跨截面标准化（每只股票都需参与）
        self._cross_sectional_standardize(universe, trade_date)

        # 4. 更新场景
        self._update_scene(trade_date)

    def _check_and_update_financials(self, symbol: str, trade_date: date) -> None:
        """查询PIT数据库，更新财报因子（含平滑过渡）

        基于 report_publish_date 判断是否有新财报可用。
        """
        if self.pit_data is None:
            return

        last_fin_date = self.last_fin_update.get(symbol, date(2000, 1, 1))

        # 从PIT数据库查询：在 trade_date 当日或之前披露，且报告期 > last_fin_date 的最新财报
        new_report = self.pit_data.get_latest_report(
            symbol, as_of=trade_date, after_period=last_fin_date
        )

        if new_report is not None:
            # 更新财报因子值
            for factor in self.fin_factors:
                try:
                    val = factor.calc(symbol, trade_date, new_report)
                    self.cache[symbol][factor.name] = val
                except Exception as e:
                    logger.warning(f"[{symbol}] Fin factor {factor.name} failed: {e}")

            # 触发平滑过渡
            self.fin_mix_starts[symbol] = trade_date
            self.last_fin_update[symbol] = new_report.get("period_end_date", trade_date)

            # 保存旧值用于平滑（简化：存快照）
            self._save_fin_snapshot(symbol)

    def _save_fin_snapshot(self, symbol: str):
        """保存财报更新前的旧因子值"""
        snapshot = {}
        for factor in self.fin_factors:
            old_val = self.cache[symbol].get(f"{factor.name}_old")
            current_val = self.cache[symbol].get(factor.name)
            if current_val is not None:
                self.cache[symbol][f"{factor.name}_old"] = current_val
                snapshot[factor.name] = current_val
        self.cache[symbol]["_fin_snapshot"] = snapshot

    def get_factor_value(self, symbol: str, factor_name: str, trade_date: date) -> Optional[float]:
        """获取因子值（含平滑过渡处理）"""
        val = self.cache.get(symbol, {}).get(factor_name)
        if val is None:
            return None

        # 检查平滑过渡
        mix_start = self.fin_mix_starts.get(symbol)
        if mix_start:
            days_since = (trade_date - mix_start).days
            if 0 < days_since < self.fin_mix_days:
                old_val = self.cache.get(symbol, {}).get(f"{factor_name}_old")
                if old_val is not None:
                    ratio = min(1.0, days_since / self.fin_mix_days)
                    val = old_val * (1 - ratio) + val * ratio

        return val

    # ------------------------------------------------------------------
    # 标准化
    # ------------------------------------------------------------------
    def _cross_sectional_standardize(self, universe: List[str], trade_date: date) -> None:
        """跨截面MAD去极值 + 行业ZScore + 市值中性化"""
        # 构造截面DataFrame
        data = []
        for symbol in universe:
            row = {"symbol": symbol}
            for f_name in list(self.factor_weights.keys()):
                row[f_name] = self.cache.get(symbol, {}).get(f_name)
            row["industry"] = self._get_industry(symbol)
            row["log_mktcap"] = self._get_log_mktcap(symbol, trade_date)
            row["date"] = trade_date
            data.append(row)

        if not data:
            return

        df = pd.DataFrame(data)

        for f_name in self.factor_weights:
            if f_name not in df.columns or df[f_name].isna().all():
                continue

            try:
                df = standardize_pipeline(
                    df, f_name,
                    group_field="industry",
                    mktcap_col="log_mktcap",
                    n_mad=5.0,
                    do_neutralize=True,
                )
                neutral_col = f"{f_name}_clean_z_neutral"
                if neutral_col in df.columns:
                    for _, row in df.iterrows():
                        symbol = row["symbol"]
                        self.cache[symbol][f"{f_name}_std"] = row[neutral_col]

            except Exception as e:
                logger.warning(f"Standardize failed for {f_name}: {e}")

    # ------------------------------------------------------------------
    # 综合得分
    # ------------------------------------------------------------------
    def get_latest_scores(self, trade_date: date, universe: List[str]) -> pd.Series:
        """计算综合基本面得分

        Returns:
            scores: Series(index=symbol, value=score 0-100)
        """
        scores = {}
        for symbol in universe:
            cache = self.cache.get(symbol, {})
            if not cache:
                scores[symbol] = 50.0  # 无财务数据时给中间分，避免硬过滤全部失效
                continue

            raw_score = self._weighted_sum(symbol)

            # 排雷检查
            redflag_result = self._check_redflag(symbol)
            if redflag_result.has_flag:
                raw_score = min(raw_score, redflag_result.score_cap)

            # 估值因子动态调整
            raw_score = self._adjust_for_market_pe(raw_score)

            # 压缩到0-100
            final_score = max(0.0, min(100.0, raw_score))
            scores[symbol] = final_score

        return pd.Series(scores)

    def _weighted_sum(self, symbol: str) -> float:
        """加权求和"""
        cache = self.cache.get(symbol, {})
        total = 0.0
        weight_sum = 0.0

        for f_name, w in self.factor_weights.items():
            std_val = cache.get(f"{f_name}_std")
            if std_val is not None and not np.isnan(std_val):
                # 动态权重（如果有IC数据）
                dynamic_w = self.factor_weights.get(f_name, w)
                total += std_val * dynamic_w
                weight_sum += abs(dynamic_w)

        return total / max(weight_sum, 0.001) * 50 + 50  # 映射到0-100

    # ------------------------------------------------------------------
    # 排雷
    # ------------------------------------------------------------------
    def _check_redflag(self, symbol: str) -> RedFlagResult:
        """检查股票是否有财务异常"""
        fin_data = {}
        cache = self.cache.get(symbol, {})
        # 提取财务数据传给排雷模块
        field_map = {
            "ar_ttm": "ar_ttm",
            "revenue_ttm": "revenue_ttm",
            "ocf_ttm": "ocf_ttm",
            "inventory": "inventory",
            "cost_of_sales": "cost_of_sales",
            "cash_assets": "cash_assets",
            "total_assets": "total_assets",
            "interest_bearing_debt": "interest_bearing_debt",
        }
        for target, src in field_map.items():
            val = cache.get(src)
            if val is not None:
                fin_data[target] = val

        return self.redflag_detector.check(symbol, fin_data)

    def _adjust_for_market_pe(self, score: float) -> float:
        """全市场估值极端时调整"""
        if self.market_pe_percentile > 0.90 or self.market_pe_percentile < 0.10:
            # 极端时降估值权重、升盈利权重
            score *= 0.9  # 整体保守
        return score

    # ------------------------------------------------------------------
    # 场景更新
    # ------------------------------------------------------------------
    def _update_scene(self, trade_date: date):
        """更新宏观场景"""
        # 需要外部提供数据，这里仅保留接口
        # 实际场景检测由主策略在外部调用
        pass

    def set_scene(self, scene: MacroScene):
        self.current_scene = scene

    # ------------------------------------------------------------------
    # IC半衰期加权
    # ------------------------------------------------------------------
    def update_ic(self, factor_name: str, ic_t: float):
        """更新IC序列"""

        self.ic_history[factor_name].append(ic_t)

        # 指数衰减权重
        n = len(self.ic_history[factor_name])
        weights = np.array([self.ic_lambda ** (n - 1 - i) for i in range(n)])
        ic_arr = np.array(self.ic_history[factor_name])

        ic_mean = np.sum(weights * ic_arr) / max(np.sum(weights), 1e-10)
        ic_std = np.sqrt(np.sum(weights * (ic_arr - ic_mean) ** 2) / max(np.sum(weights), 1e-10))

        if ic_std > 0:
            # IR = IC_mean / IC_std => 因子权重
            new_weight = max(0.0, ic_mean / ic_std)
        else:
            new_weight = 0.5

        # 限制单日变动
        old_weight = self.factor_weights.get(factor_name, 1.0)
        change = new_weight - old_weight
        clamped_change = max(-self.max_weight_change, min(self.max_weight_change, change))
        self.factor_weights[factor_name] = old_weight + clamped_change

        # 归一化
        total = sum(abs(v) for v in self.factor_weights.values())
        if total > 0:
            for k in self.factor_weights:
                self.factor_weights[k] /= total

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _get_industry(self, symbol: str) -> str:
        """获取行业分类（由外部注入）"""
        return getattr(self, '_industry_map', {}).get(symbol, "unknown")

    def _get_log_mktcap(self, symbol: str, trade_date: date) -> float:
        """获取对数市值（由外部提供）"""
        return 0.0  # placeholder

    def set_industry_map(self, industry_map: dict):
        self._industry_map = industry_map

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------
    def get_snapshot(self) -> dict:
        return {
            "cache": dict(self.cache),
            "last_fin_update": self.last_fin_update,
            "fin_mix_starts": self.fin_mix_starts,
            "factor_weights": self.factor_weights,
            "current_scene": self.current_scene.value if self.current_scene else None,
        }

    def restore_snapshot(self, snapshot: dict):
        self.cache = defaultdict(dict, snapshot.get("cache", {}))
        self.last_fin_update = snapshot.get("last_fin_update", {})
        self.fin_mix_starts = snapshot.get("fin_mix_starts", {})
        self.factor_weights = snapshot.get("factor_weights", {})
        scene_val = snapshot.get("current_scene", "balanced")
        for s in MacroScene:
            if s.value == scene_val:
                self.current_scene = s
                break

    def reset(self):
        self.cache.clear()
        self.last_fin_update.clear()
        self.fin_mix_starts.clear()
        self.factor_weights.clear()
        self.ic_history.clear()
        self.scene_detector.reset()
