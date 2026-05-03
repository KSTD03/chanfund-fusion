"""
效率比率(ER)与噪声评估
=================
v1.0 优化：
- NoiseEstimator.get_noise_ratio() 返回 ER 在历史中的分位 [0,1]
  值越高表示信号越有效（信噪比越高）
- ER < 30% 分位 → 高噪声，丢弃信号（由 FusionRules 判断）
- ER > 80% 分位 → 低噪声，高信噪比

- EfficiencyRatio: ER计算
- NoiseEstimator: 自适应噪声阈值
- ActiveStockFilter: 30分钟级别活跃股票筛选
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .chan_structure import ChanStructureState


class EfficiencyRatio:
    """效率比率 ER = |ΔP| / Σ|dP|

    衡量价格运动的"效率"——方向统一则ER高，杂乱震荡则ER低。
    ER高 = 趋势明显（低噪声）
    ER低 = 震荡剧烈（高噪声）
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calc(self, price_series: np.ndarray) -> float:
        """计算ER

        Args:
            price_series: 价格序列（一维numpy数组）

        Returns:
            er: 效率比率 [0, 1]

        Raises:
            ValueError: 数据不足
        """
        if len(price_series) < self.period + 1:
            raise ValueError(
                f"Need at least {self.period + 1} data points, got {len(price_series)}"
            )

        # 剔除停牌期（价格不变的天数）
        valid_prices = price_series[
            (np.diff(price_series, prepend=price_series[0]) != 0) |
            (range(len(price_series)) == 0)
        ]

        if len(valid_prices) < self.period + 1:
            return 0.0  # 数据不足，视为噪声

        # 取最近period期
        recent = valid_prices[-self.period - 1:]

        numerator = abs(recent[-1] - recent[0])
        denominator = np.sum(np.abs(np.diff(recent)))

        if denominator == 0:
            return 1.0  # 完全无波动 = 高度有效率

        return numerator / denominator


class NoiseEstimator:
    """噪声评估器

    基于ER的历史滚动百分位判断当前噪声水平：
    - get_noise_ratio() 返回 ER 在历史中的分位 [0,1]
    - 低分位（< 30%）→ 高噪声，信号不可靠
    - 高分位（> 80%）→ 低噪声，信号可靠
    - 自适应于市场波动率变化

    注意：
    get_noise_ratio() 返回的是"信噪比分位"而非原始的ER值。
    值越高 = 信号越可靠（噪声越低）。
    """

    def __init__(
        self,
        er_period: int = 20,
        hist_window: int = 500,
        low_percentile: float = 0.30,
        high_percentile: float = 0.80,
    ):
        self.er_calc = EfficiencyRatio(period=er_period)
        self.history = deque(maxlen=hist_window)
        self.low_percentile = low_percentile      # 低分位阈值
        self.high_percentile = high_percentile    # 高分位阈值
        self._current_er: Optional[float] = None

    def update(self, price_series: np.ndarray) -> float:
        """计算并更新ER

        Args:
            price_series: 价格序列

        Returns:
            er: 当前ER值
        """
        try:
            er_val = self.er_calc.calc(price_series)
        except ValueError:
            er_val = 0.0
        self.history.append(er_val)
        self._current_er = er_val
        return er_val

    def current_er(self) -> Optional[float]:
        return self._current_er

    def is_high_noise(self, er_val: Optional[float] = None) -> bool:
        """判断是否为高噪声环境

        ER < 低分位阈值 → 高噪声，信号不可靠

        Args:
            er_val: 当前ER值，None则用缓存

        Returns:
            True: 高噪声，信号不可靠
        """
        if er_val is not None:
            self._current_er = er_val
        if self._current_er is None or len(self.history) < 50:
            return False  # 数据不足，保守判断
        threshold = np.percentile(list(self.history), self.low_percentile * 100)
        return self._current_er < threshold

    def is_low_noise(self, er_val: Optional[float] = None) -> bool:
        """判断是否为低噪声环境

        ER > 高分位阈值 → 低噪声

        Args:
            er_val: 当前ER值，None则用缓存

        Returns:
            True: 低噪声
        """
        if er_val is not None:
            self._current_er = er_val
        if self._current_er is None or len(self.history) < 50:
            return True
        threshold = np.percentile(list(self.history), self.high_percentile * 100)
        return self._current_er > threshold

    def get_noise_ratio(self) -> float:
        """返回当前ER在历史中的分位 [0,1]

        返回值解读：
        - 0.0  = 最低分位（最噪声）
        - 0.5  = 中位数
        - 1.0  = 最高分位（最有效）
        - 值>er_low_percentile(0.30) 才允许信号通过

        判定逻辑（在 FusionRules 中实现）：
        noise_ratio < er_low_percentile → 高噪声，丢弃信号
        noise_ratio >= er_low_percentile → 信号通过
        """
        if self._current_er is None or len(self.history) < 2:
            return 0.5  # 数据不足，默认中性
        hist_arr = np.array(list(self.history))
        return float(np.mean(hist_arr <= self._current_er))

    def reset(self):
        self.history.clear()
        self._current_er = None


class ActiveStockFilter:
    """30分钟级别活跃股票筛选器

    日线引擎更新后，筛选出"结构活跃"的股票，
    仅对这些股票启用30分钟级别计算，将计算对象压缩到~300只。
    """

    def __init__(self, max_active: int = 300, top_fund_count: int = 100):
        self.max_active = max_active
        self.top_fund_count = top_fund_count

    def filter(
        self,
        chan_states: Dict[str, ChanStructureState],
        fund_scores: pd.Series,
    ) -> List[str]:
        """筛选30分钟级别活跃股票

        Args:
            chan_states: {symbol: ChanStructureState}
            fund_scores: 基本面得分Series (index=symbol, value=score)

        Returns:
            active_list: 需要计算30分钟级别的股票列表
        """
        active: Set[str] = set()

        # 1. 结构活跃
        for sym, state in chan_states.items():
            if state.is_in_zs() or state.is_diverging() or state.has_recent_breakout():
                active.add(sym)

        # 2. 基本面高分
        if fund_scores is not None and len(fund_scores) > 0:
            top_fund = set(fund_scores.nlargest(self.top_fund_count).index)
            active.update(top_fund)

        # 3. 限制数量
        return list(active)[:self.max_active]
