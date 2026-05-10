"""
MACD面积精确计算（防未来函数）
========================
核心原则：面积只计算到笔的确认点，不包含笔终点之后的未来数据
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def calc_macd_area_until_confirmed(
    bi_start_idx: int,
    confirmed_idx: int,
    macd_hist: np.ndarray,
) -> float:
    """计算MACD柱线面积（从笔起点到笔确认点）

    防未来关键：
    - 只使用到 confirmed_idx 的数据
    - 不包含笔终点之后的MACD值
    - 使用绝对值求和（柱线高度）

    Args:
        bi_start_idx: 笔的起始索引
        confirmed_idx: 笔的确认点索引（反向分型成立的前一根K线）
        macd_hist: MACD柱线序列

    Returns:
        area: MACD柱线面积（绝对值之和）
    """
    if confirmed_idx <= bi_start_idx:
        return 0.0

    # 严格截断：只取从笔起点到确认点的数据
    segment = macd_hist[bi_start_idx:confirmed_idx]
    return float(np.sum(np.abs(segment)))


def calc_macd_area_for_bi(
    bi_start_idx: int,
    bi_end_idx: int,
    confirmed_idx: int,
    macd_hist: np.ndarray,
) -> float:
    """计算整笔的MACD面积（含安全校验）

    比较两笔背驰时使用。

    Args:
        bi_start_idx: 笔起始
        bi_end_idx: 笔结束（含未来数据，仅用于校验）
        confirmed_idx: 笔确认点
        macd_hist: MACD柱线序列

    Returns:
        area: 安全截断后的面积
    """
    # 安全边界：确认点不应该超过终点
    safe_end = min(confirmed_idx, bi_end_idx)
    return calc_macd_area_until_confirmed(bi_start_idx, safe_end, macd_hist)


class MACDAreaCalculator:
    """MACD面积计算器（带缓存）"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calc_macd(self, close: np.ndarray) -> tuple:
        """计算MACD指标

        Returns:
            (dif, dea, macd_hist): MACD三线
        """
        import pandas_ta as ta
        # 尝试使用pandas_ta
        try:
            result = ta.ema(close, length=self.fast) - ta.ema(close, length=self.slow)
            # 简化的EMA计算（避免额外依赖）
            ema_fast = self._ema(close, self.fast)
            ema_slow = self._ema(close, self.slow)
            dif = ema_fast - ema_slow
            dea = self._ema(dif, self.signal)
            macd_hist = 2 * (dif - dea)
            return dif, dea, macd_hist
        except ImportError:
            pass

        # 独立实现
        ema_fast = self._ema(close, self.fast)
        ema_slow = self._ema(close, self.slow)
        dif = ema_fast - ema_slow
        dea = self._ema(dif, self.signal)
        macd_hist = 2 * (dif - dea)
        return dif, dea, macd_hist

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """指数移动平均实现"""
        result = np.zeros_like(data)
        if len(data) == 0:
            return result
        multiplier = 2.0 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
