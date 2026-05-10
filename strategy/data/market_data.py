"""
市场数据获取 — MarketDataProvider
=========================
封装行情数据获取逻辑，提供统一的数据接口。
支持Qlib数据源、Baostock数据源等。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """市场数据提供者

    从Qlib或其他数据源获取行情数据，统一接口输出。
    支持多频（日线、30分钟线）。
    """

    def __init__(self, config: dict):
        self.config = config
        self.start_date = config.get("start_date", "2017-01-01")
        self.end_date = config.get("end_date", "2025-12-31")
        self.benchmark = config.get("benchmark", "SH000300")

        # Qlib数据实例（由外部初始化后传入）
        self.qlib_provider = None
        self._initialized = False

    def init_from_qlib(self, qlib_instance):
        """使用Qlib作为数据源"""
        self.qlib_provider = qlib_instance
        self._initialized = True
        logger.info("MarketDataProvider initialized with Qlib")

    def get_kbars(
        self,
        symbol: str,
        start: str,
        end: str,
        freq: str = "day",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取K线数据

        Args:
            symbol: 股票代码
            start: 起始日期 "YYYY-MM-DD"
            end: 结束日期 "YYYY-MM-DD"
            freq: "day" 或 "30min"
            fields: 需要的字段，默认 ["open", "high", "low", "close", "volume"]

        Returns:
            df: K线DataFrame，index=datetime, columns=fields
        """
        if self.qlib_provider is not None:
            return self._get_from_qlib(symbol, start, end, freq, fields)

        # 备用：生成模拟数据（用于开发测试）
        logger.warning(f"No data provider for {symbol}, using mock data")
        return self._mock_kbars(symbol, start, end, freq)

    def _get_from_qlib(
        self, symbol: str, start: str, end: str,
        freq: str, fields: Optional[List[str]],
    ) -> pd.DataFrame:
        """从Qlib获取数据"""
        if fields is None:
            fields = ["open", "high", "low", "close", "volume"]

        try:
            from qlib.data import D
            from qlib.data.dataset import Dataset

            # Qlib的code格式: symbol + 交易所后缀
            qlib_code = self._to_qlib_code(symbol)

            # 使用Qlib的feature表达式
            expressions = []
            for f in fields:
                if f == "open":
                    expressions.append("$open")
                elif f == "high":
                    expressions.append("$high")
                elif f == "low":
                    expressions.append("$low")
                elif f == "close":
                    expressions.append("$close")
                elif f == "volume":
                    expressions.append("$volume")
                else:
                    expressions.append(f"${f}")

            df = D.features(
                [qlib_code],
                expressions,
                start_time=start,
                end_time=end,
                freq=freq,
            )

            if df.empty:
                return pd.DataFrame()

            # 重命名列
            df.columns = fields
            return df

        except Exception as e:
            logger.error(f"Qlib data fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _to_qlib_code(self, symbol: str) -> str:
        """转换股票代码为Qlib格式"""
        # 简单转换：SH/SZ前缀
        if symbol.startswith("6"):
            return f"SH{symbol}"
        elif symbol.startswith(("0", "3")):
            return f"SZ{symbol}"
        elif symbol.startswith("4"):
            return f"BJ{symbol}"
        elif symbol.startswith("8"):
            return f"BJ{symbol}"
        return symbol

    def _mock_kbars(
        self, symbol: str, start: str, end: str, freq: str = "day"
    ) -> pd.DataFrame:
        """生成模拟K线数据用于开发测试"""
        import random
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        if freq == "day":
            dates = pd.date_range(start_dt, end_dt, freq="B")
        elif freq == "30min":
            dates = pd.date_range(start_dt, end_dt, freq="30min")
        else:
            dates = pd.date_range(start_dt, end_dt, freq="B")

        base = 10.0 + random.random() * 20
        prices = base + np.cumsum(np.random.randn(len(dates)) * 0.5)
        prices = np.maximum(prices, 1.0)

        df = pd.DataFrame(index=dates[:len(prices)])
        df["open"] = prices
        df["high"] = prices * (1 + np.abs(np.random.randn(len(prices)) * 0.02))
        df["low"] = prices * (1 - np.abs(np.random.randn(len(prices)) * 0.02))
        df["close"] = prices * (1 + np.random.randn(len(prices)) * 0.01)
        df["volume"] = np.random.randint(10000, 10000000, size=len(prices))
        df["amount"] = df["close"] * df["volume"]

        return df

    def get_universe(self, date: str = None) -> List[str]:
        """获取股票池"""
        if self.qlib_provider is not None:
            try:
                from qlib.data import D
                instruments = D.instruments(market="all")
                return list(instruments)
            except Exception:
                pass
        # 模拟股票池
        return [f"{i:06d}" for i in range(600000, 600100)]

    def get_close_series(self, symbol: str, start: str, end: str, freq: str = "day") -> np.ndarray:
        """获取收盘价序列"""
        df = self.get_kbars(symbol, start, end, freq, ["close"])
        if df.empty:
            return np.array([])
        return df["close"].values

    def update_end_date(self, new_end: str):
        self.end_date = new_end
