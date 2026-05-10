"""
PIT数据管道 — PITDataPipeline
========================
Point-in-Time 数据管道，确保回测中的数据使用和当时实际可用的一致。

关键：基于 report_publish_date 判断财报可用性，避免未来函数。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PITDataPipeline:
    """PIT（Point-in-Time）数据管道

    职责：
    1. 管理财报披露日期表
    2. 查询在指定日期已披露的最新财报
    3. 确保回测中不会使用未来的数据
    """

    def __init__(self):
        # {symbol: [{period_end, publish_date, data_dict}]}
        self._fin_reports: Dict[str, List[dict]] = {}
        self._initialized = False

    def load_financial_reports(
        self,
        df: pd.DataFrame,
        symbol_col: str = "symbol",
        period_col: str = "period_end_date",
        publish_col: str = "publish_date",
    ):
        """加载财报数据

        Args:
            df: DataFrame，必须包含 symbol, period_end_date, publish_date 列
            symbol_col: 股票代码列名
            period_col: 财报期间列名
            publish_col: 披露日期列名
        """
        self._fin_reports.clear()
        for _, row in df.iterrows():
            symbol = row[symbol_col]
            report = {
                "period_end_date": row[period_col],
                "publish_date": row[publish_col],
                "data": row.drop([symbol_col, period_col, publish_col]).to_dict(),
            }
            if symbol not in self._fin_reports:
                self._fin_reports[symbol] = []
            self._fin_reports[symbol].append(report)

        # 按披露日期排序
        for symbol in self._fin_reports:
            self._fin_reports[symbol].sort(key=lambda x: x["publish_date"])
        self._initialized = True
        logger.info(f"Loaded {len(df)} financial reports for {len(self._fin_reports)} symbols")

    def get_latest_report(
        self,
        symbol: str,
        as_of: date,
        after_period: date = date(2000, 1, 1),
    ) -> Optional[dict]:
        """获取指定日期之前已披露的最新财报

        Args:
            symbol: 股票代码
            as_of: 查询日期（使用该日期及之前已披露的数据）
            after_period: 只查询报告期 > 此日期的财报

        Returns:
            report: 最新的财报数据，或 None
        """
        if not self._initialized or symbol not in self._fin_reports:
            return None

        # 找到在 as_of 之前（或当天）披露，且报告期 > after_period 的最新报告
        best_report = None
        best_publish = date(2000, 1, 1)

        for r in self._fin_reports[symbol]:
            publish = r.get("publish_date", r["period_end_date"])
            period_end = r["period_end_date"]

            if publish <= as_of and period_end > after_period:
                if publish > best_publish:
                    best_report = r
                    best_publish = publish

        return best_report

    def has_new_financial(self, symbol: str, as_of: date, last_check: date) -> bool:
        """检查是否有新的财报披露"""
        if not self._initialized or symbol not in self._fin_reports:
            return False
        for r in self._fin_reports[symbol]:
            publish = r.get("publish_date", r["period_end_date"])
            if last_check < publish <= as_of:
                return True
        return False

    def get_latest_financial_field(
        self, symbol: str, field: str, as_of: date
    ) -> Optional[float]:
        """获取最新财报的某个字段值"""
        report = self.get_latest_report(symbol, as_of)
        if report is None:
            return None
        return report["data"].get(field)
