"""
信号缓存与状态持久化 — SignalCache
===========================
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

# 直接导入避免相对导入问题
# 注意：实际使用时应通过包结构导入
try:
    from tech.chan_objects import Signal
except ImportError:
    # 作为独立模块时提供桩
    Signal = None


class SignalCache:
    """信号缓存器

    用于暂存未处理的信号，支持按日期查询和批量清理。
    """

    def __init__(self):
        # {symbol: {signal_id: Signal}}
        self._cache: Dict[str, Dict[int, "Signal"]] = defaultdict(dict)
        self._next_id: int = 0

    def add(self, signal: "Signal") -> int:
        """添加信号，返回ID"""
        sig_id = self._next_id
        self._next_id += 1
        self._cache[signal.symbol][sig_id] = signal
        return sig_id

    def get_by_symbol(self, symbol: str) -> list:
        """获取某只股票的所有信号"""
        return list(self._cache.get(symbol, {}).values())

    def get_by_date(self, trade_date: date) -> list:
        """获取某日期的所有信号"""
        result = []
        for sigs in self._cache.values():
            for sig in sigs.values():
                if sig.confirmed_time and sig.confirmed_time <= trade_date:
                    result.append(sig)
        return result

    def get_pending(self, trade_date: date) -> list:
        """获取可执行的pending信号"""
        return [
            sig for sig in self.get_by_date(trade_date)
            if sig.status.name == "PENDING"
        ]

    def update_status(self, signal_id: int, new_status) -> bool:
        """更新信号状态"""
        for sigs in self._cache.values():
            if signal_id in sigs:
                sigs[signal_id].status = new_status
                return True
        return False

    def clean(self, max_age_days: int = 365):
        """清理过期信号"""
        cutoff = date.today() - timedelta(days=max_age_days)
        for symbol in list(self._cache.keys()):
            self._cache[symbol] = {
                sid: sig for sid, sig in self._cache[symbol].items()
                if sig.confirmed_time and sig.confirmed_time > cutoff
            }
            if not self._cache[symbol]:
                del self._cache[symbol]

    def reset(self):
        self._cache.clear()
        self._next_id = 0
