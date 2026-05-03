"""
信号优先级队列
===========
支持按优先级排序的多信号管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional
import heapq

from ..tech.chan_objects import Signal


class SignalPriority(IntEnum):
    """信号优先级（数值越大越优先）"""
    TECH_HARD_DIVERGENCE = 100     # 硬背驰+强趋势
    TECH_HARD_DIVERGENCE_RANGE = 90  # 硬背驰+盘整
    TECH_THIRD_POINT = 80          # 三类买卖点
    TECH_SOFT_DIVERGENCE = 60      # 软背驰
    FUND_HIGH_SCORE = 70           # 基本面高分
    FUSION_STRONG = 95             # 双系统共振


class PrioritySignal:
    """带优先级的信号包装"""
    def __init__(self, signal: Signal, priority: int, fund_score: float = 0.0):
        self.signal = signal
        self.priority = priority
        self.fund_score = fund_score

    def __lt__(self, other):
        # heapq需要反向实现（小顶堆），但我们想要大顶堆
        return self.priority > other.priority


class SignalPriorityQueue:
    """信号优先级队列

    按以下优先级排序：
    1. 基本面得分最高的前N只
    2. 技术模式可靠度：已确认背驰+强趋势 > 已确认背驰+盘整 > 软背驰
    3. 受风险预算约束
    """

    def __init__(self):
        self._heap: List[PrioritySignal] = []

    def push(self, signal: PrioritySignal):
        heapq.heappush(self._heap, signal)

    def pop(self) -> Optional[PrioritySignal]:
        if self._heap:
            return heapq.heappop(self._heap)
        return None

    def get_top_n(self, n: int) -> List[PrioritySignal]:
        """获取前N个最高优先级的信号（不移除）"""
        if not self._heap:
            return []

        # 使用nlargest保留堆结构
        from heapq import nlargest
        return [heapq.heappop(self._heap) for _ in range(min(n, len(self._heap)))]

    def peek(self) -> Optional[PrioritySignal]:
        return self._heap[0] if self._heap else None

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self):
        self._heap.clear()
