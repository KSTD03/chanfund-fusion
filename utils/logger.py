"""
结构化决策日志 — DecisionLogger & RejectReason
========================================
实现日志系统自动化归因设计：
v1.0 新增：TREND_DIRECTION, RED_FLAG 拒绝原因
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class RejectReason(Enum):
    """信号拒绝原因枚举（用于归因分析）"""
    LOW_FUND_SCORE = "low_fund_score"
    LOW_FUSED_SCORE = "low_fused_score"
    COOLING_PERIOD = "cooling_period"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SECTOR_LIMIT = "sector_limit"
    HIGH_NOISE = "high_noise"
    JUMP_GAP = "jump_gap"
    STRUCTURE_INVALIDATED = "structure_invalidated"
    NO_TECH_SIGNAL = "no_tech_signal"
    ALREADY_HELD = "already_held"
    RED_FLAG = "red_flag"           # v1.0 新增：红牌排雷
    TREND_DIRECTION = "trend_direction"  # v1.0 新增：趋势方向过滤
    OTHER = "other"


@dataclass
class DecisionRecord:
    """单次决策记录"""
    date: date
    symbol: str
    decision: str           # "accepted" / "rejected"
    signal_type: str        # buy / sell signal
    reason: str             # 接收时的子类型或拒绝原因
    weight: float = 0.0     # 仓位权重
    fund_score: float = 0.0
    fused_score: float = 0.0
    details: str = ""


class DecisionLogger:
    """结构化决策日志器

    所有决策（接受/拒绝）都记录结构化数据，
    回测结束后可输出为DataFrame做归因分析。
    """

    def __init__(self):
        self.records: List[DecisionRecord] = []

    def log_accepted(
        self,
        symbol: str,
        signal_type: str,
        weight: float,
        fund_score: float,
        fused_score: float,
        trade_date: date,
        details: str = "",
    ):
        """记录被接受的信号"""
        record = DecisionRecord(
            date=trade_date,
            symbol=symbol,
            decision="accepted",
            signal_type=signal_type,
            reason=signal_type,
            weight=weight,
            fund_score=fund_score,
            fused_score=fused_score,
            details=details,
        )
        self.records.append(record)

    def log_rejected(
        self,
        symbol: str,
        reason: RejectReason,
        details: str,
        trade_date: date,
        signal_type: str = "unknown",
    ):
        """记录被拒绝的信号"""
        record = DecisionRecord(
            date=trade_date,
            symbol=symbol,
            decision="rejected",
            signal_type=signal_type,
            reason=reason.value,
            details=details,
        )
        self.records.append(record)

    def get_accepted(self) -> List[DecisionRecord]:
        """获取所有被接受的决策"""
        return [r for r in self.records if r.decision == "accepted"]

    def get_rejected(self) -> List[DecisionRecord]:
        """获取所有被拒绝的决策"""
        return [r for r in self.records if r.decision == "rejected"]

    def get_rejected_by_reason(self) -> dict:
        """按拒绝原因分组统计"""
        from collections import Counter
        return Counter(r.reason for r in self.get_rejected())

    def summary(self) -> str:
        """输出简短的决策统计摘要"""
        accepted = len(self.get_accepted())
        rejected = len(self.get_rejected())
        total = accepted + rejected
        if total == 0:
            return "No decisions recorded."

        reject_breakdown = "\n  ".join(
            f"{k}: {v} ({v/total*100:.1f}%)"
            for k, v in self.get_rejected_by_reason().most_common()
        )
        return (
            f"Total decisions: {total}\n"
            f"  Accepted: {accepted} ({accepted/total*100:.1f}%)\n"
            f"  Rejected: {rejected} ({rejected/total*100:.1f}%)\n"
            f"  Reject breakdown:\n  {reject_breakdown}"
        )

    def reset(self):
        self.records.clear()

    def to_dict(self) -> list:
        return [
            {
                "date": str(r.date),
                "symbol": r.symbol,
                "decision": r.decision,
                "signal_type": r.signal_type,
                "reason": r.reason,
                "weight": r.weight,
                "fund_score": r.fund_score,
                "fused_score": r.fused_score,
                "details": r.details,
            }
            for r in self.records
        ]

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
