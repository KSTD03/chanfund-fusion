"""
缠论核心数据类定义
================
Bi(笔)、FX(分型)、ZS(中枢)、Divergence(背驰)、Signal(交易信号)
所有对象均携带 confirmed_time 字段防未来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import List, Optional


class SignalStatus(Enum):
    """信号生命周期状态"""
    PENDING = auto()       # 待确认（挂在pending列表中）
    CONFIRMED = auto()     # 已确认，等待执行
    EXECUTED = auto()      # 已执行开/平仓
    INVALIDATED = auto()   # 被后续结构失效
    EXPIRED = auto()       # 超时未执行
    REJECTED = auto()      # 【v1.1】被过滤器拒绝（成交量/风险等）


class FxType(Enum):
    """分型类型"""
    DING = "ding"   # 顶分型
    DI = "di"       # 底分型


@dataclass
class FX:
    """分型（FenXing）"""
    fx_type: FxType
    index: int               # K线索引位置
    time: date               # K线日期
    high: float
    low: float
    confirmed: bool = False
    confirmed_time: Optional[date] = None  # 分型确认时间


@dataclass
class Bi:
    """笔（Bi）"""
    direction: str               # "up" or "down"
    start_fx: FX                 # 起始分型
    end_fx: FX                   # 终止分型
    high: float
    low: float
    start_index: int
    end_index: int
    start_time: date
    end_time: date
    confirmed_time: Optional[date] = None   # 笔确认时间
    amplitude: float = 0.0       # 振幅 = high - low

    def __post_init__(self):
        self.amplitude = self.high - self.low

    @property
    def direction_sign(self) -> int:
        return 1 if self.direction == "up" else -1


@dataclass
class ZS:
    """中枢（ZhongShu）

    关键字段：
        zg: 中枢上沿（min of all up-bis' high）
        zd: 中枢下沿（max of all down-bis' low）
        gg: 中枢最高点（max of all bis' high）
        dd: 中枢最低点（min of all bis' low）
    """
    zg: float                      # 中枢上沿
    zd: float                      # 中枢下沿
    gg: float                      # 中枢最高点
    dd: float                      # 中枢最低点
    start_time: date
    end_time: Optional[date] = None
    status: str = "FORMING"        # FORMING -> ACTIVE -> EXPANDING / DESTROYED
    bi_list: List[Bi] = field(default_factory=list)
    confirmed_time: Optional[date] = None

    @property
    def height(self) -> float:
        return self.zg - self.zd

    @property
    def range(self) -> float:
        return self.gg - self.dd

    def on_new_bi(self, bi: Bi) -> Optional[ZS]:
        """根据新笔更新中枢状态，返回新中枢或None"""
        if self.status == "ACTIVE":
            if bi.high > self.gg or bi.low < self.dd:
                self.status = "DESTROYED"
                return None
            if bi.low > self.zd and bi.high < self.zg:
                self.status = "EXTENDING"
                self.bi_list.append(bi)
                self.end_time = bi.end_time
                return self
        return None


@dataclass
class Divergence:
    """背驰（BeiChi）"""
    bi: Bi
    divergence_type: str          # "top" or "bottom"
    is_hard: bool = False         # True: 硬背驰, False: 软背驰
    confirmed_time: Optional[date] = None
    strength: float = 0.0         # 背驰力度（面积比等）


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: str              # "buy" or "sell"
    signal_subtype: str           # "hard_divergence", "soft_divergence",
                                  # "third_point_buy", "second_class_buy",
                                  # "trend_follow", "stop_loss", etc.
    source: str                   # "tech", "fund", "fusion"
    price_confirmed: float        # 确认价格
    confirmed_time: date          # 信号确认日期（T+1生效）
    status: SignalStatus = SignalStatus.PENDING
    tech_score: float = 0.0       # 技术信号置信度 0-100
    fund_score: float = 0.0       # 基本面得分
    fused_score: float = 0.0      # 融合后得分
    details: dict = field(default_factory=dict)  # 扩展信息
    create_time: datetime = field(default_factory=datetime.now)

    def is_invalidated_by(self, event) -> bool:
        """检查是否被新事件失效（待实现）"""
        return False
