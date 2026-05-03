"""
增量缠论结构计算
=============
ChanStructureState — 每只股票维护一个历史结构栈，增量更新。
事件驱动顺序：更新 → 检测 → 确认 → 中枢 → 信号
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

from .chan_objects import (
    Bi, FX, FxType, ZS, Divergence,
    Signal, SignalStatus,
)

logger = logging.getLogger(__name__)


class Event:
    """内部事件基类"""
    pass


class BiConfirmedEvent(Event):
    def __init__(self, bi: Bi):
        self.bi = bi


class ZSUpdatedEvent(Event):
    def __init__(self, zs: ZS, action: str):
        self.zs = zs
        self.action = action  # "formed", "extended", "destroyed", "expanded"


class SignalGenEvent(Event):
    def __init__(self, signal: Signal):
        self.signal = signal


class ChanStructureState:
    """单只股票的增量缠论结构状态机

    核心思想：每根新K线触发以下5个事件，严格按序：
        事件1 — 更新：用新K线极值更新当前笔
        事件2 — 检测：检测是否形成新潜在分型
        事件3 — 确认：若分型成立且反向，确认旧笔、开启新笔
        事件4 — 中枢：基于已确认笔序列，检查中枢状态变化
        事件5 — 信号：检查背驰或三类买卖点
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        # 当前笔（正在构建中）
        self.current_bi: Optional[Bi] = None
        # 待确认的潜在分型
        self.pending_fx: Optional[FX] = None
        # 前一根K线（用于分型判断）
        self.prev_kbar: Optional[dict] = None
        # 上上根K线（用于分型判断）
        self.prev_prev_kbar: Optional[dict] = None
        # 已确认的笔列表（按时间顺序）
        self.confirmed_bis: List[Bi] = []
        # 已确认的中枢列表
        self.zs_list: List[ZS] = []
        # K线计数器
        self.kbar_count: int = 0
        # 最近一次事件的K线信息
        self.last_kbar: Optional[dict] = None
        # 配置参数
        self.bi_min_kbar: int = 5
        self.bi_min_kbar_dynamic: bool = True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def process_kbar(
        self, kbar: dict, emit_signals: bool = True
    ) -> List[Event]:
        """增量处理一根新K线，返回本K线触发的所有事件

        Args:
            kbar: 包含 open, high, low, close, volume, time 的字典
            emit_signals: 是否生成交易信号（预热阶段设为False）

        Returns:
            events: 事件列表
        """
        events: List[Event] = []
        self.kbar_count += 1
        self.last_kbar = kbar

        # ---- 事件1：更新当前笔极值 ----
        self._update_current_bi(kbar)

        # ---- 事件2：检测新分型 ----
        new_fx = self._detect_fx(kbar)

        # ---- 事件3：笔管理与确认 ----
        if new_fx is not None:
            if self.current_bi is None:
                # 第一笔初始化：以当前分型方向创建新笔
                direction = "up" if new_fx.fx_type == FxType.DI else "down"
                self.current_bi = Bi(
                    direction=direction,
                    start_fx=new_fx,
                    end_fx=new_fx,
                    high=kbar["high"],
                    low=kbar["low"],
                    start_index=self.kbar_count,
                    end_index=self.kbar_count,
                    start_time=kbar.get("time", date.min),
                    end_time=kbar.get("time", date.min),
                )
            elif new_fx.fx_type.value != self.current_bi.direction:
                # 反向分型 = 当前笔完成
                confirmed_bi = self._confirm_bi(new_fx, kbar)
                if confirmed_bi is not None:
                    events.append(BiConfirmedEvent(confirmed_bi))

                    # ---- 事件4：中枢检查 ----
                    zs_events = self._update_zs(confirmed_bi, kbar)
                    events.extend(zs_events)

                    # ---- 事件5：信号检查 ----
                    if emit_signals:
                        sig_events = self._check_signal(confirmed_bi, kbar)
                        events.extend(sig_events)

        # 更新K线缓存用于分型判断
        self.prev_prev_kbar = self.prev_kbar
        self.prev_kbar = kbar

        return events

    # ------------------------------------------------------------------
    # 事件1：更新当前笔
    # ------------------------------------------------------------------
    def _update_current_bi(self, kbar: dict):
        """用新K线极值更新当前笔"""
        if self.current_bi is None:
            # 新建一笔（由分型触发，不在此处）
            return
        if kbar["high"] > self.current_bi.high:
            self.current_bi.high = kbar["high"]
        if kbar["low"] < self.current_bi.low:
            self.current_bi.low = kbar["low"]

    # ------------------------------------------------------------------
    # 事件2：分型检测
    # ------------------------------------------------------------------
    def _detect_fx(self, kbar: dict) -> Optional[FX]:
        """检测当前K线是否形成新分型

        顶分型条件：高点 > 前一根高点 且 高点 > 后一根高点
        底分型条件：低点 < 前一根低点 且 低点 < 后一根低点
        这里使用三根K线：prev_prev, prev, current
        """
        if self.prev_prev_kbar is None or self.prev_kbar is None:
            return None

        pp = self.prev_prev_kbar
        p = self.prev_kbar
        c = kbar

        # 顶分型：中间K线最高
        if p["high"] > pp["high"] and p["high"] > c["high"]:
            fx = FX(
                fx_type=FxType.DING,
                index=self.kbar_count - 1,
                time=p.get("time", date.min),
                high=p["high"],
                low=p["low"],
            )
            return fx

        # 底分型：中间K线最低
        if p["low"] < pp["low"] and p["low"] < c["low"]:
            fx = FX(
                fx_type=FxType.DI,
                index=self.kbar_count - 1,
                time=p.get("time", date.min),
                high=p["high"],
                low=p["low"],
            )
            return fx

        return None

    # ------------------------------------------------------------------
    # 事件3：确认笔
    # ------------------------------------------------------------------
    def _confirm_bi(self, new_fx: FX, kbar: dict) -> Bi:
        """确认当前笔完成，创建新笔

        Args:
            new_fx: 新检测到的反向分型
            kbar: 当前K线

        Returns:
            confirmed_bi: 刚确认完成的笔
        """
        # 完成当前笔
        completed_bi = self.current_bi

        # 创建新笔
        direction = "up" if new_fx.fx_type == FxType.DI else "down"
        new_bi = Bi(
            direction=direction,
            start_fx=new_fx,
            end_fx=new_fx,  # 临时，K线更新时会修正
            high=kbar["high"],
            low=kbar["low"],
            start_index=self.kbar_count,
            end_index=self.kbar_count,
            start_time=kbar.get("time", date.min),
            end_time=kbar.get("time", date.min),
        )
        self.current_bi = new_bi

        # 标记旧笔完成
        if completed_bi is not None:
            completed_bi.end_fx = new_fx
            completed_bi.end_index = self.kbar_count - 1
            completed_bi.end_time = self.prev_kbar.get("time", date.min) if self.prev_kbar else date.min
            completed_bi.confirmed_time = completed_bi.end_time
            self.confirmed_bis.append(completed_bi)

        return completed_bi if completed_bi is not None else new_bi

    # ------------------------------------------------------------------
    # 事件4：中枢更新
    # ------------------------------------------------------------------
    def _update_zs(self, confirmed_bi: Bi, kbar: dict) -> List[Event]:
        """基于新确认的笔 + 当前K线OHLC更新中枢状态

        重要补丁：除了检查笔，还需检查单根K线是否直接击穿中枢DD/GG
        """
        events: List[Event] = []
        bis = self.confirmed_bis

        # 需要至少3笔才能形成中枢
        if len(bis) < 3:
            return events

        # --- 检查现有中枢是否被单根K线击穿 ---
        for zs in self.zs_list:
            if zs.status == "ACTIVE":
                if kbar["low"] < zs.dd or kbar["high"] > zs.gg:
                    old_status = zs.status
                    zs.status = "DESTROYED"
                    events.append(ZSUpdatedEvent(zs, "destroyed"))
                    # 检查是否构成第三类买卖点
                    self._check_third_point_signal(zs, kbar, events)

        # --- 检查最近3笔能否构成新中枢 ---
        recent_bis = bis[-3:]
        if self._is_zs_overlap(recent_bis):
            # 计算中枢区间
            up_bis = [b for b in recent_bis if b.direction == "up"]
            down_bis = [b for b in recent_bis if b.direction == "down"]
            if up_bis and down_bis:
                up_lows = [b.low for b in up_bis]
                up_highs = [b.high for b in up_bis]
                down_lows = [b.low for b in down_bis]
                down_highs = [b.high for b in down_bis]
                zg = min(up_highs)   # 中枢上沿：向上笔高点的最小值
                zd = max(down_lows)  # 中枢下沿：向下笔低点的最大值
                if zg > zd:  # 有效中枢
                    zs = ZS(
                        zg=zg,
                        zd=zd,
                        gg=max(up_highs + down_highs),
                        dd=min(up_lows + down_lows),
                        start_time=recent_bis[0].start_time,
                        status="ACTIVE",
                        bi_list=list(recent_bis),
                        confirmed_time=confirmed_bi.confirmed_time,
                    )
                    self.zs_list.append(zs)
                    events.append(ZSUpdatedEvent(zs, "formed"))

        return events

    def _is_zs_overlap(self, bis: List[Bi]) -> bool:
        """检查连续三笔是否有重叠区间"""
        if len(bis) < 3:
            return False
        b1, b2, b3 = bis[-3:]
        # 检查笔1和笔3是否有重叠
        overlap_high = min(b1.high, b3.high)
        overlap_low = max(b1.low, b3.low)
        return overlap_high > overlap_low

    def _check_third_point_signal(self, zs: ZS, kbar: dict, events: List[Event]):
        """检查中枢被破坏时是否构成第三类买卖点

        三买：价格离开中枢后回抽不进入中枢区间（价格 > zg）
        三卖：价格离开中枢后回抽不进入中枢区间（价格 < zd）
        """
        # 简化实现：用当前K线收盘价判断
        close = kbar.get("close", (kbar["high"] + kbar["low"]) / 2)
        if close > zs.zg:
            # 可能的第三类买点
            sig = Signal(
                symbol=self.symbol,
                signal_type="buy",
                signal_subtype="third_point_buy",
                source="tech",
                price_confirmed=close,
                confirmed_time=kbar.get("time", date.min),
                details={"zs_zg": zs.zg, "zs_zd": zs.zd, "ref_bi_low": zs.dd, "ref_zs_zg": zs.zg},
            )
            events.append(SignalGenEvent(sig))
        elif close < zs.zd:
            # 可能的第三类卖点
            sig = Signal(
                symbol=self.symbol,
                signal_type="sell",
                signal_subtype="third_point_sell",
                source="tech",
                price_confirmed=close,
                confirmed_time=kbar.get("time", date.min),
                details={"zs_zg": zs.zg, "zs_zd": zs.zd},
            )
            events.append(SignalGenEvent(sig))

    # ------------------------------------------------------------------
    # 事件5：信号检查
    # ------------------------------------------------------------------
    def _check_signal(self, confirmed_bi: Bi, kbar: dict) -> List[Event]:
        """检查当前笔确认后是否产生背驰信号或买卖点

        背驰判断：比较相邻两段同向笔的力度（MACD面积/振幅等）
        """
        events: List[Event] = []
        bis = self.confirmed_bis

        if len(bis) < 2:
            return events

        # 取最近两笔（必须方向相同才能比较背驰）
        b_current = bis[-1]
        b_prev = bis[-2]

        if b_current.direction != b_prev.direction:
            # 方向不同，跳过
            return events

        # 检查是否有中枢作为背驰比较的"锚"
        has_zs = len(self.zs_list) > 0

        # 背驰判断：当前笔的幅度小于前一笔 => 可能存在背驰
        is_weaker = b_current.amplitude < b_prev.amplitude * 0.8

        if is_weaker and has_zs:
            if b_current.direction == "up" and b_current.high > b_prev.high:
                # 顶背驰
                divergence = Divergence(
                    bi=b_current,
                    divergence_type="top",
                    is_hard=True,
                    confirmed_time=b_current.confirmed_time,
                    strength=(b_prev.amplitude - b_current.amplitude) / b_prev.amplitude,
                )
                sig = Signal(
                    symbol=self.symbol,
                    signal_type="sell",
                    signal_subtype="hard_divergence",
                    source="tech",
                    price_confirmed=kbar.get("close", kbar["high"]),
                    confirmed_time=b_current.confirmed_time or date.min,
                    details={"divergence_type": "top", "strength": divergence.strength, "ref_bi_low": b_current.low},
                )
                events.append(SignalGenEvent(sig))
            elif b_current.direction == "down" and b_current.low < b_prev.low:
                # 底背驰
                divergence = Divergence(
                    bi=b_current,
                    divergence_type="bottom",
                    is_hard=True,
                    confirmed_time=b_current.confirmed_time,
                    strength=(b_prev.amplitude - b_current.amplitude) / b_prev.amplitude,
                )
                sig = Signal(
                    symbol=self.symbol,
                    signal_type="buy",
                    signal_subtype="hard_divergence",
                    source="tech",
                    price_confirmed=kbar.get("close", kbar["low"]),
                    confirmed_time=b_current.confirmed_time or date.min,
                    details={"divergence_type": "bottom", "strength": divergence.strength, "ref_bi_low": b_current.low},
                )
                events.append(SignalGenEvent(sig))

        return events

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def is_in_zs(self) -> bool:
        """是否处于某个中枢区间内"""
        if not self.last_kbar or not self.zs_list:
            return False
        close = self.last_kbar.get("close",
                                    (self.last_kbar["high"] + self.last_kbar["low"]) / 2)
        for zs in self.zs_list:
            if zs.status in ("ACTIVE", "EXTENDING"):
                if zs.zd <= close <= zs.zg:
                    return True
        return False

    def is_diverging(self) -> bool:
        """是否处于背驰段（简化：笔数足够且最近方向相同）"""
        if len(self.confirmed_bis) < 4:
            return False
        recent = self.confirmed_bis[-2:]
        return recent[0].direction == recent[1].direction

    def has_recent_breakout(self) -> bool:
        """最近是否有突破（中三买卖点）"""
        if not self.zs_list:
            return False
        last_zs = self.zs_list[-1]
        if last_zs.status == "DESTROYED":
            return True
        return False

    def get_latest_zs(self) -> Optional[ZS]:
        """获取最近的中枢"""
        return self.zs_list[-1] if self.zs_list else None

    def get_latest_bi(self) -> Optional[Bi]:
        """获取最近确认的笔"""
        return self.confirmed_bis[-1] if self.confirmed_bis else self.current_bi
