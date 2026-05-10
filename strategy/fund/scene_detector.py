"""
宏观场景识别 — SceneDetector
=======================
基于三个指标的流程化场景判定：
1. 中证全指波动率（年化波动率及历史分位）
2. 信用利差 Z-Score
3. 价值动量 Z-Score

输出四种场景：防御避险 / 质量成长 / 深度价值 / 均衡中性
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd


class MacroScene(Enum):
    """宏观场景"""
    DEFENSIVE = "defensive"          # 防御避险
    QUALITY_GROWTH = "quality_growth"  # 质量成长
    DEEP_VALUE = "deep_value"        # 深度价值
    BALANCED = "balanced"            # 均衡中性


@dataclass
class SceneResult:
    """场景判定结果"""
    scene: MacroScene
    vol_annualized: float = 0.0
    vol_percentile: float = 0.0
    credit_z: float = 0.0
    value_momentum_z: float = 0.0
    details: dict = field(default_factory=dict)


class SceneDetector:
    """宏观场景检测器

    三个判定指标：
    a) 波动率：中证全指 (000985) 日收益率年化波动率
    b) 信用利差 Z-Score：信用债 - 国债
    c) 价值动量 Z-Score：EP前20% - 后20%的20日滚动累计
    """

    def __init__(self, params: dict):
        self.params = params
        # 阈值
        self.vol_high = params.get("vol_high", 0.25)
        self.vol_low = params.get("vol_low", 0.18)
        self.vol_window = params.get("vol_window", 20)
        self.credit_z_high = params.get("credit_z_high", 1.5)
        self.credit_z_low = params.get("credit_z_low", -1.5)
        self.value_momentum_z = params.get("value_momentum_z", 1.0)
        self.vol_extreme_percentile = params.get("vol_extreme_percentile", 0.95)

        # 历史波动率滚动序列（用于分位计算）
        self._vol_history: list = []

    def detect(
        self,
        returns: np.ndarray,           # 中证全指日收益率序列（最近500天）
        credit_spread: np.ndarray,     # 信用利差序列（最近20天）
        ep_top_minus_bottom: np.ndarray,  # 价值动量序列（最近1年）
    ) -> SceneResult:
        """综合场景判定

        流程化逻辑：
        1. (波动率>25% 且 信用Z>1.5) 或 波动率分位>95% → 防御避险
        2. 否则 (波动率<18% 且 信用Z<-1.5 且 价值动量Z<-0.5) → 质量成长
        3. 否则 (价值动量Z>1.5 且 波动率<30%) → 深度价值
        4. 其余 → 均衡中性
        """
        # --- a) 波动率 ---
        if len(returns) >= self.vol_window:
            vol = np.std(returns[-self.vol_window:]) * np.sqrt(252)
        else:
            vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        # 更新波动率历史
        self._vol_history.append(vol)
        if len(self._vol_history) > 500:
            self._vol_history = self._vol_history[-500:]

        # 波动率分位
        if len(self._vol_history) >= 20:
            vol_percentile = np.mean(np.array(self._vol_history) <= vol)
        else:
            vol_percentile = 0.5

        # --- b) 信用利差 Z-Score ---
        if len(credit_spread) >= 20:
            credit_z = (credit_spread[-1] - np.mean(credit_spread[-20:])) / np.std(credit_spread[-20:])
            if np.isnan(credit_z):
                credit_z = 0.0
        else:
            credit_z = 0.0

        # --- c) 价值动量 Z-Score ---
        if len(ep_top_minus_bottom) >= 20:
            vm_mean = np.mean(ep_top_minus_bottom[-20:])
            vm_std = np.std(ep_top_minus_bottom[-20:])
            if vm_std > 0:
                # 滚动20日累计和，相对年度Z-Score
                cum_sum = np.sum(ep_top_minus_bottom[-20:])
                value_momentum_z = (cum_sum - vm_mean) / vm_std
            else:
                value_momentum_z = 0.0
        else:
            value_momentum_z = 0.0

        # --- 综合决策 ---
        scene = self._decide(vol, vol_percentile, credit_z, value_momentum_z)

        return SceneResult(
            scene=scene,
            vol_annualized=vol,
            vol_percentile=vol_percentile,
            credit_z=credit_z,
            value_momentum_z=value_momentum_z,
            details={
                "vol_window": self.vol_window,
                "vol_high_threshold": self.vol_high,
                "vol_low_threshold": self.vol_low,
                "credit_z_high": self.credit_z_high,
                "credit_z_low": self.credit_z_low,
            },
        )

    def _decide(
        self,
        vol: float,
        vol_percentile: float,
        credit_z: float,
        value_momentum_z: float,
    ) -> MacroScene:
        """流程化场景综合决策"""
        # 规则1: 防御避险
        if (vol > self.vol_high and credit_z > self.credit_z_high) or vol_percentile > self.vol_extreme_percentile:
            return MacroScene.DEFENSIVE

        # 规则2: 质量成长
        if vol < self.vol_low and credit_z < self.credit_z_low and value_momentum_z < -0.5:
            return MacroScene.QUALITY_GROWTH

        # 规则3: 深度价值
        if value_momentum_z > 1.5 and vol < 0.30:
            return MacroScene.DEEP_VALUE

        # 规则4: 均衡
        return MacroScene.BALANCED

    def reset(self):
        """重置历史状态"""
        self._vol_history.clear()
