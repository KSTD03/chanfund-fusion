"""
环境分类器

支持 MA规则 和 ER规则 两种模式。
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import date, timedelta
from typing import Dict, List, Optional
import numpy as np

from quant_framework.config.config import BacktestConfig

logger = logging.getLogger(__name__)


class EnvironmentClassifier:
    """市场环境分类器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.mode = config.env_classifier
        
        # 缓存
        self.close_history: Dict[str, deque] = {}  # {symbol: deque of closes}
        self.index_history: deque = deque(maxlen=max(config.env_params.get("ma_slow", 60) + 10, 252))
        
        # 状态
        self.current_env: str = "UNKNOWN"
        
        # 历史波动率
        self.returns: List[float] = []
        
    def update(self, trade_date: date) -> str:
        """更新市场环境判断"""
        if self.mode == "none":
            self.current_env = "BULL"
            return self.current_env
        
        # 简化的环境判断：用模拟序列
        # 实际使用时应从数据加载器传入指数价格
        self._simulate_env(trade_date)
        return self.current_env
    
    def _simulate_env(self, trade_date: date):
        """简化的环境判断（placeholder）
        
        实际环境分类需要指数数据（沪深300）。
        此处返回默认值，框架集成时替换为真实实现。
        """
        if self.mode == "ma" and self.index_history:
            closes = np.array(list(self.index_history))
            if len(closes) >= self.config.env_params.get("ma_slow", 60):
                ma_fast = np.mean(closes[-self.config.env_params.get("ma_fast", 20):])
                ma_slow = np.mean(closes[-self.config.env_params.get("ma_slow", 60):])
                
                # 波动率保护
                if len(closes) >= 20:
                    rets = np.diff(closes) / closes[:-1]
                    vol = np.std(rets[-20:]) * np.sqrt(252)
                    vol_threshold = self.config.env_params.get("vol_percentile", 0.85)
                    # 简化：这里直接返回
                
                if ma_fast > ma_slow:
                    self.current_env = "BULL"
                else:
                    self.current_env = "BEAR"
            else:
                self.current_env = "OSCILLATE"
        elif self.mode == "none":
            self.current_env = "BULL"
        else:
            self.current_env = "OSCILLATE"
