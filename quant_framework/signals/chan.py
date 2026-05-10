"""
缠论信号检测器适配器（placeholder）

实际信号逻辑引用自 chanfund_fusion/fusion/rules.py
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ChanSignalDetector:
    """缠论信号检测器（占位）"""
    
    def __init__(self, config):
        self.config = config
    
    def update(self, trade_date: date, date_str: str = None):
        """更新状态机"""
        pass
    
    def scan(self, universe: List[str], trade_date: date, date_str: str) -> List[dict]:
        """扫描信号"""
        return []
