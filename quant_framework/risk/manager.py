"""
风险管理器

职责：
1. 止损检查
2. 分级止盈
3. 移动止损（追踪止盈）
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from quant_framework.config.config import BacktestConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.sl_pct = config.stop_loss_pct
        self.tp1_pct = config.take_profit_1
        self.tp2_pct = config.take_profit_2
        self.ts_pct = config.trailing_stop_pct
    
    def check_stop_loss(self, current_price: float, entry_price: float) -> bool:
        """检查是否触发止损"""
        if entry_price <= 0:
            return False
        loss_pct = (current_price / entry_price - 1)
        return loss_pct <= -self.sl_pct
    
    def check_take_profit(self, current_price: float, entry_price: float, 
                          position: dict) -> Optional[str]:
        """检查是否触发止盈
        
        Returns:
            None: 不触发
            "tp1": 第一止盈
            "tp2": 第二止盈  
            "ts": 移动止损
        """
        if entry_price <= 0:
            return None
        
        gain_pct = (current_price / entry_price - 1)
        
        # 移动止损（已触发过tp1）
        if position.get("trailing_active"):
            highest = position.get("highest_price", entry_price)
            if current_price > highest:
                position["highest_price"] = current_price
            elif (highest - current_price) / highest > self.ts_pct:
                return "ts"
            return None
        
        # tp2
        if gain_pct >= self.tp2_pct:
            return "tp2"
        
        # tp1
        if gain_pct >= self.tp1_pct:
            position["trailing_active"] = True
            position["highest_price"] = current_price
            return "tp1"
        
        return None
