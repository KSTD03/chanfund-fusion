"""
统一回测引擎

职责：
1. 提供单一 run() 入口，接受 BacktestConfig
2. 解耦信号生成、环境分类、评分过滤
3. 统一结果收集与输出

用法：
    from quant_framework.core.engine import run_backtest
    from quant_framework.config.config import BacktestConfig
    
    config = BacktestConfig.load_v2_4_default()
    result = run_backtest(config)
    result.save("output/v2.4_framework/")
"""

from __future__ import annotations

import sys
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import pandas as pd

from quant_framework.config.config import BacktestConfig
from quant_framework.data.loader import DataLoader
from quant_framework.signals.chan import ChanSignalDetector
from quant_framework.environment.classifier import EnvironmentClassifier
from quant_framework.risk.manager import RiskManager
from quant_framework.output.reporter import BacktestReporter

logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.signal_detector = None  # 懒加载
        self.env_classifier = EnvironmentClassifier(config)
        self.risk_manager = RiskManager(config)
        self.reporter = BacktestReporter(config)
        
        # 运行时状态
        self.portfolio_value = config.initial_capital
        self.positions: Dict[str, dict] = {}
        self.trade_log: List[dict] = []
        self.equity_curve: List[dict] = []
        self.scores_cache: Dict = {}
        self.env_history: List[str] = []
        
    def run(self) -> BacktestReporter:
        """执行完整回测"""
        t0 = time.time()
        logger.info(f"{'='*55}")
        logger.info(f"🔰 ChanFund Fusion {self.config.name} 回测")
        logger.info(f"{'='*55}")
        
        # 1. 加载数据
        data = self.data_loader.load_all()
        
        # 2. 加载评分
        self.scores_cache = self.data_loader.load_scores()
        
        # 3. 加载行情
        all_dates, daily_data = self.data_loader.load_market_data(data)
        
        # 4. 预热阶段
        logger.info(f"预热 {self.config.warmup_days} 天...")
        warmup_end = self.config.warmup_days
        self._warmup(all_dates[:warmup_end], daily_data)
        
        # 5. 回测阶段
        backtest_dates = all_dates[warmup_end:warmup_end + self.config.backtest_days]
        logger.info(f"回测 {len(backtest_dates)} 天...")
        self._run_backtest(backtest_dates, daily_data)
        
        # 6. 生成报告
        elapsed = time.time() - t0
        self.reporter.collect_results(
            trade_log=self.trade_log,
            equity_curve=self.equity_curve,
            env_history=self.env_history,
            config=self.config,
            elapsed=elapsed,
        )
        self.reporter.save()
        
        logger.info(f"✅ {self.config.name} 回测完成 ({elapsed:.0f}s)")
        return self.reporter
    
    def _warmup(self, warmup_dates: List[date], daily_data: dict):
        """预热：启动缠论状态机和环境分类器"""
        for i, trade_date in enumerate(warmup_dates):
            date_str = str(trade_date)
            
            # 更新环境分类器
            self.env_classifier.update(trade_date)
            
            # 更新信号检测器（预热期）
            self.signal_detector.update(trade_date, date_str=date_str)
            
            if (i + 1) % 500 == 0:
                logger.info(f"   {trade_date}({i+1}/{len(warmup_dates)})")
    
    def _run_backtest(self, backtest_dates: List[date], daily_data: dict):
        """回测主循环"""
        total_dates = len(backtest_dates)
        report_interval = max(1, total_dates // 8)
        
        for i, trade_date in enumerate(backtest_dates):
            date_str = str(trade_date)
            
            # 1. 更新环境
            env = self.env_classifier.update(trade_date)
            self.env_history.append(env)
            env_mult = self.config.env_multipliers.get(env, 1.0)
            
            # 2. 更新信号检测器
            self.signal_detector.update(trade_date, date_str=date_str)
            
            # 3. 处理现有持仓
            self._manage_positions(trade_date, date_str, daily_data)
            
            # 4. 寻找新信号
            new_signals = self.signal_detector.scan(
                list(daily_data.get(date_str, {}).keys()),
                trade_date, date_str
            )
            
            # 5. 执行新信号
            for signal in new_signals:
                if self._check_can_enter(signal, trade_date, date_str):
                    self._enter_position(signal, trade_date, date_str, env_mult)
            
            # 6. 记录净值
            nav = self._calc_nav(trade_date, daily_data)
            dd = self._calc_drawdown(nav)
            self.equity_curve.append({
                "date": str(trade_date),
                "nav": round(nav, 2),
                "dd": round(dd, 2),
            })
            
            # 进度报告
            if (i + 1) % report_interval == 0:
                pos_count = len(self.positions)
                logger.info(
                    f"   [{trade_date}]({i+1}/{total_dates}) "
                    f"NAV={nav:,.0f} pos={pos_count} t={len(self.trade_log)} "
                    f"env={env}"
                )
        
        # 最终结果
        final_nav = self.equity_curve[-1]["nav"]
        total_return = (final_nav / self.config.initial_capital - 1) * 100
        logger.info(f"\n📊 {self.config.name} 全量结果")
        logger.info(f"   总收益率: {total_return:.2f}%")
    
    def _manage_positions(self, trade_date: date, date_str: str, daily_data: dict):
        """管理现有持仓：止损/止盈/评分检查"""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_price = self._get_price(symbol, trade_date, daily_data)
            if current_price is None:
                continue
            
            entry_price = pos["entry_price"]
            
            # 止损检查
            if self.risk_manager.check_stop_loss(current_price, entry_price):
                self._close_position(symbol, current_price, trade_date, "stop_loss")
                continue
            
            # 止盈检查
            tp_action = self.risk_manager.check_take_profit(current_price, entry_price, pos)
            if tp_action:
                self._close_position(symbol, current_price, trade_date, tp_action)
                continue
            
            # 基本面评分减仓
            if self.config.fund_score_mode == "post_trade":
                self._apply_score_reduction(symbol, date_str)
    
    def _apply_score_reduction(self, symbol: str, date_str: str):
        """post-trade 基本面评分减仓"""
        score = self._get_fund_score(symbol, date_str)
        if score < self.config.fund_score_min:
            pos = self.positions[symbol]
            reduction = 1.0 - self.config.fund_score_pos_reduce
            pos["weight"] *= reduction
    
    def _check_can_enter(self, signal: dict, trade_date: date, date_str: str) -> bool:
        """检查是否可以入场"""
        # 仓位限制
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # 已有相同持仓
        if signal.get("symbol") in self.positions:
            return False
        
        # pre-trade 评分过滤
        if self.config.fund_score_mode == "pre_trade":
            score = self._get_fund_score(signal["symbol"], date_str)
            if score < self.config.fund_score_min:
                return False
        
        # 共振过滤
        if self.config.resonance_enabled:
            res = self._calc_resonance(signal)
            if res < self.config.min_resonance:
                return False
        
        return True
    
    def _enter_position(self, signal: dict, trade_date: date, date_str: str, env_mult: float):
        """开仓"""
        symbol = signal["symbol"]
        price = float(signal.get("price", signal.get("close", 0)))
        
        # 计算仓位
        position_value = self.portfolio_value * self.config.position_pct * env_mult
        weight = position_value / self.portfolio_value
        
        self.positions[symbol] = {
            "entry_date": str(trade_date),
            "entry_price": price,
            "weight": weight,
            "signal": signal.get("signal_type", "unknown"),
        }
        
        self.trade_log.append({
            "date": str(trade_date),
            "symbol": symbol,
            "action": "BUY",
            "price": price,
            "signal": signal.get("signal_type"),
            "fund_score": self._get_fund_score(symbol, date_str),
            "resonance": self._calc_resonance(signal),
            "env": self.env_history[-1] if self.env_history else "UNKNOWN",
        })
    
    def _close_position(self, symbol: str, price: float, trade_date: date, reason: str):
        """平仓"""
        pos = self.positions.pop(symbol, None)
        if pos:
            pnl = (price / pos["entry_price"] - 1) * 100
            self.trade_log.append({
                "date": str(trade_date),
                "symbol": symbol,
                "action": "SELL",
                "price": price,
                "pnl": pnl,
                "reason": reason,
            })
    
    def _get_fund_score(self, symbol: str, date_str: str) -> float:
        """获取基本面评分"""
        if not self.scores_cache:
            return self.config.fund_score_default
        
        # 策略根据格式转换（scores 用 Tushare 格式: 000001.SZ）
        # 回测内部用 Qlib 格式: sh.600519, 需要转换
        ts_code = self.data_loader.to_ts_code(symbol)
        scores_data = self.scores_cache.get(ts_code, {})
        if not scores_data:
            return self.config.fund_score_default
        
        # 找最近的季度
        dates = sorted(scores_data.keys())
        best = None
        for d in dates:
            if d <= date_str:
                best = d
            else:
                break
        return scores_data.get(best, self.config.fund_score_default)
    
    def _calc_resonance(self, signal: dict) -> float:
        """计算共振分数"""
        if not self.config.resonance_enabled:
            return 1.0
        tech = signal.get("tech_factor", 0.5)
        fund = self._get_fund_score(signal.get("symbol", ""), "") / 100.0
        return min(1.0, tech * self.config.tech_weight + fund * self.config.fund_weight)
    
    def _get_price(self, symbol: str, trade_date: date, daily_data: dict) -> Optional[float]:
        """获取指定日期的价格"""
        day_data = daily_data.get(str(trade_date), {})
        kbar = day_data.get(symbol)
        if kbar:
            return kbar.get("close")
        return None
    
    def _calc_nav(self, trade_date: date, daily_data: dict) -> float:
        """计算当前净值"""
        nav = self.config.initial_capital - sum(
            p["weight"] * self.config.initial_capital for p in self.positions.values()
        )
        for symbol, pos in self.positions.items():
            price = self._get_price(symbol, trade_date, daily_data)
            if price and pos["entry_price"] > 0:
                pos_value = self.config.initial_capital * pos["weight"]
                nav += pos_value * (price / pos["entry_price"])
            else:
                nav += self.config.initial_capital * pos["weight"]
        return nav
    
    def _calc_drawdown(self, current_nav: float) -> float:
        """计算回撤"""
        peak = self.reporter.peak_nav
        if current_nav > peak:
            self.reporter.peak_nav = current_nav
            return 0.0
        return (current_nav / peak - 1) * 100


# 便捷入口
def run_backtest(config: BacktestConfig) -> BacktestReporter:
    """便捷入口：运行回测"""
    engine = BacktestEngine(config)
    return engine.run()
