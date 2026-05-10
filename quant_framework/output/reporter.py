"""
结果报告器

职责：
1. 收集回测结果
2. 计算性能指标（总收益、夏普、回撤、胜率等）
3. 输出 CSV/JSON/Markdown 报告
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from quant_framework.config.config import BacktestConfig

logger = logging.getLogger(__name__)


class BacktestReporter:
    """回测结果报告器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trade_log: List[dict] = []
        self.equity_curve: List[dict] = []
        self.env_history: List[str] = []
        self.peak_nav: float = config.initial_capital
        self.elapsed: float = 0
        
        # 计算结果缓存
        self._summary: Optional[dict] = None
    
    def collect_results(self, trade_log: List[dict], equity_curve: List[dict],
                        env_history: List[str], config: BacktestConfig,
                        elapsed: float):
        """收集回测结果"""
        self.trade_log = trade_log
        self.equity_curve = equity_curve
        self.env_history = env_history
        self.config = config
        self.elapsed = elapsed
    
    def save(self):
        """保存结果到输出目录"""
        output_dir = Path(self.config.output_dir) / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 交易日志
        if self.trade_log:
            df_trades = pd.DataFrame(self.trade_log)
            df_trades.to_csv(output_dir / "trades.csv", index=False)
        
        # 净值曲线
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity.to_csv(output_dir / "equity.csv", index=False)
        
        # 汇总报告
        summary = self.calc_summary()
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 月收益
        monthly = self.calc_monthly_returns()
        if monthly is not None:
            monthly.to_csv(output_dir / "monthly_returns.csv", index=False)
        
        logger.info(f"   💾 结果保存至: {output_dir}")
    
    def calc_summary(self) -> dict:
        """计算绩效汇总"""
        if self._summary:
            return self._summary
        
        nav_df = pd.DataFrame(self.equity_curve)
        trade_df = pd.DataFrame(self.trade_log)
        
        # 基础指标
        initial = self.config.initial_capital
        final = nav_df["nav"].iloc[-1] if not nav_df.empty else initial
        total_return = (final / initial - 1) * 100
        
        # 年化
        years = len(nav_df) / 245.0 if len(nav_df) > 0 else 1
        annual_return = ((final / initial) ** (1 / years) - 1) * 100
        
        # 夏普
        if len(nav_df) > 1:
            daily_rets = nav_df["nav"].pct_change().dropna()
            if daily_rets.std() > 0:
                sharpe = np.sqrt(245) * daily_rets.mean() / daily_rets.std()
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # 最大回撤
        max_dd = nav_df["dd"].min() if not nav_df.empty else 0.0
        
        # 交易统计
        buys = trade_df[trade_df["action"] == "BUY"] if not trade_df.empty else pd.DataFrame()
        sells = trade_df[trade_df["action"] == "SELL"] if not trade_df.empty else pd.DataFrame()
        
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        
        if not sells.empty and "pnl" in sells.columns:
            wins = sells[sells["pnl"] > 0]
            losses = sells[sells["pnl"] < 0]
            win_rate = len(wins) / len(sells) * 100 if len(sells) > 0 else 0
            
            if not wins.empty:
                avg_win = wins["pnl"].mean()
            if not losses.empty:
                avg_loss = losses["pnl"].mean()
            
            total_win = wins["pnl"].sum() if not wins.empty else 0
            total_loss = abs(losses["pnl"].sum()) if not losses.empty else 1
            profit_factor = total_win / total_loss if total_loss > 0 else 0
        
        # 环境分布
        env_counts = {}
        if self.env_history:
            env_series = pd.Series(self.env_history)
            env_counts = env_series.value_counts().to_dict()
        
        self._summary = {
            "config_name": self.config.name,
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(self.trade_log),
            "buy_signals": len(buys),
            "sell_signals": len(sells),
            "environment_distribution": {k: int(v) for k, v in env_counts.items()},
            "elapsed_seconds": round(self.elapsed, 1),
        }
        
        return self._summary
    
    def calc_monthly_returns(self) -> Optional[pd.DataFrame]:
        """计算月收益"""
        if not self.equity_curve:
            return None
        
        df = pd.DataFrame(self.equity_curve)
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.strftime("%Y-%m")
        
        month_groups = df.groupby("month")
        results = []
        for month, group in month_groups:
            first_nav = group["nav"].iloc[0]
            last_nav = group["nav"].iloc[-1]
            ret = (last_nav / first_nav - 1) * 100
            results.append({"month": month, "ret": round(ret, 2)})
        
        return pd.DataFrame(results)
    
    def print_summary(self):
        """打印汇总到日志"""
        s = self.calc_summary()
        logger.info(f"📊 {s['config_name']} 结果")
        logger.info(f"   总收益率: {s['total_return_pct']:.2f}%")
        logger.info(f"   年化收益率: {s['annual_return_pct']:.2f}%")
        logger.info(f"   夏普比率: {s['sharpe_ratio']:.4f}")
        logger.info(f"   最大回撤: {s['max_drawdown_pct']:.2f}%")
        logger.info(f"   胜率: {s['win_rate_pct']:.1f}%")
        logger.info(f"   盈利因子: {s['profit_factor']:.2f}")
        logger.info(f"   交易: {s['total_trades']}")
        logger.info(f"   环境: {s['environment_distribution']}")
        logger.info(f"   ⏱️ {s['elapsed_seconds']:.0f}s")
