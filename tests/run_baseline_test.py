#!/usr/bin/env python3
"""
P0.4: 轻量基线一致性验证
=========================
简化版：500只股票 + 2021年上半年 + 10万资金
验证与最初基线(-1.74%)一致。

运行方式：
  PYTHONUNBUFFERED=1 python3 tests/run_baseline_test.py
"""

import sys, os, time
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime
import numpy as np

_WORKSPACE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _WORKSPACE)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline_test")

from run_chanfund_v20_backtest import (
    load_daily_csv_data, load_financial_data, get_fund_score
)

# 轻量参数
N_STOCKS = 500           # 仅500只股票
INITIAL_CAPITAL = 100_000
MAX_POSITIONS = 50       # 分散持仓
POSITION_SIZE = 0.01     # 每只1%
FIXED_STOP_LOSS = 0.05   # 固定5%止损
START = "2021-01-01"
END = "2021-06-30"

logger.info("=" * 60)
logger.info("P0.4: 轻量基线一致性验证")
logger.info(f"股票: {N_STOCKS}只 | 区间: {START} ~ {END} | 资金: {INITIAL_CAPITAL:,}")
logger.info(f"目标总收益率: -1.74%")
logger.info("=" * 60)

t0 = time.time()

# 加载数据（仅500只）
data = load_daily_csv_data(str(Path(_WORKSPACE)/"quant"/"data"/"daily_csv"), max_files=N_STOCKS)
all_dates = sorted(data.keys())
warmup_dates = [d for d in all_dates if d < START]
bt_dates = [d for d in all_dates if START <= d <= END]
all_stocks = sorted(set().union(*(d.keys() for d in data.values())))

logger.info(f"数据: {len(all_stocks)}只股票, 预热{len(warmup_dates)}天, 回测{len(bt_dates)}天")

# 初始化信号引擎
from chanfund_fusion.tech.signal_engine import TechSignalEngine
from chanfund_fusion.config_schema import load_config
cfg = load_config(str(Path(_WORKSPACE)/"chanfund_fusion"/"config.yaml"))
tech_engine = TechSignalEngine(cfg.get("tech", {}))
logger.info("信号引擎初始化完成")

# 预热
logger.info("预热中...")
for i, ds in enumerate(warmup_dates):
    day_data = data.get(ds, {})
    for stock, kbar in day_data.items():
        try: tech_engine.update(stock, kbar, emit_signals=False)
        except: pass
    if (i+1) % 300 == 0:
        logger.info(f"  预热 {i+1}/{len(warmup_dates)}")

logger.info(f"预热完成: {len(tech_engine.chan_cache)}只股票有结构")

# 回测
logger.info("回测中...")
positions = {}
cash = INITIAL_CAPITAL
trade_log = []
signal_stats = defaultdict(int)
exit_stats = defaultdict(int)
total_signals = 0
peak_equity = cash

for di, ds in enumerate(bt_dates):
    trade_date = date.fromisoformat(ds)
    day_data = data.get(ds, {})
    
    # 更新引擎
    for stock, kbar in day_data.items():
        try: tech_engine.update(stock, kbar, emit_signals=True)
        except: pass
    
    # 获取信号
    signals = tech_engine.get_confirmed_signals(trade_date)
    total_signals += len(signals)
    
    # 买入（基线：分散+5%固定止损）
    for sig in signals:
        if len(positions) >= MAX_POSITIONS: break
        if sig.symbol in positions or sig.symbol not in day_data: continue
        price = day_data[sig.symbol].get("close", 0)
        if price <= 0: continue
        cost = cash * POSITION_SIZE
        if cost > cash: continue
        shares = cost / price
        stop = price * (1 - FIXED_STOP_LOSS)
        positions[sig.symbol] = {"entry_price": price, "shares": shares,
                                  "entry_date": ds, "stop_price": stop,
                                  "peak_price": price, "signal": sig.signal_subtype}
        cash -= cost
        signal_stats[sig.signal_subtype] += 1
        trade_log.append({"date": ds, "symbol": sig.symbol, "action": "BUY",
                          "price": round(price,3), "signal": sig.signal_subtype})
    
    # 止损
    to_close = []
    for stock, pos in list(positions.items()):
        price = day_data.get(stock, {}).get("close", pos["entry_price"])
        if price <= pos["stop_price"]:
            to_close.append(stock)
    
    for stock in to_close:
        pos = positions.pop(stock)
        price = day_data.get(stock, {}).get("close", pos["entry_price"])
        cash += pos["shares"] * price
        pnl = (price/pos["entry_price"]-1)*100
        exit_stats["stop_loss"] += 1
        trade_log.append({"date": ds, "symbol": stock, "action": "SELL",
                          "price": round(price,3), "reason": "stop_loss",
                          "pnl_pct": round(pnl,2)})
    
    peak_equity = max(peak_equity, cash + sum(
        p["shares"] * day_data.get(s,{}).get("close",p["entry_price"])
        for s,p in positions.items() if s in day_data))
    
    if (di+1) % 30 == 0:
        nav = cash + sum(p["shares"]*(day_data.get(s,{}).get("close",p["entry_price"]))
                         for s,p in positions.items() if s in day_data)
        logger.info(f"  {ds} ({di+1}/{len(bt_dates)}) NAV={nav:.0f} pos={len(positions)}")

# 强制平仓
for stock, pos in list(positions.items()):
    cash += pos["shares"] * pos["entry_price"]
    trade_log.append({"date": bt_dates[-1], "symbol": stock, "action": "SELL",
                      "price": round(pos["entry_price"],3), "reason": "force_close",
                      "pnl_pct": 0.0})

final_nav = cash
total_return = (final_nav/INITIAL_CAPITAL-1)*100
n_sells = len([t for t in trade_log if t["action"]=="SELL"])

elapsed = time.time() - t0
logger.info(f"\n{'='*50}")
logger.info("P0.4 基线验证结果")
logger.info(f"{'='*50}")
logger.info(f"最终净值: {final_nav:,.0f}")
logger.info(f"总收益率: {total_return:.2f}%")
logger.info(f"总交易数: {len(trade_log)} (买入+卖出)")
logger.info(f"止损次数: {exit_stats['stop_loss']}")
logger.info(f"总信号数: {total_signals}")
logger.info(f"耗时: {elapsed:.0f}s")

target = -1.74
lower = target - 0.5
upper = target + 0.5
logger.info(f"\n目标总收益率: {target}%")
logger.info(f"允许范围: {lower}% ~ {upper}%")

if lower <= total_return <= upper:
    logger.info(f"✅ 基线验证通过! 收益率在允许范围内")
elif total_return > upper:
    logger.info(f"⚠️ 收益率高于预期 (+{total_return-target:.2f}%) — 可能是代码优化提升")
else:
    logger.info(f"⚠️ 收益率低于预期 ({total_return-target:.2f}%) — 需检查代码回归")

logger.info(f"基础信号分布: {dict(signal_stats)}")
