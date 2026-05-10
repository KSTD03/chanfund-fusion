#!/usr/bin/env python3
"""
Task 1: 随机标签测试 — 验证策略信号是否有统计显著性
=====================================================
方法：将 v2.0 回测的信号序列随机打乱，保持风控不变，
重新跑回测。如果随机标签的绩效与原始无显著差异，
说明策略信号本身没有信息量。

判断标准：如果随机标签夏普 > -0.04 或年化收益 > 0.5%，
测试不合格，说明回测框架存在结构性偏误。
"""

import os, sys, time, json, copy
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

_WORKSPACE = str(Path(__file__).resolve().parent.parent)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from run_chanfund_v20_backtest import (
    run_backtest, calc_metrics, load_daily_csv_data, load_financial_data, load_index_data,
    setup_logging, TECH_WEIGHT, FUND_WEIGHT,
)
from run_chanfund_v20_backtest import (
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE, FIXED_STOP_LOSS,
    TP1_PCT, TP1_REDUCE, TP2_PCT, TP2_REDUCE, TP_TRAIL_RETRACE, TP_BREAKEVEN_LOCK,
    COOLING_PERIOD_DAYS, COOLING_STOP_COUNT, VOL_MIN_RATIO,
    FUND_SCORE_MIN, FUND_DEFAULT_SCORE,
    ENV_MULTIPLIER,
)

REPORT_DIR = Path("/home/quant/backtest report")
LOG_PATH = _WORKSPACE + "/task1_random_label.log"
DATA_DIR = Path(_WORKSPACE) / "quant" / "data" / "daily_parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
)
logger = logging.getLogger("task1_random")

# ---- 配置 ----
RANDOM_SEED = 42
TEST_START = "2020-01-02"
TEST_END = "2026-04-29"


def load_parquet_data(max_stocks: int = 4441) -> dict:
    """加载日线数据"""
    all_pq = Path(DATA_DIR) / "_all.parquet"
    if not all_pq.exists():
        logger.error(f"❌ Parquet不存在: {all_pq}")
        return {}
    
    logger.info(f"📂 加载Parquet: {all_pq}")
    t0 = time.time()
    PQC = ["date", "open", "high", "low", "close", "volume"]
    all_df = pd.read_parquet(all_pq, columns=["code"] + PQC)
    if max_stocks > 0:
        codes = all_df["code"].unique()[:max_stocks]
        all_df = all_df[all_df["code"].isin(codes)]
    
    data: dict = {}
    for code, grp in all_df.groupby("code"):
        c = code
        if not (c.startswith("sh.") or c.startswith("sz.") or c.startswith("bj.")):
            c = f"sh.{c}" if c[0] == '6' else f"sz.{c}"
        for _, row in grp.iterrows():
            d_str = str(row["date"]).strip()[:10]
            if len(d_str) < 8: continue
            close = float(row["close"])
            if close <= 0: continue
            data.setdefault(d_str, {})[c] = {
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": close,
                "volume": float(row["volume"]), "amount": 0.0,
            }
    del all_df
    logger.info(f"✅ Parquet: {len(data)}天, {time.time()-t0:.0f}s")
    return data


def run_shuffled_backtest(data: dict, fin_cache: dict, index_data: dict,
                          start: str = TEST_START, end: str = TEST_END,
                          initial_capital: float = INITIAL_CAPITAL,
                          max_stocks: int = 4441) -> dict:
    """
    运行随机标签回测：
    1. 先用原始信号引擎跑一遍，记录所有买入信号的 (日期, 股票, 信号类型) 序列
    2. 将信号序列的时间顺序打乱（保持同一只股票的相对顺序内部随机化）
    3. 将打乱后的信号重新注入回测引擎
    
    简化方案：不从底层修改技术引擎，而是：
    - 正常跑回测
    - 在 get_confirmed_signals 收到信号后，判断是否应该"通过"该信号
    - 对每个信号，根据随机打乱后的信号表判断是否允许买入
    """
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    from chanfund_fusion.tech.chan_objects import Signal, SignalStatus
    
    cfg = load_config(str(Path(_WORKSPACE) / "chanfund_fusion" / "config.yaml"))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    
    all_dates = sorted(data.keys())
    warmup_dates = [d for d in all_dates if d < start]
    backtest_dates = [d for d in all_dates if start <= d <= end]
    
    all_stocks = sorted(set().union(*(d.keys() for d in data.values())))
    if max_stocks > 0:
        all_stocks = all_stocks[:max_stocks]
    
    logger.info(f"📊 股票池: {len(all_stocks)} 只, 回测 {len(backtest_dates)} 天")
    
    # Phase 1: Warm-up
    logger.info("Phase 1: Warm-up")
    for i, ds in enumerate(warmup_dates):
        day_data = data.get(ds, {})
        for stock in all_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=False)
                except: pass
        if (i+1) % 100 == 0:
            logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
    
    # Phase 2: 先跑一次回测收集所有信号
    logger.info(f"Phase 2a: 收集原始信号序列...")
    all_signals: List[dict] = []
    
    for ds in backtest_dates:
        trade_date = date.fromisoformat(ds)
        day_data = data.get(ds, {})
        if not day_data: continue
        
        for stock in all_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=True)
                except: pass
        
        # 收集今日信号
        signals = tech_engine.get_confirmed_signals(trade_date)
        for sig in signals:
            all_signals.append({
                "date": ds,
                "symbol": sig.symbol,
                "signal_subtype": sig.signal_subtype,
                "price_confirmed": sig.price_confirmed,
            })
    
    logger.info(f"📊 原始信号总数: {len(all_signals)}")
    
    # 将信号按股票分组，打乱每只股票信号的出现日期
    np.random.seed(RANDOM_SEED)
    
    # 方法：按股票分组，将每只股票的信号日期列表打乱
    signals_by_stock: Dict[str, List[dict]] = defaultdict(list)
    for sig in all_signals:
        signals_by_stock[sig["symbol"]].append(sig)
    
    shuffled_signals_by_date: Dict[str, List[dict]] = defaultdict(list)
    for symbol, sigs in signals_by_stock.items():
        if len(sigs) < 2:
            # 只有0或1个信号，无法打乱
            for s in sigs:
                shuffled_signals_by_date[s["date"]].append(s)
            continue
        
        # 提取所有原始日期
        orig_dates = [s["date"] for s in sigs]
        shuffled_dates = orig_dates.copy()
        np.random.shuffle(shuffled_dates)
        
        # 将每个信号分配到打乱后的日期
        for s, new_date in zip(sigs, shuffled_dates):
            new_s = dict(s)
            new_s["date"] = new_date  # 信号出现在新日期
            new_s["original_date"] = s["date"]
            shuffled_signals_by_date[new_date].append(new_s)
    
    logger.info(f"✅ 信号打乱完成。打乱后有效信号(按日期): {sum(len(v) for v in shuffled_signals_by_date.values())}")
    
    # Phase 2b: 重新初始化引擎，重新跑回测（但用打乱后的信号表控制买入）
    logger.info(f"Phase 2b: 打乱信号回测...")
    
    # 重新初始化
    tech_engine2 = TechSignalEngine(cfg.get("tech", {}))
    for i, ds in enumerate(warmup_dates):
        day_data = data.get(ds, {})
        for stock in all_stocks:
            if stock in day_data:
                try: tech_engine2.update(stock, day_data[stock], emit_signals=False)
                except: pass
    
    # ---- 市场环境 ----
    index_data = load_index_data(str(Path(_WORKSPACE) / "quant" / "data" / "daily_csv"))
    
    class SimpleEnv:
        def classify(self, ds): return "OSCILLATE"
    env_cls = SimpleEnv()
    
    # ---- 状态 ----
    cash = initial_capital
    positions: Dict[str, dict] = {}
    equity_curve = []
    cooling: Dict[str, date] = {}
    trade_log = []
    vol_cache: Dict[str, list] = {}
    signal_stats = defaultdict(int)
    
    for di, ds in enumerate(backtest_dates):
        trade_date = date.fromisoformat(ds)
        day_data = data.get(ds, {})
        if not day_data: continue
        
        # 更新技术引擎
        for stock in all_stocks:
            if stock in day_data:
                try: tech_engine2.update(stock, day_data[stock], emit_signals=True)
                except: pass
                v = day_data[stock].get("volume", 0)
                if v > 0:
                    vol_cache.setdefault(stock, []).append(v)
                    if len(vol_cache[stock]) > 50:
                        vol_cache[stock] = vol_cache[stock][-50:]
        
        # 获取今日已被批准买入的信号（来自打乱后的信号表）
        approved_signals = shuffled_signals_by_date.get(ds, [])
        
        for sig_info in approved_signals:
            symbol = sig_info["symbol"]
            if len(positions) >= MAX_POSITIONS: break
            if symbol in positions: continue
            if symbol in cooling and trade_date <= cooling[symbol]: continue
            if symbol not in day_data: continue
            
            entry_price = day_data[symbol].get("close", 0)
            if entry_price <= 0: continue
            
            # 基础过滤（与v2.0一致）
            fund_score = 55.0  # 默认分
            if fund_score < FUND_SCORE_MIN: continue
            
            vols = vol_cache.get(symbol, [])
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                entry_vol = day_data[symbol].get("volume", 0)
                if avg_vol > 0 and entry_vol < avg_vol * VOL_MIN_RATIO: continue
            
            # 简化的仓位计算（用固定仓位）
            shares = int(cash * POSITION_SIZE / entry_price / 100) * 100
            if shares <= 0: continue
            cost = shares * entry_price
            if cost > cash or cost <= 0: continue
            
            stop_price = entry_price * (1 - FIXED_STOP_LOSS)
            positions[symbol] = {
                "entry_price": entry_price, "shares": shares,
                "entry_date": ds, "stop_price": stop_price,
                "peak_price": entry_price, "tp1_triggered": False, "tp2_triggered": False,
                "signal_type": sig_info["signal_subtype"],
            }
            cash -= cost
            signal_stats[sig_info["signal_subtype"]] += 1
            trade_log.append({
                "date": ds, "symbol": symbol, "action": "BUY",
                "price": round(entry_price, 3), "signal": sig_info["signal_subtype"],
            })
        
        # 止盈/止损（与v2.0一致）
        to_close, to_reduce = [], []
        for symbol, pos in list(positions.items()):
            if symbol in day_data and day_data[symbol].get("close", 0) > 0:
                price = day_data[symbol]["close"]
                pos["_last_price"] = price
            else:
                price = pos.get("_last_price", pos["entry_price"])
            
            pos["peak_price"] = max(pos["peak_price"], price)
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
            hold_days = (trade_date - date.fromisoformat(pos["entry_date"])).days
            
            # 保本移损
            if pnl_pct >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            # 长持仓
            if hold_days >= 60 and abs(pnl_pct) < 0.05:
                pos["stop_price"] = min(pos["stop_price"], pos["entry_price"] * 0.85)
            # 峰值回落
            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= TP_TRAIL_RETRACE:
                    to_close.append((symbol, "trailing_stop", price)); continue
            # TP2
            if pnl_pct >= TP2_PCT and not pos["tp2_triggered"]:
                pos["tp2_triggered"] = True
                to_reduce.append((symbol, "tp2", TP2_REDUCE, price)); continue
            # TP1
            if pnl_pct >= TP1_PCT and not pos["tp1_triggered"]:
                pos["tp1_triggered"] = True
                to_reduce.append((symbol, "tp1", TP1_REDUCE, price)); continue
            # 止损
            if price <= pos["stop_price"]:
                to_close.append((symbol, "stop_loss", price))
        
        # 执行
        for symbol, reason, reduce_pct, price in to_reduce:
            pos = positions[symbol]
            r_shares = int(pos["shares"] * reduce_pct)
            if r_shares > 0 and pos["shares"] > r_shares:
                pos["shares"] -= r_shares
                cash += r_shares * price
                pnl = (price / pos["entry_price"] - 1) * 100
                trade_log.append({
                    "date": ds, "symbol": symbol, "action": "SELL",
                    "price": round(price, 3), "reason": reason, "pnl_pct": round(pnl, 2),
                })
        
        for symbol, reason, price in to_close:
            pos = positions.pop(symbol)
            cash += pos["shares"] * price
            pnl = (price / pos["entry_price"] - 1) * 100
            trade_log.append({
                "date": ds, "symbol": symbol, "action": "SELL",
                "price": round(price, 3), "reason": reason, "pnl_pct": round(pnl, 2),
            })
            if reason == "stop_loss":
                consec = sum(1 for t in reversed(trade_log) if t.get("symbol") == symbol and t.get("reason") == "stop_loss")
                if consec >= COOLING_STOP_COUNT:
                    cooling[symbol] = trade_date + timedelta(days=COOLING_PERIOD_DAYS)
        
        # 净值
        pos_value = 0.0
        for s, p in list(positions.items()):
            if s in day_data and day_data[s].get("close", 0) > 0:
                px = day_data[s]["close"]
                p["_last_price"] = px
            else:
                px = p.get("_last_price", p["entry_price"])
            pos_value += p["shares"] * px
        equity = cash + pos_value
        peak_equity = max(equity_curve[-1]["nav"] if equity_curve else cash, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        equity_curve.append({"date": ds, "nav": round(equity, 2), "dd": round(dd, 2)})
        
        if (di+1) % 200 == 0:
            logger.info(f"  Shuffled {ds} ({di+1}/{len(backtest_dates)}) NAV={equity:,.0f} trades={len(trade_log)}")
    
    # 强制平仓
    for symbol, pos in list(positions.items()):
        cp = pos.get("_last_price", pos["entry_price"])
        cash += pos["shares"] * cp
        pnl = (cp / pos["entry_price"] - 1) * 100
        trade_log.append({
            "date": backtest_dates[-1], "symbol": symbol, "action": "SELL",
            "price": round(cp, 3), "reason": "force_close", "pnl_pct": round(pnl, 2),
        })
    positions.clear()
    
    metrics = calc_metrics(trade_log, equity_curve, initial_capital)
    metrics["n_signals"] = len(all_signals)
    
    return {
        "metrics": metrics,
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "signal_stats": dict(signal_stats),
    }


def main():
    logger.info("=" * 60)
    logger.info("🔰 Task 1: 随机标签测试")
    logger.info("=" * 60)
    
    # 使用已有的 Parquet 数据
    DATA_DIR_PARQUET = Path(_WORKSPACE) / "quant" / "data" / "daily_parquet"
    all_pq = DATA_DIR_PARQUET / "_all.parquet"
    
    if not all_pq.exists():
        logger.error(f"❌ 无Parquet数据: {all_pq}")
        return
    
    # 这是一个计算密集型任务，先跑500只股票（平衡速度和代表性）
    n_stocks = 500
    data = load_parquet_data(max_stocks=n_stocks)
    if not data:
        return
    
    # 先跑原始 v2.0 获取基线（500只）
    logger.info("⚙️ 步骤1: 运行原始 v2.0 回测 (500只)...")
    fin_cache = load_financial_data(str(Path(_WORKSPACE) / "quant" / "data" / "financial"))
    index_data = load_index_data(str(Path(_WORKSPACE) / "quant" / "data" / "daily_csv"))
    
    time_start = time.time()
    
    # 使用 run_backtest 但让它使用我们加载的 parquet 数据
    # 需要转换 run_backtest 的 data 格式
    original_result = run_backtest(data, fin_cache, index_data, TEST_START, TEST_END,
                                   INITIAL_CAPITAL, "test", n_stocks)
    orig_m = original_result["metrics"]
    logger.info(f"✅ 原始回测完成: 收益 {orig_m['total_return']:.2f}% | "
                f"夏普 {orig_m['sharpe']:.2f} | 回撤 {orig_m['max_drawdown']:.2f}%")
    
    # 跑随机标签
    logger.info("⚙️ 步骤2: 运行随机标签回测...")
    shuffled_result = run_shuffled_backtest(data, fin_cache, index_data,
                                            TEST_START, TEST_END, INITIAL_CAPITAL, n_stocks)
    shuf_m = shuffled_result["metrics"]
    
    elapsed = time.time() - time_start
    
    # 输出结果
    logger.info(f"\n{'='*55}")
    logger.info("📊 Task 1 结果")
    logger.info(f"{'='*55}")
    logger.info(f"  {'指标':<20} {'原始 v2.0':<15} {'随机标签':<15}")
    logger.info(f"  {'-'*52}")
    for key in ["total_return", "ann_return", "sharpe", "max_drawdown", "win_rate", "n_trades"]:
        v1 = round(orig_m.get(key, 0), 2)
        v2 = round(shuf_m.get(key, 0), 2)
        label = {"total_return": "总收益率(%)", "ann_return": "年化收益率(%)",
                 "sharpe": "夏普比率", "max_drawdown": "最大回撤(%)",
                 "win_rate": "胜率(%)", "n_trades": "交易次数"}.get(key, key)
        logger.info(f"  {label:<20} {v1:<15} {v2:<15}")
    
    shuf_sharpe = shuf_m.get("sharpe", -999)
    shuf_ann = shuf_m.get("ann_return", -999)
    threshold_sharpe = -0.04  # -0.16 * 25%
    threshold_ann = 0.5
    
    logger.info(f"\n{'='*55}")
    logger.info("🧪 判断")
    logger.info(f"  随机标签夏普: {shuf_sharpe:.4f} (阈值: > {threshold_sharpe} 不合格)")
    logger.info(f"  随机年化收益: {shuf_ann:.4f}% (阈值: > {threshold_ann}% 不合格)")
    
    if shuf_sharpe > threshold_sharpe or shuf_ann > threshold_ann:
        logger.warning(f"❌ 测试不合格! 随机标签表现异常, 回测框架存在偏误!")
        logger.warning(f"   原因: {'夏普超过阈值' if shuf_sharpe > threshold_sharpe else ''} "
                       f"{'年化超过阈值' if shuf_ann > threshold_ann else ''}")
        logger.warning(f"   建议: 停止所有迭代, 先修复回测框架")
    else:
        logger.info(f"✅ 测试合格! 随机标签绩效低于阈值, 信号具有统计显著性")
    
    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "original": {k: round(orig_m.get(k, 0), 4) if isinstance(orig_m.get(k, 0), float) else orig_m.get(k, 0) for k in orig_m},
        "shuffled": {k: round(shuf_m.get(k, 0), 4) if isinstance(shuf_m.get(k, 0), float) else shuf_m.get(k, 0) for k in shuf_m},
        "test_passed": not (shuf_sharpe > threshold_sharpe or shuf_ann > threshold_ann),
        "n_stocks": n_stocks,
        "elapsed_s": round(elapsed),
    }
    sp = REPORT_DIR / f"task1_random_label_{ts}_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    
    # 保存净值曲线
    pd.DataFrame(shuffled_result["equity_curve"]).to_csv(
        REPORT_DIR / f"task1_random_label_{ts}_equity.csv", index=False)
    pd.DataFrame(original_result["equity_curve"]).to_csv(
        REPORT_DIR / f"task1_random_label_{ts}_orig_equity.csv", index=False)
    
    logger.info(f"✅ 摘要保存: {sp}")
    logger.info(f"⏱️ 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return summary


if __name__ == "__main__":
    main()
