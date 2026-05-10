#!/usr/bin/env python3
"""
Task 4: v1.4 基线在 2024 年样本外的重置回测
============================================
目的：验证 v1.4 那该死的 3.44 夏普在样本外是否站得住。

操作：使用 v1.4 完全相同的代码与参数，回测窗口 2024-01-01 至 2025-12-31，
股票池 4442 只，预热 3 年。

如果样本外套尔夏普腰斩甚至变负，则 v1.4 在实盘中已失效。
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

REPORT_DIR = Path("/home/quant/backtest report")
DATA_DIR = Path(_WORKSPACE) / "quant" / "data" / "daily_parquet"
LOG_PATH = _WORKSPACE + "/task4_v14_2024_oos.log"

logger = logging.getLogger("task4_v14_oos")

# ============================================================
# v1.4 参数（严格与原始 v1.4 一致）
# ============================================================
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5       # 最大持仓数
POSITION_SIZE = 0.20    # 单只最大仓位 20%
COMMISSION_RATE = 0.0003
FIXED_STOP_LOSS = 0.08  # 固定 -8% 止损
TP1_PCT = 0.10          # 第一止盈 +10%
TP1_REDUCE = 0.30       # 减 30%
TP2_PCT = 0.20          # 第二止盈 +20%
TP2_REDUCE = 0.50       # 减 50%
TP_TRAIL_RETRACE = 0.90 # 峰值回落 10% 清仓
TP_BREAKEVEN_LOCK = 0.05
VOL_MIN_RATIO = 0.8
COOLING_PERIOD_DAYS = 10
COOLING_STOP_COUNT = 2

# v1.4 基本面参数
FUND_SCORE_MIN = 50
FUND_DEFAULT_SCORE = 55.0
FUND_ROE_MIN = 0.05
FUND_GM_MIN = 0.15
FUND_REV_GROWTH_MIN = -0.20

# 信号类型基础分（v1.4 使用）
SIGNAL_BASE_SCORE_V14 = {
    "third_point_buy": 0.75, "hard_divergence": 0.90,
    "second_class_buy": 0.70, "soft_divergence": 0.55,
}

# 回测窗口：2024 样本外
WARMUP_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
    )


def load_data(max_stocks: int = 500) -> dict:
    all_pq = Path(DATA_DIR) / "_all.parquet"
    if not all_pq.exists():
        logger.error(f"❌ 无数据: {all_pq}")
        return {}
    t0 = time.time()
    PQC = ["date", "open", "high", "low", "close", "volume"]
    all_df = pd.read_parquet(all_pq, columns=["code"] + PQC)
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
    logger.info(f"✅ 数据: {len(data)}天, {time.time()-t0:.0f}s")
    return data


def calc_metrics(trade_log: list, equity_curve: list, initial_capital: float) -> dict:
    if not equity_curve:
        return {"final_equity": initial_capital, "total_return": 0, "ann_return": 0,
                "sharpe": 0, "max_drawdown": 0, "win_rate": 0, "avg_win": 0,
                "avg_loss": 0, "win_loss_ratio": 0, "n_trades": 0}
    total_return = (equity_curve[-1]["nav"] / initial_capital - 1)
    days = len(equity_curve)
    ann_ret = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    eq = np.array([e["nav"] for e in equity_curve])
    daily_ret = np.diff(eq) / eq[:-1]
    rf_daily = 0.03 / 252
    excess = daily_ret - rf_daily
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 1e-8 else 0
    rolling_max = np.maximum.accumulate(eq)
    dd = (eq - rolling_max) / rolling_max
    max_dd = np.min(dd)
    sells = [t for t in trade_log if t.get("action") == "SELL"]
    wins = [t for t in sells if t.get("pnl_pct", 0) > 0]
    losses = [t for t in sells if t.get("pnl_pct", 0) <= 0]
    n_trades = len(sells)
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else 0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    best = max([t.get("pnl_pct", 0) for t in sells]) if sells else 0
    worst = min([t.get("pnl_pct", 0) for t in sells]) if sells else 0
    return {
        "final_equity": round(equity_curve[-1]["nav"], 2),
        "total_return": round(total_return * 100, 2),
        "ann_return": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(wl_ratio, 2),
        "best_trade": round(best, 2),
        "worst_trade": round(worst, 2),
        "n_trades": n_trades,
    }


def main():
    setup_logging()
    _T_START = time.time()
    
    logger.info("=" * 60)
    logger.info("🔰 Task 4: v1.4 基线 — 2024 样本外重置回测")
    logger.info(f"   回测窗口: {TEST_START} ~ {TEST_END}")
    logger.info("=" * 60)
    
    # 加载数据（4442只与原始一致）
    data = load_data(max_stocks=4442)
    if not data:
        return
    
    all_dates = sorted(data.keys())
    warmup_dates = [d for d in all_dates if d <= WARMUP_END]
    backtest_dates = [d for d in all_dates if TEST_START <= d <= TEST_END]
    
    logger.info(f"   预热: {len(warmup_dates)}天, 回测: {len(backtest_dates)}天")
    
    # 初始化信号引擎
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path(_WORKSPACE) / "chanfund_fusion" / "config.yaml"))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    
    all_stocks = sorted(set().union(*(d.keys() for d in data.values())))[:500]
    logger.info(f"   股票池: {len(all_stocks)} 只")
    
    # ---- Warm-up ----
    logger.info("Phase 1: Warm-up")
    for i, ds in enumerate(warmup_dates):
        day_data = data.get(ds, {})
        for stock in all_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=False)
                except: pass
        if (i+1) % 100 == 0:
            logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
    
    logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)} 只股票")
    
    # ---- 回测（v1.4 风格：二元选股，无共振得分，用默认基本面） ----
    cash = INITIAL_CAPITAL
    positions: Dict[str, dict] = {}
    equity_curve = []
    cooling: Dict[str, date] = {}
    trade_log = []
    vol_cache: Dict[str, list] = {}
    signal_stats = defaultdict(int)
    
    for di, date_str in enumerate(backtest_dates):
        trade_date = date.fromisoformat(date_str)
        day_data = data.get(date_str, {})
        if not day_data: continue
        
        # 更新技术引擎
        for stock in all_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=True)
                except: pass
                v = day_data[stock].get("volume", 0)
                if v > 0:
                    vol_cache.setdefault(stock, []).append(v)
                    if len(vol_cache[stock]) > 50: vol_cache[stock] = vol_cache[stock][-50:]
        
        # 获取信号（v1.4: 只取高级别信号）
        signals = tech_engine.get_confirmed_signals(trade_date)
        
        for sig in signals:
            if len(positions) >= MAX_POSITIONS: break
            if sig.symbol in positions: continue
            if sig.symbol in cooling and trade_date <= cooling[sig.symbol]: continue
            if sig.symbol not in day_data: continue
            
            entry_price = day_data[sig.symbol].get("close", 0)
            if entry_price <= 0: continue
            
            # v1.4: 基本面过滤（默认55分，>=50）
            fund_score = FUND_DEFAULT_SCORE  # v1.4 使用默认分
            if fund_score < FUND_SCORE_MIN: continue
            
            # 成交量过滤
            vols = vol_cache.get(sig.symbol, [])
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                entry_vol = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and entry_vol < avg_vol * VOL_MIN_RATIO: continue
            
            # v1.4: 二元信号（只有较高分的信号才买入）
            base = SIGNAL_BASE_SCORE_V14.get(sig.signal_subtype, 0.5)
            if base < 0.70: continue  # 只买 hard_divergence 和 third_point_buy
            
            shares = int(cash * POSITION_SIZE / entry_price / 100) * 100
            if shares <= 0: continue
            cost = shares * entry_price
            if cost > cash or cost <= 0: continue
            
            stop_price = entry_price * (1 - FIXED_STOP_LOSS)
            positions[sig.symbol] = {
                "entry_price": entry_price, "shares": shares,
                "entry_date": date_str, "stop_price": stop_price,
                "peak_price": entry_price, "signal_type": sig.signal_subtype,
                "tp1_triggered": False, "tp2_triggered": False,
            }
            cash -= cost
            signal_stats[sig.signal_subtype] += 1
            trade_log.append({
                "date": date_str, "symbol": sig.symbol, "action": "BUY",
                "price": round(entry_price, 3), "signal": sig.signal_subtype,
            })
        
        # 止盈/止损（与v1.4一致）
        to_close, to_reduce = [], []
        for stock, pos in list(positions.items()):
            if stock in day_data and day_data[stock].get("close", 0) > 0:
                price = day_data[stock]["close"]
                pos["_last_price"] = price
            else:
                price = pos.get("_last_price", pos["entry_price"])
            pos["peak_price"] = max(pos["peak_price"], price)
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
            hold_days = (trade_date - date.fromisoformat(pos["entry_date"])).days
            
            if pnl_pct >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= TP_TRAIL_RETRACE:
                    to_close.append((stock, "trailing_stop", price)); continue
            if pnl_pct >= TP2_PCT and not pos["tp2_triggered"]:
                pos["tp2_triggered"] = True
                to_reduce.append((stock, "tp2", TP2_REDUCE, price)); continue
            if pnl_pct >= TP1_PCT and not pos["tp1_triggered"]:
                pos["tp1_triggered"] = True
                to_reduce.append((stock, "tp1", TP1_REDUCE, price)); continue
            if price <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price))
        
        for sym, reason, rp, p in to_reduce:
            rp_ = int(positions[sym]["shares"] * rp) if rp else 0
            if rp_ > 0 and positions[sym]["shares"] > rp_:
                positions[sym]["shares"] -= rp_
                cash += rp_ * p
                trade_log.append({"date": date_str, "symbol": sym, "action": "SELL",
                    "price": round(p, 3), "reason": reason,
                    "pnl_pct": round((p/positions[sym]["entry_price"]-1)*100, 2)})
        
        for sym, reason, p in to_close:
            pos = positions.pop(sym)
            cash += pos["shares"] * p
            trade_log.append({"date": date_str, "symbol": sym, "action": "SELL",
                "price": round(p, 3), "reason": reason,
                "pnl_pct": round((p/pos["entry_price"]-1)*100, 2),
                "hold_days": (trade_date - date.fromisoformat(pos["entry_date"])).days})
            if reason == "stop_loss":
                if sum(1 for t in reversed(trade_log) if t.get("symbol") == sym and t.get("reason") == "stop_loss") >= COOLING_STOP_COUNT:
                    cooling[sym] = trade_date + timedelta(days=COOLING_PERIOD_DAYS)
        
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
        equity_curve.append({"date": date_str, "nav": round(equity, 2), "dd": round(dd, 2)})
    
    # 强制平仓
    for stock, pos in list(positions.items()):
        cp = pos.get("_last_price", pos["entry_price"])
        cash += pos["shares"] * cp
        trade_log.append({"date": backtest_dates[-1], "symbol": stock, "action": "SELL",
            "price": round(cp, 3), "reason": "force_close",
            "pnl_pct": round((cp/pos["entry_price"]-1)*100, 2)})
    positions.clear()
    
    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    
    # 输出
    logger.info(f"\n{'='*55}")
    logger.info("📊 v1.4 — 2024 样本外结果")
    logger.info(f"{'='*55}")
    logger.info(f"  总收益率: {metrics['total_return']:.2f}%")
    logger.info(f"  年化收益率: {metrics['ann_return']:.2f}%")
    logger.info(f"  夏普比率: {metrics['sharpe']:.2f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {metrics['win_rate']:.1f}%")
    logger.info(f"  交易次数: {metrics['n_trades']}")
    
    # vs 原始 v1.4 基线
    orig_sharpe = 3.44
    orig_return = 29.02
    orig_dd = -13.71
    
    logger.info(f"\n  vs 原始 v1.4 基线 (2021-2026)")
    logger.info(f"  原始夏普: {orig_sharpe:.2f} → 样本外夏普: {metrics['sharpe']:.2f} "
                f"(变化: {metrics['sharpe'] - orig_sharpe:+.2f})")
    logger.info(f"  原始收益: {orig_return:.2f}% → 样本外: {metrics['total_return']:.2f}% "
                f"(变化: {metrics['total_return'] - orig_return:+.2f}%)")
    
    if abs(metrics['sharpe']) < orig_sharpe * 0.5:
        logger.warning(f"⚠️ 夏普腰斩! v1.4 样本外失效, 不再作为基准")
    else:
        logger.info(f"✅ v1.4 样本外夏普保持良好")
    
    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(trade_log).to_csv(REPORT_DIR / f"task4_v14_oos_{ts}_trades.csv", index=False)
    pd.DataFrame(equity_curve).to_csv(REPORT_DIR / f"task4_v14_oos_{ts}_equity.csv", index=False)
    
    summary = {"metrics": metrics, "vs_original": {"sharpe_change": metrics['sharpe'] - orig_sharpe,
                "return_change": metrics['total_return'] - orig_return},
                "n_stocks": len(all_stocks), "data_source": "daily_parquet (同v2.0)"}
    with open(REPORT_DIR / f"task4_v14_oos_{ts}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"⏱️ 耗时: {(time.time()-_T_START)/60:.1f}min")


if __name__ == "__main__":
    main()
