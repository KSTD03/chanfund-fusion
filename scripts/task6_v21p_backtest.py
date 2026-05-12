#!/usr/bin/env python3
"""
Task 6: v2.1+MA 全量回测 — 极简包装版
========================================
在 task3_v21 基础上添加 MA20/60 环境分类器
"""

import sys, time, json, os, logging
from pathlib import Path
import numpy as np
from datetime import timedelta
import pandas as pd

sys.path.insert(0, '/home/quant/.openclaw/workspace')

# Logging
OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/output/v21p")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "v21p_run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("v21p")

# Import verified task3 components
from scripts.task3_v21_backtest import (
    load_data, load_scores_parquet, calc_metrics,
    get_fund_score_v21, resonance_score_v21, get_next_trade_date,
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE,
    FIXED_STOP_LOSS, TP1_PCT, TP1_REDUCE, TP2_PCT, TP2_REDUCE,
    COOLING_PERIOD_DAYS, COOLING_STOP_COUNT, VOL_MIN_RATIO,
    TP_BREAKEVEN_LOCK, TP_TRAIL_RETRACE, FUND_SCORE_MIN,
    TECH_WEIGHT, FUND_WEIGHT,
)

# ============================================================
# MA20/60 环境分类器
# ============================================================
class MAEnvClassifier:
    def __init__(self):
        self.prices = []
        self.current_env = "OSCILLATE"
    
    def update(self, price: float):
        if price <= 0:
            return
        self.prices.append(price)
        if len(self.prices) < 60:
            return
        if len(self.prices) >= 21:
            rets = [(self.prices[i] - self.prices[i-1]) / self.prices[i-1] for i in range(-20, 0)]
            vol = np.std(rets) * np.sqrt(252)
            if vol > 0.35:
                self.current_env = "OSCILLATE"
                return
        ma20 = sum(self.prices[-20:]) / 20
        ma60 = sum(self.prices[-60:]) / 60
        self.current_env = "BULL" if ma20 > ma60 else "BEAR"


ENV_MULT = {"BULL": 1.0, "OSCILLATE": 1.0, "BEAR": 0.6}


# ============================================================
# 包装回测函数
# ============================================================
def run_v21p(data: dict, scores_cache: dict, all_dates: list,
             start: str = "2020-01-01", end: str = "2026-12-31"):
    
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    
    cfg = load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion/config.yaml")))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    
    warmup_dates = [d for d in all_dates if d < start]
    bt_dates = [d for d in all_dates if start <= d <= end]
    
    all_stocks = sorted(set().union(*(d.keys() for d in data.values())))
    env_cls = MAEnvClassifier()
    
    logger.info(f"📊 {len(all_stocks)}只, 预热{len(warmup_dates)}d, 回测{len(bt_dates)}d")
    
    # 构建指数序列
    logger.info("🏗️ 构建指数序列...")
    index_prices = []
    for ds in warmup_dates + bt_dates:
        day = data.get(ds, {})
        if day:
            closes = [v["close"] for v in day.values() if v.get("close", 0) > 0]
            index_prices.append(np.mean(closes) if closes else (index_prices[-1] if index_prices else 3000))
        else:
            index_prices.append(index_prices[-1] if index_prices else 3000)
    
    # ---- Warm-up ----
    logger.info("Phase 1: Warm-up")
    for i, ds in enumerate(warmup_dates):
        env_cls.update(index_prices[i])
        day_data = data.get(ds, {})
        for stock in all_stocks:
            if stock in day_data:
                try:
                    tech_engine.update(stock, day_data[stock], emit_signals=False)
                except Exception:
                    pass
        if (i+1) % 500 == 0:
            logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
    
    logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)}只")
    
    # ---- 回测状态 ----
    cash = INITIAL_CAPITAL
    peak_equity = cash
    positions = {}
    equity_curve = []
    cooling = {}
    trade_log = []
    vol_cache = {}
    sig_stats = {}
    exit_stats = {}
    tp_stats = {}
    env_counts = {}
    score_stats = {"used": 0, "default": 0, "rejected": 0, "passed": 0}
    
    logger.info("Phase 2: Backtest v2.1+MA...")
    
    for di, ds in enumerate(bt_dates):
        trade_date = type(ds) if not isinstance(ds, str) else ds
        if not isinstance(ds, str):
            ds = str(ds)
        
        # Update env classifier
        idx = len(warmup_dates) + di
        if idx < len(index_prices):
            env_cls.update(index_prices[idx])
        env = env_cls.current_env
        env_counts[env] = env_counts.get(env, 0) + 1
        env_mult = ENV_MULT.get(env, 1.0)
        
        day_data = data.get(ds, {})
        if not day_data:
            continue
        
        # A. Update tech engine
        for stock in all_stocks:
            if stock in day_data:
                try:
                    tech_engine.update(stock, day_data[stock], emit_signals=True)
                except:
                    pass
                v = day_data[stock].get("volume", 0)
                if v > 0:
                    vol_cache.setdefault(stock, []).append(v)
                    if len(vol_cache[stock]) > 50:
                        vol_cache[stock] = vol_cache[stock][-50:]
        
        # B. Signals + pre-trade filter
        from datetime import date
        trade_date_dt = date.fromisoformat(ds) if isinstance(ds, str) else ds
        signals = tech_engine.get_confirmed_signals(trade_date_dt)
        
        for sig in signals:
            if len(positions) >= MAX_POSITIONS:
                break
            if sig.symbol in positions:
                continue
            if sig.symbol in cooling and (isinstance(ds, str) and trade_date_dt <= cooling[sig.symbol]):
                continue
            if sig.symbol not in day_data:
                continue
            
            nd = get_next_trade_date(ds, all_dates)
            if nd is None or sig.symbol not in data.get(nd, {}):
                continue
            entry_price = data[nd][sig.symbol]["open"]
            if entry_price <= 0:
                continue
            
            # Pre-trade score filter
            fund_score = get_fund_score_v21(sig.symbol, ds, scores_cache)
            if fund_score is None:
                fund_score = 50.0
                score_stats["default"] += 1
            else:
                score_stats["used"] += 1
            if fund_score < FUND_SCORE_MIN:
                score_stats["rejected"] += 1
                continue
            score_stats["passed"] += 1
            
            # Volume filter
            vols = vol_cache.get(sig.symbol, [])
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                entry_vol = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and entry_vol < avg_vol * VOL_MIN_RATIO:
                    continue
            
            # Factor + resonance
            vols_20 = vol_cache.get(sig.symbol, [])
            vol_ratio = 1.0
            if len(vols_20) >= 20:
                mv = sum(vols_20[-20:]) / 20
                vol_ratio = day_data[sig.symbol].get("volume", 0) / mv if mv > 0 else 1.0
            tech_factor = (0.6 if sig.signal_subtype == "third_point_buy" else
                          0.8 if sig.signal_subtype == "hard_divergence" else 0.5) * min(vol_ratio, 2.0)
            res = resonance_score_v21(tech_factor, fund_score)
            
            # Position size (resonance * env)
            pos_coeff = min(res, 1.0) * POSITION_SIZE / 0.20
            pos_coeff = min(1.0, max(0.0, pos_coeff)) * env_mult
            cost = cash * POSITION_SIZE * pos_coeff
            if cost > cash or cost <= 0:
                continue
            shares = int(cost / entry_price / 100) * 100
            if shares <= 0:
                continue
            cost = shares * entry_price
            if cost > cash or cost <= 0:
                continue
            
            stop_price = entry_price * (1 - FIXED_STOP_LOSS)
            positions[sig.symbol] = {
                "entry_price": entry_price, "shares": shares,
                "entry_date": ds, "buy_exec_date": nd,
                "stop_price": stop_price, "peak_price": entry_price,
                "signal_type": sig.signal_subtype,
                "tp1_triggered": False, "tp2_triggered": False,
                "fund_score": fund_score, "resonance": round(res, 3), "env": env,
            }
            cash -= cost
            sig_stats[sig.signal_subtype] = sig_stats.get(sig.signal_subtype, 0) + 1
            
            trade_log.append({
                "date": ds, "symbol": sig.symbol, "action": "BUY",
                "price": round(entry_price, 3), "signal": sig.signal_subtype,
                "fund_score": fund_score, "resonance": round(res, 3),
                "env": env,
            })
        
        # C. Stop/Take profit
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
            hold_days = (trade_date_dt - date.fromisoformat(pos["entry_date"])).days
            
            if pnl_pct >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            if hold_days >= 60 and abs(pnl_pct) < 0.05:
                pos["stop_price"] = min(pos["stop_price"], pos["entry_price"] * 0.85)
            
            low = day_data.get(stock, {}).get("low", price)
            if low <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price, "close"))
                continue
            
            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= TP_TRAIL_RETRACE:
                    nd2 = get_next_trade_date(ds, all_dates)
                    sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2, {}) else price
                    to_close.append((stock, "trailing_stop", sp, "next_open"))
                    tp_stats["trailing_stop"] = tp_stats.get("trailing_stop", 0) + 1
                    continue
            
            if pnl_pct >= TP2_PCT and not pos["tp2_triggered"]:
                pos["tp2_triggered"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2, {}) else price
                to_reduce.append((stock, "tp2", TP2_REDUCE, sp, "next_open"))
                tp_stats["tp2"] = tp_stats.get("tp2", 0) + 1
                continue
            
            if pnl_pct >= TP1_PCT and not pos["tp1_triggered"]:
                pos["tp1_triggered"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2, {}) else price
                to_reduce.append((stock, "tp1", TP1_REDUCE, sp, "next_open"))
                tp_stats["tp1"] = tp_stats.get("tp1", 0) + 1
                continue
        
        for sym, reason, rpct, sp, pt in to_reduce:
            pos = positions[sym]
            r_shares = int(pos["shares"] * rpct)
            if r_shares > 0 and pos["shares"] > r_shares:
                pos["shares"] -= r_shares
                cash += r_shares * sp
                exit_stats[f"reduce_{reason}"] = exit_stats.get(f"reduce_{reason}", 0) + 1
        
        for sym, reason, price, pt in to_close:
            pos = positions.pop(sym, None)
            if pos is None:
                continue
            pnl = (price - pos["entry_price"]) / pos["entry_price"] * 100
            cash += pos["shares"] * price
            exit_stats[reason] = exit_stats.get(reason, 0) + 1
            trade_log.append({
                "date": ds, "symbol": sym, "action": "SELL",
                "price": round(price, 3), "pnl_pct": round(pnl, 2),
                "reason": reason,
                "hold_days": hold_days,
                "fund_score": pos["fund_score"], "resonance": pos["resonance"], "env": pos.get("env", env),
            })
            if reason == "stop_loss":
                if pos.get("cooling_count", 0) >= COOLING_STOP_COUNT:
                    cooling[sym] = trade_date_dt + timedelta(days=COOLING_PERIOD_DAYS)
        
        # NAV
        equity = cash + sum(
            pos["shares"] * day_data.get(sym, {}).get("close", pos["entry_price"])
            for sym, pos in positions.items()
        )
        peak_equity = max(peak_equity, equity)
        dd = (equity / peak_equity - 1) * 100
        equity_curve.append({"date": ds, "nav": round(equity, 2), "dd": round(dd, 2)})
        
        if (di+1) % 200 == 0 or di == len(bt_dates)-1:
            logger.info(f"  [{ds}]({di+1}/{len(bt_dates)}) NAV={equity:,.0f} pos={len(positions)} t={len(trade_log)} env={env}")
    
    # ---- Metrics ----
    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    buys = [t for t in trade_log if t["action"] == "BUY"]
    sells = [t for t in trade_log if t["action"] == "SELL"]
    wins = [t for t in sells if t.get("pnl", 0) > 0]
    win_rate = len(wins) / max(len(sells), 1) * 100
    
    logger.info(f"\n{'='*55}")
    logger.info(f"📊 v2.1+MA 全量结果")
    logger.info(f"{'='*55}")
    logger.info(f"  总收益率: {metrics['total_return']:.2f}%")
    logger.info(f"  年化收益率: {metrics['ann_return']:.2f}%")
    logger.info(f"  夏普比率: {metrics['sharpe']:.4f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {win_rate:.1f}%")
    logger.info(f"  交易: {len(trade_log)}")
    logger.info(f"  信号: {sig_stats}")
    logger.info(f"  平仓: {exit_stats}")
    logger.info(f"  TP: {tp_stats}")
    logger.info(f"  环境: {env_counts}")
    logger.info(f"  scores: {score_stats}")
    
    return {
        "metrics": metrics, "trade_log": trade_log, "equity_curve": equity_curve,
    }


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    logger.info(f"{'='*55}")
    logger.info(f"🔰 Task 6: v2.1+MA 全量回测")
    logger.info(f"  pre-trade score<30过滤 + 共振加权 + MA20/60环境")
    logger.info(f"{'='*55}")
    
    # Load data (takes ~5min for 7.3M rows)
    data = load_data(max_stocks=0)
    all_dates = sorted(data.keys())
    logger.info(f"  日期: {all_dates[0]} ~ {all_dates[-1]}, {len(all_dates)}天")
    
    # Load scores
    scores_cache = load_scores_parquet()
    
    # Run
    result = run_v21p(data, scores_cache, all_dates,
                       start="2020-01-01", end="2026-12-31")
    
    # Save
    pd.DataFrame(result["trade_log"]).to_csv(OUTPUT_DIR / "v21p_trades.csv", index=False)
    pd.DataFrame(result["equity_curve"]).to_csv(OUTPUT_DIR / "v21p_equity.csv", index=False)
    
    with open(OUTPUT_DIR / "v21p_summary.json", "w") as f:
        json.dump(result["metrics"], f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n⏱️ 总耗时: {time.time()-t0:.0f}s")
    logger.info(f"💾 产出: {OUTPUT_DIR}")
    logger.info(f"\n📊 版本对比:")
    logger.info(f"  v2.3 (默认50, 无过滤):       23.70%  夏普0.11")
    logger.info(f"  v2.4 (真实scores, post减仓):  13.03%  夏普-0.12")
    logger.info(f"  v2.1+MA (pre<30+共振+MA):     {result['metrics']['total_return']:.2f}%  夏普{result['metrics']['sharpe']:.4f}")


if __name__ == "__main__":
    main()
