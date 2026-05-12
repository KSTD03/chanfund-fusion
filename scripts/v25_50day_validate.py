#!/usr/bin/env python3
"""v2.5 50天快速验证：确认 scores 接入修复"""
import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
sys.path.insert(0, '/home/quant/.openclaw/workspace')

OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/output/v25")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(),
    logging.FileHandler(str(OUTPUT_DIR / "v25_validate.log"), mode="w", encoding="utf-8")])
logger = logging.getLogger("v25_val")

from scripts.task3_v21_backtest import load_data, load_scores_parquet, get_next_trade_date

TP1_PCT, TP1_REDUCE, TP2_PCT, TP2_REDUCE, FUND_SCORE_MIN = 0.08, 0.30, 0.13, 0.50, 30.0
ENV_MULT = {"BULL": 1.0, "OSCILLATE": 0.8, "BEAR": 0.6}
INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE = 1_000_000, 5, 0.20
FIXED_STOP_LOSS, VOL_MIN_RATIO, TP_BREAKEVEN_LOCK = 0.08, 0.3, 0.05

class MAEnvClassifier:
    def __init__(self):
        self.prices = []
        self.current_env = "OSCILLATE"
    def update(self, price: float):
        if price <= 0: return
        self.prices.append(price)
        if len(self.prices) < 60: return
        if len(self.prices) >= 21:
            rets = [(self.prices[i]-self.prices[i-1])/self.prices[i-1] for i in range(-20,0)]
            if np.std(rets) * np.sqrt(252) > 0.35: self.current_env = "OSCILLATE"; return
        ma20 = sum(self.prices[-20:])/20; ma60 = sum(self.prices[-60:])/60
        self.current_env = "BULL" if ma20 > ma60 else "BEAR"

def get_fund_score(code, date_str, cache):
    """查询 <= date_str 的最新 score，无匹配返回 None"""
    sc = cache.get(code)
    if not sc: return None
    for d in reversed(sorted(sc.keys())):
        if d <= date_str: return sc[d]
    return None

def signal_to_factor(subtype, vol_ratio):
    base = {"third_point_buy":0.6,"hard_divergence":0.8}.get(subtype, 0.5)
    return base * min(vol_ratio, 2.0)

def resonance(tech_score, fund_score):
    return min(1.0, tech_score * 0.6 + (fund_score/100.0) * 0.4)

logger.info("="*55)
logger.info("🔰 v2.5 50天快速验证")
logger.info("="*55)

# 1. 加载数据
data = load_data(max_stocks=0)
all_dates = sorted(data.keys())
bt_dates = [d for d in all_dates if "2020-01-01" <= d <= "2026-12-31"]
logger.info(f"  总天数: {len(all_dates)}, 回测期: {len(bt_dates)}天")

# 只用前 50 个回测日
test_dates = bt_dates[:50]
logger.info(f"  验证期: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")

# 2. 加载 scores
scores_cache = load_scores_parquet()
sample_keys = list(scores_cache.keys())[:5]
logger.info(f"  cache key样本: {sample_keys}")
if sample_keys:
    # 测试寻找：用 key 自身查
    r = get_fund_score(sample_keys[0], '2020-01-10', scores_cache)
    logger.info(f"  自查询 '{sample_keys[0]}': {'✅ found' if r is not None else '❌ None'}, score={r}")
    # 测试寻找：用可能的 Qlib 格式查
    clean = sample_keys[0].split(".")[-1]  # "sh.600519" -> "600519"
    qlib_fmt = f"sh.{clean}" if clean[0]=='6' else f"sz.{clean}"
    r2 = get_fund_score(qlib_fmt, '2020-01-10', scores_cache)
    logger.info(f"  Qlib格式查询 '{qlib_fmt}': {'✅ found' if r2 is not None else '❌ None'}, score={r2}")

# 3. 筛选池子
logger.info("  🔍 筛选池子: 近250日均量前800只...")
cutoff = test_dates[0]; vol_ranking = defaultdict(list)
for ds in all_dates:
    if ds >= cutoff: break
    for sym, bar in data.get(ds, {}).items():
        v = bar.get("volume", 0)
        if v > 0: vol_ranking[sym].append(v)
avg_vols = {sym: np.mean(vs[-250:]) for sym, vs in vol_ranking.items() if len(vs) >= 20}
top_stocks = set(sorted(avg_vols, key=avg_vols.get, reverse=True)[:800])
logger.info(f"  筛选后: {len(top_stocks)} 只")

# 4. 引擎
from chanfund_fusion.tech.signal_engine import TechSignalEngine
from chanfund_fusion.config_schema import load_config
cfg = load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion/config.yaml")))
tech_engine = TechSignalEngine(cfg.get("tech", {}))
env_cls = MAEnvClassifier()
warmup_dates = [d for d in all_dates if d < "2020-01-01"]

# 指数序列
index_prices = []
for ds in warmup_dates + test_dates:
    day = data.get(ds, {})
    if day:
        closes = [v["close"] for v in day.values() if v.get("close", 0) > 0]
        index_prices.append(np.mean(closes) if closes else (index_prices[-1] if index_prices else 3000))
    else:
        index_prices.append(index_prices[-1] if index_prices else 3000)

# Phase 1: Warm-up
logger.info("Phase 1: Warm-up")
for i, ds in enumerate(warmup_dates):
    env_cls.update(index_prices[i])
    for stock in top_stocks:
        bar = data.get(ds, {}).get(stock)
        if bar:
            try: tech_engine.update(stock, bar, emit_signals=False)
            except: pass
    if (i+1) % 500 == 0:
        logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)}只")

# Phase 2: 50天回测
cash, peak_equity = INITIAL_CAPITAL, cash = INITIAL_CAPITAL, INITIAL_CAPITAL
positions = {}; cooling = {}; trade_log = []; vol_cache = {}
sig_stats = {}; exit_stats = {}; tp_stats = {}; env_counts = {}
score_stats = {"used": 0, "default": 0, "rejected": 0, "passed": 0}

logger.info("Phase 2: 50天回测...")
for di, ds in enumerate(test_dates):
    trade_date_dt = date.fromisoformat(ds)
    idx = len(warmup_dates) + di
    if idx < len(index_prices): env_cls.update(index_prices[idx])
    env = env_cls.current_env; env_counts[env] = env_counts.get(env, 0) + 1
    env_mult = ENV_MULT.get(env, 1.0)
    day_data = data.get(ds, {})
    if not day_data: continue

    for stock in top_stocks:
        bar = day_data.get(stock)
        if bar:
            try: tech_engine.update(stock, bar, emit_signals=True)
            except: pass
            v = bar.get("volume", 0)
            if v > 0:
                vol_cache.setdefault(stock, []).append(v)
                if len(vol_cache[stock]) > 50: vol_cache[stock] = vol_cache[stock][-50:]

    signals = tech_engine.get_confirmed_signals(trade_date_dt)
    for sig in signals:
        if len(positions) >= MAX_POSITIONS: break
        if sig.symbol in positions or sig.symbol not in top_stocks: continue
        if sig.symbol in cooling and trade_date_dt <= cooling[sig.symbol]: continue
        if sig.symbol not in day_data: continue

        nd = get_next_trade_date(ds, all_dates)
        if nd is None or sig.symbol not in data.get(nd, {}): continue
        entry_price = data[nd][sig.symbol]["open"]
        if entry_price <= 0: continue

        fund_score = get_fund_score(sig.symbol, ds, scores_cache)
        if fund_score is None: fund_score = 50.0; score_stats["default"] += 1
        else: score_stats["used"] += 1
        if fund_score < FUND_SCORE_MIN: score_stats["rejected"] += 1; continue
        score_stats["passed"] += 1

        vols = vol_cache.get(sig.symbol, [])
        if len(vols) >= 20:
            avg_vol = sum(vols[-20:])/20; ev = day_data[sig.symbol].get("volume", 0)
            if avg_vol > 0 and ev < avg_vol * VOL_MIN_RATIO: continue

        vols20 = vol_cache.get(sig.symbol, [])
        vol_ratio = 1.0
        if len(vols20) >= 20:
            mv = sum(vols20[-20:])/20; vol_ratio = day_data[sig.symbol].get("volume", 0)/mv if mv > 0 else 1.0
        tech_factor = signal_to_factor(sig.signal_subtype, vol_ratio)
        res = resonance(tech_factor, fund_score)
        pos_coeff = min(res, 1.0) * POSITION_SIZE / 0.20
        pos_coeff = min(1.0, max(0.0, pos_coeff)) * env_mult
        cost = cash * POSITION_SIZE * pos_coeff
        if cost > cash or cost <= 0: continue
        shares = int(cost / entry_price / 100) * 100
        if shares <= 0: continue; cost = shares * entry_price
        if cost > cash or cost <= 0: continue

        stop_price = entry_price * (1 - FIXED_STOP_LOSS)
        positions[sig.symbol] = {"entry_price": entry_price, "shares": shares,
            "entry_date": ds, "buy_exec_date": nd, "stop_price": stop_price,
            "peak_price": entry_price, "signal_type": sig.signal_subtype,
            "tp1_done": False, "tp2_done": False,
            "fund_score": fund_score, "resonance": round(res,3), "env": env}
        cash -= cost
        sig_stats[sig.signal_subtype] = sig_stats.get(sig.signal_subtype,0)+1
        trade_log.append({"date": ds, "symbol": sig.symbol, "action":"BUY",
            "price": round(entry_price,3), "signal": sig.signal_subtype,
            "fund_score": fund_score, "resonance": round(res,3), "env": env})

    trail_threshold = 0.88 if env == "BULL" else 0.91
    to_close, to_reduce = [], []
    for stock, pos in list(positions.items()):
        price = day_data.get(stock, {}).get("close", pos.get("_last_price", pos["entry_price"]))
        if stock in day_data and day_data[stock].get("close",0) > 0:
            pos["_last_price"] = price
        pos["peak_price"] = max(pos["peak_price"], price)
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
        peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
        hold_days = (trade_date_dt - date.fromisoformat(pos["entry_date"])).days

        if pnl_pct >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
            pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
        if hold_days >= 60 and abs(pnl_pct) < 0.05:
            pos["stop_price"] = min(pos["stop_price"], pos["entry_price"]*0.85)
        low = day_data.get(stock, {}).get("low", price)
        if low <= pos["stop_price"]: to_close.append((stock, "stop_loss", price, "close")); continue
        if peak_pnl >= TP_BREAKEVEN_LOCK:
            retrace = price / pos["peak_price"]
            if retrace <= trail_threshold:
                nd2 = get_next_trade_date(ds, all_dates); sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_close.append((stock, "trailing_stop", sp, "next_open")); tp_stats["trailing_stop"] = tp_stats.get("trailing_stop",0)+1; continue
        if pnl_pct >= TP2_PCT: pos["tp2_done"]=True; nd2 = get_next_trade_date(ds, all_dates); sp=data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price; to_reduce.append((stock,"tp2",TP2_REDUCE,sp,"next_open")); tp_stats["tp2"]=tp_stats.get("tp2",0)+1; continue
        if pnl_pct >= TP1_PCT: pos["tp1_done"]=True; nd2 = get_next_trade_date(ds, all_dates); sp=data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price; to_reduce.append((stock,"tp1",TP1_REDUCE,sp,"next_open")); tp_stats["tp1"]=tp_stats.get("tp1",0)+1; continue

    for sym, reason, rpct, sp, pt in to_reduce:
        pos = positions[sym]; r_shares = int(pos["shares"] * rpct)
        if r_shares > 0 and pos["shares"] > r_shares: pos["shares"] -= r_shares; cash += r_shares * sp; exit_stats[f"reduce_{reason}"] = exit_stats.get(f"reduce_{reason}",0)+1
    for sym, reason, price, pt in to_close:
        pos = positions.pop(sym, None)
        if pos is None: continue
        pnl = (price-pos["entry_price"])/pos["entry_price"]*100; cash += pos["shares"]*price; exit_stats[reason]=exit_stats.get(reason,0)+1
        trade_log.append({"date":ds,"symbol":sym,"action":"SELL","price":round(price,3),"pnl_pct":round(pnl,2),"reason":reason,"hold_days":hold_days,"fund_score":pos["fund_score"],"resonance":pos["resonance"],"env":pos.get("env",env)})

    equity = cash + sum(pos["shares"] * day_data.get(sym,{}).get("close", pos["entry_price"]) for sym, pos in positions.items())
    peak_equity = max(peak_equity, equity)

logger.info(f"\n{'='*30}")
logger.info(f"📊 50天验证结果")
logger.info(f"{'='*30}")
logger.info(f"  交易: 买入{len([t for t in trade_log if t['action']=='BUY'])}/卖出{len([t for t in trade_log if t['action']=='SELL'])}")
logger.info(f"  scores: {score_stats}")

if score_stats["used"] > 0:
    logger.info(f"\n✅ scores接入验证通过! used={score_stats['used']}, default={score_stats['default']}, rejected={score_stats['rejected']}")
else:
    logger.error(f"\n❌ scores接入失败! 全部 default={score_stats['default']}, 仍有bug")

with open(OUTPUT_DIR/"v25_validate.json","w") as f:
    json.dump({"scores": score_stats, "trades": len(trade_log), "passed": score_stats["used"] > 0}, f, indent=2)
logger.info(f"  验证结果: {OUTPUT_DIR/'v25_validate.json'}")
