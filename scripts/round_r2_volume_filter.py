#!/usr/bin/env python3
"""
Round R2: 成交量过滤
===========================
设计 (2026-05-11 22:22):
  - 额外过滤：入场前要求当日成交量 > 80% MA20
  - 其余逻辑完全同 v2.5
  - 测试: 在低量能环境下是否提高信号质量
"""

import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, '/home/quant/.openclaw/workspace')

ROUND = "R2"
OUTPUT_DIR = Path(f"/home/quant/.openclaw/workspace/output/r2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOL_FILTER_RATIO = 0.80  # volume > 80% MA20

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "r2_run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("R2")

from scripts.task3_v21_backtest import (
    load_data, load_scores_parquet, calc_metrics,
    get_fund_score_v21, resonance_score_v21, get_next_trade_date,
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE,
    FIXED_STOP_LOSS, COOLING_PERIOD_DAYS, COOLING_STOP_COUNT, VOL_MIN_RATIO,
    TP_BREAKEVEN_LOCK,
)

TP1_PCT = 0.08
TP1_REDUCE = 0.30
TP2_PCT = 0.13
TP2_REDUCE = 0.50

ENV_MULT = {"BULL": 1.0, "OSCILLATE": 0.8, "BEAR": 0.6}

FUND_SCORE_MIN = 30.0
TECH_WEIGHT = 0.6
FUND_WEIGHT = 0.4

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
            vol = np.std(rets) * np.sqrt(252)
            if vol > 0.35:
                self.current_env = "OSCILLATE"
                return
        ma20 = sum(self.prices[-20:])/20
        ma60 = sum(self.prices[-60:])/60
        self.current_env = "BULL" if ma20 > ma60 else "BEAR"

def to_ts_code(qlib_code: str) -> str:
    parts = qlib_code.split(".")
    if len(parts) != 2: return qlib_code
    m = {"sh":"SH","sz":"SZ","bj":"BJ"}
    return f"{parts[1]}.{m.get(parts[0], parts[0].upper())}"

def get_fund_score(qlib_code: str, date_str: str, cache: dict) -> Optional[float]:
    sc = cache.get(qlib_code)
    if not sc: return None
    for d in reversed(sorted(sc.keys())):
        if d <= date_str:
            return sc[d]
    return None

def signal_to_factor(subtype: str, vol_ratio: float) -> float:
    base = {"third_point_buy":0.6,"hard_divergence":0.8}.get(subtype, 0.5)
    return base * min(vol_ratio, 2.0)

def resonance(tech_score: float, fund_score: float) -> float:
    return min(1.0, tech_score * TECH_WEIGHT + (fund_score/100.0) * FUND_WEIGHT)

# === R2: 成交量过滤 ===
class VolumeFilter:
    def __init__(self, vol_cache, ratio=VOL_FILTER_RATIO):
        self.vol_cache = vol_cache
        self.ratio = ratio
        self.rejected_count = 0
    def check(self, stock: str, ds: str, day_data: dict) -> bool:
        vols = self.vol_cache.get(stock, [])
        if len(vols) < 20:
            # 数据不足时放行
            return True
        ma20 = sum(vols[-20:]) / 20
        today_vol = day_data.get("volume", 0)
        if ma20 <= 0 or today_vol <= 0:
            return True
        ratio = today_vol / ma20
        if ratio < self.ratio:
            self.rejected_count += 1
            return False
        return True

def run_r2():
    t_start = time.time()
    logger.info(f"{'='*55}")
    logger.info(f"🔰 R2 — 成交量过滤 (volume > {VOL_FILTER_RATIO*100:.0f}% MA20)")
    logger.info(f"  其余同 v2.5: TP1=8%, 尾随0.88/0.91, 池=800只")
    logger.info(f"{'='*55}")

    # 1. 加载数据
    data = load_data(max_stocks=0)
    all_dates = sorted(data.keys())
    logger.info(f"  日期: {all_dates[0]} ~ {all_dates[-1]}, {len(all_dates)}天")

    # 2. 加载scores
    scores_cache = load_scores_parquet()

    # 3. 筛选池子
    logger.info("  🔍 筛选池子: 近250日均量前800只...")
    cutoff = "2020-01-01"
    vol_ranking = defaultdict(list)
    for ds in all_dates:
        if ds >= cutoff: break
        day = data.get(ds, {})
        for sym, bar in day.items():
            v = bar.get("volume", 0)
            if v > 0: vol_ranking[sym].append(v)
    avg_vols = {sym: np.mean(vs[-250:]) for sym, vs in vol_ranking.items() if len(vs) >= 20}
    top_stocks = set(sorted(avg_vols, key=avg_vols.get, reverse=True)[:800])
    logger.info(f"  筛选后: {len(top_stocks)} 只")

    # 4. 初始化
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion/config.yaml")))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))

    env_cls = MAEnvClassifier()
    warmup_dates = [d for d in all_dates if d < "2020-01-01"]
    bt_dates = [d for d in all_dates if "2020-01-01" <= d <= "2026-12-31"]

    index_prices = []
    for ds in warmup_dates + bt_dates:
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
        day_data = data.get(ds, {})
        for stock in top_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=False)
                except: pass
        if (i+1) % 500 == 0:
            logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
    logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)}只")

    # 状态
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
    score_stats = {"used":0,"default":0,"rejected":0,"passed":0}
    vol_filter_obj = VolumeFilter(vol_cache, VOL_FILTER_RATIO)

    # Phase 2: Backtest
    logger.info("Phase 2: Backtest R2...")
    for di, ds in enumerate(bt_dates):
        trade_date_dt = date.fromisoformat(ds)
        idx = len(warmup_dates) + di
        if idx < len(index_prices):
            env_cls.update(index_prices[idx])
        env = env_cls.current_env
        env_counts[env] = env_counts.get(env, 0) + 1
        env_mult = ENV_MULT.get(env, 1.0)
        day_data = data.get(ds, {})
        if not day_data: continue

        # A. 更新技术引擎 + vol cache
        for stock in top_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=True)
                except: pass
                v = day_data[stock].get("volume", 0)
                if v > 0:
                    vol_cache.setdefault(stock, []).append(v)
                    if len(vol_cache[stock]) > 50:
                        vol_cache[stock] = vol_cache[stock][-50:]

        # B. 信号 + 过滤（含R2成交量过滤）
        signals = tech_engine.get_confirmed_signals(trade_date_dt)
        for sig in signals:
            if len(positions) >= MAX_POSITIONS: break
            if sig.symbol in positions: continue
            if sig.symbol not in top_stocks: continue
            if sig.symbol in cooling and trade_date_dt <= cooling[sig.symbol]: continue
            if sig.symbol not in day_data: continue

            # === R2: 成交量过滤 ===
            if not vol_filter_obj.check(sig.symbol, ds, day_data[sig.symbol]):
                continue

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
                avg_vol = sum(vols[-20:])/20
                ev = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and ev < avg_vol * VOL_MIN_RATIO: continue

            vols20 = vol_cache.get(sig.symbol, [])
            vol_ratio = 1.0
            if len(vols20) >= 20:
                mv = sum(vols20[-20:])/20
                vol_ratio = day_data[sig.symbol].get("volume", 0)/mv if mv > 0 else 1.0
            tech_factor = signal_to_factor(sig.signal_subtype, vol_ratio)
            res = resonance(tech_factor, fund_score)
            pos_coeff = min(res, 1.0) * POSITION_SIZE / 0.20
            pos_coeff = min(1.0, max(0.0, pos_coeff)) * env_mult
            cost = cash * POSITION_SIZE * pos_coeff
            if cost > cash or cost <= 0: continue
            shares = int(cost / entry_price / 100) * 100
            if shares <= 0: continue
            cost = shares * entry_price
            if cost > cash or cost <= 0: continue

            positions[sig.symbol] = {
                "entry_price": entry_price, "shares": shares,
                "entry_date": ds, "buy_exec_date": nd,
                "stop_price": entry_price * (1 - FIXED_STOP_LOSS),
                "peak_price": entry_price,
                "signal_type": sig.signal_subtype,
                "tp1_done": False, "tp2_done": False,
                "fund_score": fund_score, "resonance": round(res,3), "env": env,
            }
            cash -= cost
            sig_stats[sig.signal_subtype] = sig_stats.get(sig.signal_subtype,0)+1
            trade_log.append({"date": ds, "symbol": sig.symbol, "action":"BUY",
                "price": round(entry_price,3), "signal": sig.signal_subtype,
                "fund_score": fund_score, "resonance": round(res,3), "env": env})

        # C. 止盈止损（同v2.5）
        trail_threshold = 0.88 if env == "BULL" else 0.91
        to_close, to_reduce = [], []
        for stock, pos in list(positions.items()):
            if stock in day_data and day_data[stock].get("close",0) > 0:
                price = day_data[stock]["close"]; pos["_last_price"] = price
            else: price = pos.get("_last_price", pos["entry_price"])
            pos["peak_price"] = max(pos["peak_price"], price)
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
            hold_days = (trade_date_dt - date.fromisoformat(pos["entry_date"])).days

            if pnl_pct >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            if hold_days >= 60 and abs(pnl_pct) < 0.05:
                pos["stop_price"] = min(pos["stop_price"], pos["entry_price"]*0.85)

            low = day_data.get(stock, {}).get("low", price)
            if low <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price, "close")); continue

            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= trail_threshold:
                    nd2 = get_next_trade_date(ds, all_dates)
                    sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                    to_close.append((stock, "trailing_stop", sp, "next_open"))
                    tp_stats["trailing_stop"] = tp_stats.get("trailing_stop",0)+1
                    continue

            if pnl_pct >= TP2_PCT:
                pos["tp2_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock, "tp2", TP2_REDUCE, sp, "next_open"))
                tp_stats["tp2"] = tp_stats.get("tp2",0)+1
                continue

            if pnl_pct >= TP1_PCT:
                pos["tp1_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock, "tp1", TP1_REDUCE, sp, "next_open"))
                tp_stats["tp1"] = tp_stats.get("tp1",0)+1
                continue

        for sym, reason, rpct, sp, pt in to_reduce:
            pos = positions[sym]
            r_shares = int(pos["shares"] * rpct)
            if r_shares > 0 and pos["shares"] > r_shares:
                pos["shares"] -= r_shares; cash += r_shares * sp
                exit_stats[f"reduce_{reason}"] = exit_stats.get(f"reduce_{reason}",0)+1

        for sym, reason, price, pt in to_close:
            pos = positions.pop(sym, None)
            if pos is None: continue
            pnl = (price - pos["entry_price"]) / pos["entry_price"] * 100
            cash += pos["shares"] * price
            exit_stats[reason] = exit_stats.get(reason,0)+1
            trade_log.append({"date": ds, "symbol": sym, "action":"SELL",
                "price": round(price,3), "pnl_pct": round(pnl,2), "reason": reason,
                "hold_days": hold_days, "fund_score": pos["fund_score"],
                "resonance": pos["resonance"], "env": pos.get("env",env)})
            if reason == "stop_loss" and pos.get("cooling_count",0) >= COOLING_STOP_COUNT:
                cooling[sym] = trade_date_dt + timedelta(days=COOLING_PERIOD_DAYS)

        equity = cash + sum(
            pos["shares"] * day_data.get(sym,{}).get("close", pos["entry_price"])
            for sym, pos in positions.items()
        )
        peak_equity = max(peak_equity, equity)
        dd = (equity / peak_equity - 1) * 100
        equity_curve.append({"date": ds, "nav": round(equity,2), "dd": round(dd,2)})
        if (di+1) % 200 == 0 or di == len(bt_dates)-1:
            logger.info(f"  [{ds}]({di+1}/{len(bt_dates)}) NAV={equity:,.0f} pos={len(positions)} t={len(trade_log)} env={env}")

    # 指标
    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    buys = [t for t in trade_log if t["action"]=="BUY"]
    sells = [t for t in trade_log if t["action"]=="SELL"]
    wins = [t for t in sells if t.get("pnl_pct",0) > 0]
    win_rate = len(wins) / max(len(sells),1) * 100
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl_pct"]) for t in sells if t.get("pnl_pct",0) <= 0]) or 1
    profit_factor = sum(t["pnl_pct"] for t in wins) / max(sum(abs(t["pnl_pct"]) for t in sells if t.get("pnl_pct",0) <= 0), 1)

    logger.info(f"\n{'='*55}")
    logger.info(f"📊 R2 全量结果")
    logger.info(f"{'='*55}")
    logger.info(f"  总收益率: {metrics['total_return']:.2f}%")
    logger.info(f"  年化收益率: {metrics['ann_return']:.2f}%")
    logger.info(f"  夏普比率: {metrics['sharpe']:.4f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {win_rate:.1f}%")
    logger.info(f"  盈亏比: {metrics.get('win_loss_ratio', profit_factor):.2f}")
    logger.info(f"  交易: {len(trade_log)} (买入{len(buys)}/卖出{len(sells)})")
    logger.info(f"  信号: {sig_stats}")
    logger.info(f"  平仓: {exit_stats}")
    logger.info(f"  TP: {tp_stats}")
    logger.info(f"  环境: {env_counts}")
    logger.info(f"  scores: {score_stats}")
    logger.info(f"  R2: 成交量过滤 ratio={VOL_FILTER_RATIO}, 被拒={vol_filter_obj.rejected_count}")

    pd = __import__('pandas')
    pd.DataFrame(trade_log).to_csv(OUTPUT_DIR/"r2_trades.csv", index=False)
    pd.DataFrame(equity_curve).to_csv(OUTPUT_DIR/"r2_equity.csv", index=False)
    metrics_save = {k:v for k,v in metrics.items() if isinstance(v,(int,float,str))}
    metrics_save.update({"win_rate": round(win_rate,1), "profit_factor": round(profit_factor,2),
                         "vol_filter_ratio": VOL_FILTER_RATIO,
                         "vol_filter_rejected": vol_filter_obj.rejected_count})
    with open(OUTPUT_DIR/"r2_summary.json","w") as f:
        json.dump(metrics_save, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    logger.info(f"\n⏱️ 总耗时: {elapsed:.0f}s")
    logger.info(f"💾 产出: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_r2()
