#!/usr/bin/env python3
"""
v3.2 — v3.1 + 渐进式仓位上限 (VOL_CAP)
==============================================
改动（仅一处）：在波动率仓位公式后加 VOL_CAP 硬性上限
  BULL=30%, OSCILLATE=15%, BEAR=10%
  其余完全同 v3.1
"""

import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, '/home/quant/.openclaw/workspace')

OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/output/v32")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "v32_run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("v32")

from scripts.task3_v21_backtest import (
    load_data, load_scores_parquet, calc_metrics,
    get_fund_score_v21, resonance_score_v21, get_next_trade_date,
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE,
    COOLING_PERIOD_DAYS, COOLING_STOP_COUNT, VOL_MIN_RATIO,
    TP_BREAKEVEN_LOCK,
)

FIXED_STOP_LOSS = 0.08
FUND_SCORE_MIN = 30.0
TECH_WEIGHT = 0.6
FUND_WEIGHT = 0.4

# ===== v3.2 新参数：VOL_CAP =====
VOL_CAP = {
    "BULL": 0.30,       # 牛市最多单只30%
    "OSCILLATE": 0.15,  # 震荡市最多单只15%
    "BEAR": 0.10,       # 熊市最多单只10%
}

# ============================================================
# 数据加载
# ============================================================
def load_csi300():
    import pandas as pd
    df = pd.read_csv("/home/quant/.openclaw/workspace/data/index_sh000300.csv")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    data = {}
    for _, row in df.iterrows():
        data[row['date']] = (row['open'], row['high'], row['low'], row['close'])
    logger.info(f"✅ 沪深300加载: {len(data)} 天")
    return data

# ============================================================
# 环境分类器 (同v3.1)
# ============================================================
class CSI300EnvClassifier:
    def __init__(self):
        self.dates, self.closes, self.highs, self.lows = [], [], [], []
        self.current_env = "OSCILLATE"
        self.daily_log = []

    def update(self, date_str, ohlc):
        o, h, l, c = ohlc
        if c <= 0: return
        self.dates.append(date_str); self.closes.append(c)
        self.highs.append(h); self.lows.append(l)
        if len(self.closes) < 120: return

        ma20 = np.mean(self.closes[-20:]); ma60 = np.mean(self.closes[-60:]); ma120 = np.mean(self.closes[-120:])
        bull_trend = ma20 > ma60 > ma120; bear_trend = ma20 < ma60 < ma120
        adx = self._calc_adx(14); strong_trend = adx > 22
        atr20 = self._calc_atr(20); atr_ratio = atr20 / c

        if len(self.closes) >= 200:
            hist = []
            for i in range(20, min(120, len(self.closes)-1) + 20):
                tr = [max(self.highs[j]-self.lows[j], abs(self.highs[j]-self.closes[j-1]), abs(self.lows[j]-self.closes[j-1]))
                      for j in range(i-20, i)]
                hist.append(np.mean(tr) / self.closes[i-1])
            p85, p30 = np.percentile(hist, 85), np.percentile(hist, 30)
            high_vol, low_vol = atr_ratio > p85, atr_ratio < p30
        else:
            high_vol, low_vol = False, False

        if bull_trend and strong_trend and not high_vol:
            self.current_env = "BULL"
        elif bear_trend and strong_trend and not high_vol:
            self.current_env = "BEAR"
        elif high_vol or (not strong_trend and not low_vol):
            self.current_env = "OSCILLATE"
        elif low_vol:
            self.current_env = "BULL" if bull_trend else "BEAR"
        else:
            self.current_env = "BULL" if bull_trend else "BEAR"

    def _calc_adx(self, p=14):
        n = len(self.closes); lookback = min(p*3, n-1)
        if n < p*2+2: return 0.0
        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(n-lookback, n):
            hl = self.highs[i]-self.lows[i]
            hc = abs(self.highs[i]-self.closes[i-1])
            lc = abs(self.lows[i]-self.closes[i-1])
            tr_list.append(max(hl, hc, lc))
            up = self.highs[i]-self.highs[i-1]; down = self.lows[i-1]-self.lows[i]
            pdm_list.append(up if up>down and up>0 else 0.0)
            ndm_list.append(down if down>up and down>0 else 0.0)
        tr_p = np.mean(tr_list[-p:]) if len(tr_list)>=p else np.mean(tr_list)
        pdm_p = np.mean(pdm_list[-p:]); ndm_p = np.mean(ndm_list[-p:])
        dp = 100*pdm_p/tr_p if tr_p>0 else 0; dn = 100*ndm_p/tr_p if tr_p>0 else 0
        return 100*abs(dp-dn)/(dp+dn) if dp+dn>0 else 0.0

    def _calc_atr(self, p=20):
        return np.mean([max(self.highs[i]-self.lows[i], abs(self.highs[i]-self.closes[i-1]), abs(self.lows[i]-self.closes[i-1]))
                       for i in range(-p, 0)])

class MAEnvClassifier:
    def __init__(self):
        self.prices = []; self.current_env = "OSCILLATE"
    def update(self, price):
        if price <= 0: return
        self.prices.append(price)
        if len(self.prices) < 60: return
        if len(self.prices) >= 21:
            rets = [(self.prices[i]-self.prices[i-1])/self.prices[i-1] for i in range(-20,0)]
            if np.std(rets)*np.sqrt(252) > 0.35: return
        ma20 = sum(self.prices[-20:])/20; ma60 = sum(self.prices[-60:])/60
        self.current_env = "BULL" if ma20 > ma60 else "BEAR"

def get_fund_score(qlib_code, date_str, cache):
    sc = cache.get(qlib_code)
    if not sc: return None
    for d in reversed(sorted(sc.keys())):
        if d <= date_str: return sc[d]
    return None

def calc_stock_atr(stock_data):
    if len(stock_data) < 21: return 0.02
    recent = stock_data[-(21):]
    tr = [max(recent[i][1]-recent[i][2], abs(recent[i][1]-recent[i-1][3]), abs(recent[i][2]-recent[i-1][3]))
          for i in range(1, len(recent))]
    return np.mean(tr) / recent[-1][3] if recent[-1][3] > 0 else 0.02

def signal_to_baseline(subtype):
    return {"third_point_buy": 0.6, "hard_divergence": 1.0}.get(subtype, 0.5)

def get_env_params(env):
    if env == "BULL":    return {"tp1": 0.12, "tp2": 0.18, "trail": 0.88, "env_stop": None}
    if env == "OSCILLATE": return {"tp1": 0.06, "tp2": 0.10, "trail": 0.94, "env_stop": None}
    return {"tp1": 0.03, "tp2": 0.05, "trail": None, "env_stop": 0.03}

def run_v32():
    t_start = time.time()
    logger.info(f"{'='*55}")
    logger.info(f"🔰 v3.2 — VOL_CAP 回撤控制")
    logger.info(f"  BULL≤30%, OSC≤15%, BEAR≤10% (单只硬性上限)")
    logger.info(f"  其余同v3.1: CSI300环境分类器 + 波动率仓位")
    logger.info(f"{'='*55}")

    data = load_data(max_stocks=0)
    all_dates = sorted(data.keys())
    scores_cache = load_scores_parquet()

    cutoff = "2020-01-01"
    vol_ranking = defaultdict(list)
    for ds in all_dates:
        if ds >= cutoff: break
        for sym, bar in data.get(ds, {}).items():
            if bar.get("volume", 0) > 0: vol_ranking[sym].append(bar["volume"])
    avg_vols = {sym: np.mean(vs[-250:]) for sym, vs in vol_ranking.items() if len(vs) >= 20}
    top_stocks = set(sorted(avg_vols, key=avg_vols.get, reverse=True)[:800])
    logger.info(f"  筛选: {len(top_stocks)}只 | {all_dates[0]} ~ {all_dates[-1]}")

    csi300 = load_csi300()

    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion/config.yaml")))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))

    warmup_dates = [d for d in all_dates if d < "2020-01-01"]
    bt_dates = [d for d in all_dates if "2020-01-01" <= d <= "2026-12-31"]

    csi_env = CSI300EnvClassifier()
    old_env = MAEnvClassifier()

    logger.info("Phase 1: Warm-up")
    for i, ds in enumerate(warmup_dates):
        if ds in csi300: csi_env.update(ds, csi300[ds]); old_env.update(csi300[ds][3])
        for stock in top_stocks:
            if stock in data.get(ds, {}):
                try: tech_engine.update(stock, data[ds][stock], False)
                except: pass
        if (i+1) % 500 == 0: logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
    logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)}只")

    cash, peak_equity = INITIAL_CAPITAL, INITIAL_CAPITAL
    positions, equity_curve, cooling, trade_log = {}, [], {}, []
    vol_cache, sig_stats, exit_stats, tp_stats = {}, {}, {}, {}
    env_counts, old_env_counts = {}, {}
    score_stats = {"used":0,"default":0,"rejected":0,"passed":0}
    stock_history = defaultdict(list)

    logger.info("Phase 2: Backtest v3.2...")
    for di, ds in enumerate(bt_dates):
        trade_date_dt = date.fromisoformat(ds)
        if ds in csi300:
            csi_env.update(ds, csi300[ds]); old_env.update(csi300[ds][3])
            csi_env.daily_log.append(csi_env.current_env)
        env = csi_env.current_env
        old_env_counts[old_env.current_env] = old_env_counts.get(old_env.current_env, 0) + 1
        env_counts[env] = env_counts.get(env, 0) + 1
        ep = get_env_params(env)
        ecap = VOL_CAP.get(env, 0.15)  # VOL_CAP 替代 env_mult

        day_data = data.get(ds, {})
        if not day_data: continue

        for stock in top_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], True)
                except: pass
                b = day_data[stock]
                stock_history[stock].append((ds, b.get("high",0), b.get("low",0), b.get("close",0)))
                if len(stock_history[stock]) > 250: stock_history[stock] = stock_history[stock][-250:]
                if b.get("volume", 0) > 0:
                    vol_cache.setdefault(stock, []).append(b["volume"])
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
                if sum(vols[-20:])/20 > 0 and day_data[sig.symbol].get("volume", 0) < sum(vols[-20:])/20 * VOL_MIN_RATIO:
                    continue

            # 波动率仓位公式（同v3.1）
            base = signal_to_baseline(sig.signal_subtype)
            atr_r = calc_stock_atr(stock_history[sig.symbol])
            atr_r = max(atr_r, 0.005)
            vol_fc = base / max(atr_r, 0.01)
            fund_m = fund_score / 100.0
            target_r = 0.04

            raw_amt = (cash * target_r * vol_fc * fund_m) / max(atr_r, 0.01)

            # ======== v3.2: VOL_CAP 硬性上限 ========
            raw_amt = min(raw_amt, cash * ecap)

            shares = int(raw_amt / entry_price / 100) * 100
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
                "fund_score": fund_score, "env": env,
                "pct_of_cash": round(cost / cash * 100, 1),
            }
            cash -= cost
            sig_stats[sig.signal_subtype] = sig_stats.get(sig.signal_subtype,0)+1
            trade_log.append({"date": ds, "symbol": sig.symbol, "action":"BUY",
                "price": round(entry_price,3), "signal": sig.signal_subtype,
                "fund_score": fund_score, "env": env, "pct": round(cost/cash*100, 1)})

        # C. 止盈止损（同R1）
        to_close, to_reduce = [], []
        for stock, pos in list(positions.items()):
            price = day_data[stock]["close"] if stock in day_data and day_data[stock].get("close",0) > 0 else pos.get("_last_price", pos["entry_price"])
            pos["_last_price"] = price
            pos["peak_price"] = max(pos["peak_price"], price)
            pnl = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
            hold = (trade_date_dt - date.fromisoformat(pos["entry_date"])).days

            if pnl >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            if hold >= 60 and abs(pnl) < 0.05:
                pos["stop_price"] = min(pos["stop_price"], pos["entry_price"]*0.85)

            if ep["env_stop"] is not None:
                es = pos["entry_price"]*(1-ep["env_stop"])
                if day_data.get(stock,{}).get("low",price) <= es: to_close.append((stock,"env_stop_loss",price,"close")); continue
            if day_data.get(stock,{}).get("low",price) <= pos["stop_price"]: to_close.append((stock,"stop_loss",price,"close")); continue
            if ep["trail"] and peak_pnl >= TP_BREAKEVEN_LOCK and price/pos["peak_price"] <= ep["trail"]:
                nd2 = get_next_trade_date(ds, all_dates); sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_close.append((stock,"trailing_stop",sp,"next_open")); tp_stats["trailing_stop"]=tp_stats.get("trailing_stop",0)+1; continue
            if pnl >= ep["tp2"]:
                pos["tp2_done"]=True; nd2=get_next_trade_date(ds,all_dates); sp=data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock,"tp2",0.50,sp,"next_open")); tp_stats["tp2"]=tp_stats.get("tp2",0)+1; continue
            if pnl >= ep["tp1"]:
                pos["tp1_done"]=True; nd2=get_next_trade_date(ds,all_dates); sp=data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock,"tp1",0.30,sp,"next_open")); tp_stats["tp1"]=tp_stats.get("tp1",0)+1; continue

        for sym, reason, rpct, sp, pt in to_reduce:
            p = positions[sym]; rs = int(p["shares"]*rpct)
            if rs > 0 and p["shares"] > rs: p["shares"]-=rs; cash+=rs*sp; exit_stats[f"reduce_{reason}"]=exit_stats.get(f"reduce_{reason}",0)+1
        for sym, reason, price, pt in to_close:
            p = positions.pop(sym, None)
            if not p: continue
            pnl = (price-p["entry_price"])/p["entry_price"]*100; cash+=p["shares"]*price
            exit_stats[reason]=exit_stats.get(reason,0)+1
            trade_log.append({"date":ds,"symbol":sym,"action":"SELL","price":round(price,3),"pnl_pct":round(pnl,2),
                "reason":reason,"hold_days":hold,"fund_score":p["fund_score"],"env":p.get("env",env)})
            if reason=="stop_loss": cooling[sym]=trade_date_dt+timedelta(days=COOLING_PERIOD_DAYS)

        equity = cash + sum(p["shares"]*day_data.get(sym,{}).get("close",p["entry_price"]) for sym,p in positions.items())
        peak_equity = max(peak_equity, equity)
        equity_curve.append({"date":ds,"nav":round(equity,2),"dd":round((equity/peak_equity-1)*100,2)})
        if (di+1)%200==0 or di==len(bt_dates)-1:
            logger.info(f"  [{ds}]({di+1}/{len(bt_dates)}) NAV={equity:,.0f} pos={len(positions)} t={len(trade_log)} env={env}(旧={old_env.current_env})")

    # 指标
    pd = __import__('pandas')
    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    buys = [t for t in trade_log if t["action"]=="BUY"]; sells = [t for t in trade_log if t["action"]=="SELL"]
    wins = [t for t in sells if t.get("pnl_pct",0)>0]; wr=len(wins)/max(len(sells),1)*100
    pf = sum(t["pnl_pct"] for t in wins)/max(sum(abs(t["pnl_pct"]) for t in sells if t.get("pnl_pct",0)<=0),1)

    logger.info(f"\n{'='*55}")
    logger.info(f"📊 v3.2 全量结果")
    logger.info(f"{'='*55}")
    logger.info(f"  总收益率: {metrics['total_return']:.2f}%")
    logger.info(f"  年化收益率: {metrics['ann_return']:.2f}%")
    logger.info(f"  夏普比率: {metrics['sharpe']:.4f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {wr:.1f}%  盈亏比: {metrics.get('win_loss_ratio',pf):.2f}")
    logger.info(f"  交易: {len(trade_log)}(买入{len(buys)}/卖出{len(sells)})")
    logger.info(f"  信号: {sig_stats}  平仓: {exit_stats}  TP: {tp_stats}")
    logger.info(f"  环境: {dict(sorted(env_counts.items()))}(旧R1: {dict(sorted(old_env_counts.items()))})")
    logger.info(f"  scores: {score_stats}")
    logger.info(f"  VOL_CAP: BULL=30%, OSC=15%, BEAR=10%")

    # 2024环境对照
    logger.info(f"\n📋 2024年环境切换对照")
    bt_env = csi_env.daily_log
    log_map = {bt_dates[i]: bt_env[i] for i in range(min(len(bt_dates),len(bt_env)))}
    old_cls2 = MAEnvClassifier()
    dates_24, new_24, old_24 = [], [], []
    for d in all_dates:
        if d in csi300: old_cls2.update(csi300[d][3])
        if "2024-01-01" <= d <= "2024-12-31":
            dates_24.append(d); old_24.append(old_cls2.current_env)
            new_24.append(log_map.get(d, "N/A"))
    nb = sum(1 for e in new_24 if e=="BULL"); ns = sum(1 for e in new_24 if e=="OSCILLATE"); nr = sum(1 for e in new_24 if e=="BEAR")
    ob = sum(1 for e in old_24 if e=="BULL"); os_ = sum(1 for e in old_24 if e=="OSCILLATE"); or_ = sum(1 for e in old_24 if e=="BEAR")
    mc = [(n,o) for d,n,o in zip(dates_24,new_24,old_24) if "2024-01-01"<=d<="2024-02-08"]
    logger.info(f"  2024年({len(dates_24)}天) 新: B={nb} O={ns} B={nr} | 旧: B={ob} O={os_} B={or_}")
    logger.info(f"  微盘危机BEAR: 新={sum(1 for n,_ in mc if n=='BEAR')}/28 旧={sum(1 for _,o in mc if o=='BEAR')}/28")
    for d,n,o in zip(dates_24,new_24,old_24):
        if "2024-09-15"<=d<="2024-10-15": logger.info(f"    {d}: 新={n} 旧={o}")
    pd.DataFrame({"date":dates_24,"new_env":new_24,"old_env":old_24}).to_csv(OUTPUT_DIR/"v32_env_comparison_2024.csv",index=False)

    pd.DataFrame(trade_log).to_csv(OUTPUT_DIR/"v32_trades.csv",index=False)
    pd.DataFrame(equity_curve).to_csv(OUTPUT_DIR/"v32_equity.csv",index=False)
    ms = {k:v for k,v in metrics.items() if isinstance(v,(int,float,str))}
    ms.update({"win_rate":round(wr,1),"profit_factor":round(pf,2)})
    with open(OUTPUT_DIR/"v32_summary.json","w") as f: json.dump(ms,f,indent=2,ensure_ascii=False)
    logger.info(f"\n⏱️ {time.time()-t_start:.0f}s 💾 {OUTPUT_DIR}")

if __name__ == "__main__":
    run_v32()
