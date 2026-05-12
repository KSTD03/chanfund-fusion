#!/usr/bin/env python3
"""
v3.1 — 基于 CSI 300 真实 OHLC 的环境分类器 + 波动率仓位
==========================================================
相对 v3.0 的修复：
  1. 环境分类器使用沪深300真实OHLC数据（非合成指数）
  2. ATR比率基于真实high/low计算
  3. 分类阈值基于指数本身的历史分位数而非绝对数值
"""

import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, '/home/quant/.openclaw/workspace')

OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/output/v31")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "v31_run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("v31")

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

# ============================================================
# CSI 300 数据加载
# ============================================================
def load_csi300():
    """加载沪深300 OHLC 数据，返回 dict[date_str] -> (open, high, low, close)"""
    import pandas as pd
    df = pd.read_csv("/home/quant/.openclaw/workspace/data/index_sh000300.csv")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    data = {}
    for _, row in df.iterrows():
        data[row['date']] = (row['open'], row['high'], row['low'], row['close'])
    logger.info(f"✅ 沪深300加载: {len(data)} 天, 范围 {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    return data

# ============================================================
# 环境分类器（基于CSI 300真实数据）
# ============================================================
class CSI300EnvClassifier:
    """基于沪深300真实OHLC数据的三层环境分类器"""
    
    def __init__(self):
        self.dates = []
        self.closes = []
        self.highs = []
        self.lows = []
        self.current_env = "OSCILLATE"
        self.daily_log = []  # date -> env
    
    def update(self, date_str: str, ohlc: tuple):
        """用沪深300的day数据更新"""
        o, h, l, c = ohlc
        if c <= 0: return
        self.dates.append(date_str)
        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(l)
        
        if len(self.closes) < 120:
            return
        
        # -- 层1：趋势方向 --
        ma20 = np.mean(self.closes[-20:])
        ma60 = np.mean(self.closes[-60:])
        ma120 = np.mean(self.closes[-120:])
        bull_trend = ma20 > ma60 > ma120
        bear_trend = ma20 < ma60 < ma120
        
        # -- 层2：ADX(14) --
        adx = self._calc_adx(14)
        strong_trend = adx > 22
        
        # -- 层3：ATR比率 --
        atr20 = self._calc_atr(20)
        atr_ratio = atr20 / c
        
        # ATR的历史分位数
        if len(self.closes) >= 200:
            lookback = min(120, len(self.closes) - 1)
            hist_atr_ratios = []
            for i in range(20, lookback + 20):
                tr = [max(self.highs[j]-self.lows[j],
                           abs(self.highs[j]-self.closes[j-1]),
                           abs(self.lows[j]-self.closes[j-1]))
                      for j in range(i-20, i)]
                hist_atr_ratios.append(np.mean(tr) / self.closes[i-1])
            p85 = np.percentile(hist_atr_ratios, 85)
            p30 = np.percentile(hist_atr_ratios, 30)
            high_vol = atr_ratio > p85
            low_vol = atr_ratio < p30
        else:
            high_vol = False
            low_vol = False
        
        # -- 综合 -- 
        if bull_trend and strong_trend and not high_vol:
            self.current_env = "BULL"
        elif bear_trend and strong_trend and not high_vol:
            self.current_env = "BEAR"
        elif high_vol or (not strong_trend and not low_vol):
            self.current_env = "OSCILLATE"
        elif low_vol:
            # 低波时按趋势走
            self.current_env = "BULL" if bull_trend else "BEAR"
        else:
            self.current_env = "BULL" if bull_trend else "BEAR"
    
    def _calc_adx(self, period=14):
        """计算ADX"""
        n = len(self.closes)
        if n < period * 2 + 2: return 0.0
        lookback = min(period * 3, n - 1)
        
        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(n - lookback, n):
            hl = self.highs[i] - self.lows[i]
            hc = abs(self.highs[i] - self.closes[i-1])
            lc = abs(self.lows[i] - self.closes[i-1])
            tr_list.append(max(hl, hc, lc))
            
            up_move = self.highs[i] - self.highs[i-1]
            down_move = self.lows[i-1] - self.lows[i]
            pdm_list.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            ndm_list.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        
        tr14 = np.mean(tr_list[-period:])
        pdm14 = np.mean(pdm_list[-period:])
        ndm14 = np.mean(ndm_list[-period:])
        
        di_plus = 100 * pdm14 / tr14 if tr14 > 0 else 0
        di_minus = 100 * ndm14 / tr14 if tr14 > 0 else 0
        
        if di_plus + di_minus == 0: return 0.0
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        return dx
    
    def _calc_atr(self, period=20):
        """计算ATR"""
        tr_vals = []
        for i in range(-period, 0):
            hl = self.highs[i] - self.lows[i]
            hc = abs(self.highs[i] - self.closes[i-1])
            lc = abs(self.lows[i] - self.closes[i-1])
            tr_vals.append(max(hl, hc, lc))
        return np.mean(tr_vals)

# ============================================================
# 工具函数
# ============================================================
def to_ts_code(qlib_code):
    parts = qlib_code.split(".")
    if len(parts) != 2: return qlib_code
    m = {"sh":"SH","sz":"SZ","bj":"BJ"}
    return f"{parts[1]}.{m.get(parts[0], parts[0].upper())}"

def get_fund_score(qlib_code, date_str, cache):
    sc = cache.get(qlib_code)
    if not sc: return None
    for d in reversed(sorted(sc.keys())):
        if d <= date_str: return sc[d]
    return None

def calc_stock_atr_ratio(stock_data):
    if len(stock_data) < 21: return 0.02
    recent = stock_data[-(21):]
    tr_vals = []
    for i in range(1, len(recent)):
        _, h, l, c = recent[i]
        _, _, _, cp = recent[i-1]
        tr_vals.append(max(h-l, abs(h-cp), abs(l-cp)))
    atr = np.mean(tr_vals)
    _, _, _, close = recent[-1]
    return atr / close if close > 0 else 0.02

def signal_to_baseline(subtype):
    return {"third_point_buy": 0.6, "hard_divergence": 1.0}.get(subtype, 0.5)

def get_env_params(env):
    if env == "BULL":
        return {"tp1": 0.12, "tp2": 0.18, "trail": 0.88, "env_stop": None}
    elif env == "OSCILLATE":
        return {"tp1": 0.06, "tp2": 0.10, "trail": 0.94, "env_stop": None}
    else:
        return {"tp1": 0.03, "tp2": 0.05, "trail": None, "env_stop": 0.03}

def env_cap(env):
    return {"BULL": 1.0, "OSCILLATE": 0.7, "BEAR": 0.5}.get(env, 1.0)

# ============================================================
# R1旧分类器（用于对照）
# ============================================================
class MAEnvClassifier:
    def __init__(self):
        self.prices = []
        self.current_env = "OSCILLATE"
    def update(self, price):
        if price <= 0: return
        self.prices.append(price)
        if len(self.prices) < 60: return
        if len(self.prices) >= 21:
            rets = [(self.prices[i]-self.prices[i-1])/self.prices[i-1] for i in range(-20,0)]
            vol = np.std(rets) * np.sqrt(252)
            if vol > 0.35: return
        ma20 = sum(self.prices[-20:])/20
        ma60 = sum(self.prices[-60:])/60
        self.current_env = "BULL" if ma20 > ma60 else "BEAR"

# ============================================================
# 主回测
# ============================================================
def run_v31():
    t_start = time.time()
    logger.info(f"{'='*55}")
    logger.info(f"🔰 v3.1 — CSI300环境分类器 + 波动率标准化仓位")
    logger.info(f"  分类器: 沪深300真实OHLC, MA三线+ADX(14)+ATR历史分位")
    logger.info(f"  仓位: vol_forecast, 目标风险4%, ATR地板0.005")
    logger.info(f"  TP/SL: 同R1, 池=800只")
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
    
    # 4. 加载CSI300
    csi300 = load_csi300()
    
    # 5. 初始化
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion/config.yaml")))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    
    warmup_dates = [d for d in all_dates if d < "2020-01-01"]
    bt_dates = [d for d in all_dates if "2020-01-01" <= d <= "2026-12-31"]
    
    # 环境分类器（CSI300 + 旧分类器对照）
    csi_env = CSI300EnvClassifier()
    old_env = MAEnvClassifier()
    
    # Phase 1: Warm-up
    logger.info("Phase 1: Warm-up")
    for i, ds in enumerate(warmup_dates):
        # CSI300更新
        if ds in csi300:
            csi_env.update(ds, csi300[ds])
            old_env.update(csi300[ds][3])  # close
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
    old_env_counts = {}
    score_stats = {"used":0,"default":0,"rejected":0,"passed":0}
    stock_history = defaultdict(list)
    
    # Phase 2: Backtest
    logger.info("Phase 2: Backtest v3.1...")
    for di, ds in enumerate(bt_dates):
        trade_date_dt = date.fromisoformat(ds)
        
        # 更新环境分类器
        if ds in csi300:
            csi_env.update(ds, csi300[ds])
            old_env.update(csi300[ds][3])
            csi_env.daily_log.append(csi_env.current_env)  # 记录每日环境
        env = csi_env.current_env
        old_env_counts[old_env.current_env] = old_env_counts.get(old_env.current_env, 0) + 1
        env_counts[env] = env_counts.get(env, 0) + 1
        ep = get_env_params(env)
        ecap = env_cap(env)
        
        day_data = data.get(ds, {})
        if not day_data: continue
        
        # A. 更新技术引擎 + 个股历史
        for stock in top_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=True)
                except: pass
                b = day_data[stock]
                stock_history[stock].append((ds, b.get("high",0), b.get("low",0), b.get("close",0)))
                if len(stock_history[stock]) > 250:
                    stock_history[stock] = stock_history[stock][-250:]
                v = b.get("volume", 0)
                if v > 0:
                    vol_cache.setdefault(stock, []).append(v)
                    if len(vol_cache[stock]) > 50:
                        vol_cache[stock] = vol_cache[stock][-50:]
        
        # B. 信号 + 过滤 + 波动率仓位
        signals = tech_engine.get_confirmed_signals(trade_date_dt)
        for sig in signals:
            if len(positions) >= MAX_POSITIONS: break
            if sig.symbol in positions: continue
            if sig.symbol not in top_stocks: continue
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
                avg_vol = sum(vols[-20:])/20
                ev = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and ev < avg_vol * VOL_MIN_RATIO: continue
            
            # 波动率标准化仓位
            base_signal = signal_to_baseline(sig.signal_subtype)
            stock_atr = calc_stock_atr_ratio(stock_history[sig.symbol])
            stock_atr = max(stock_atr, 0.005)
            
            vol_forecast = base_signal / max(stock_atr, 0.01)
            fund_mult = fund_score / 100.0
            target_risk = 0.04
            
            raw_amount = (cash * target_risk * vol_forecast * fund_mult) / max(stock_atr, 0.01)
            raw_amount = min(raw_amount, cash * 0.40)
            capped_amount = raw_amount * ecap
            
            shares = int(capped_amount / entry_price / 100) * 100
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
                "stock_atr": round(stock_atr, 4),
                "vol_forecast": round(vol_forecast, 2),
            }
            cash -= cost
            sig_stats[sig.signal_subtype] = sig_stats.get(sig.signal_subtype,0)+1
            trade_log.append({"date": ds, "symbol": sig.symbol, "action":"BUY",
                "price": round(entry_price,3), "signal": sig.signal_subtype,
                "fund_score": fund_score, "env": env,
                "atr_ratio": round(stock_atr, 4)})
        
        # C. 止盈止损（同R1）
        to_close, to_reduce = [], []
        for stock, pos in list(positions.items()):
            if stock in day_data and day_data[stock].get("close",0) > 0:
                price = day_data[stock]["close"]; pos["_last_price"] = price
            else: price = pos.get("_last_price", pos["entry_price"])
            pos["peak_price"] = max(pos["peak_price"], price)
            pnl = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
            hold_days = (trade_date_dt - date.fromisoformat(pos["entry_date"])).days
            
            if pnl >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            if hold_days >= 60 and abs(pnl) < 0.05:
                pos["stop_price"] = min(pos["stop_price"], pos["entry_price"]*0.85)
            
            if ep["env_stop"] is not None:
                es = pos["entry_price"] * (1 - ep["env_stop"])
                low = day_data.get(stock, {}).get("low", price)
                if low <= es: to_close.append((stock, "env_stop_loss", price, "close")); continue
            
            low = day_data.get(stock, {}).get("low", price)
            if low <= pos["stop_price"]: to_close.append((stock, "stop_loss", price, "close")); continue
            
            if ep["trail"] is not None and peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= ep["trail"]:
                    nd2 = get_next_trade_date(ds, all_dates)
                    sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                    to_close.append((stock, "trailing_stop", sp, "next_open"))
                    tp_stats["trailing_stop"] = tp_stats.get("trailing_stop",0)+1
                    continue
            
            if pnl >= ep["tp2"]:
                pos["tp2_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock, "tp2", 0.50, sp, "next_open"))
                tp_stats["tp2"] = tp_stats.get("tp2",0)+1; continue
            
            if pnl >= ep["tp1"]:
                pos["tp1_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock, "tp1", 0.30, sp, "next_open"))
                tp_stats["tp1"] = tp_stats.get("tp1",0)+1; continue
        
        for sym, reason, rpct, sp, pt in to_reduce:
            pos = positions[sym]
            rs = int(pos["shares"] * rpct)
            if rs > 0 and pos["shares"] > rs:
                pos["shares"] -= rs; cash += rs * sp
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
                "env": pos.get("env",env)})
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
            logger.info(f"  [{ds}]({di+1}/{len(bt_dates)}) NAV={equity:,.0f} pos={len(positions)} t={len(trade_log)} env={env}(旧={old_env.current_env})")
    
    # === 指标 ===
    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    buys = [t for t in trade_log if t["action"]=="BUY"]
    sells = [t for t in trade_log if t["action"]=="SELL"]
    wins = [t for t in sells if t.get("pnl_pct",0) > 0]
    win_rate = len(wins)/max(len(sells),1)*100
    profit_factor = sum(t["pnl_pct"] for t in wins)/max(sum(abs(t["pnl_pct"]) for t in sells if t.get("pnl_pct",0) <= 0), 1)
    
    pd = __import__('pandas')
    
    logger.info(f"\n{'='*55}")
    logger.info(f"📊 v3.1 全量结果")
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
    logger.info(f"  环境(沪深300): {dict(sorted(env_counts.items()))}")
    logger.info(f"  环境(旧R1): {dict(sorted(old_env_counts.items()))}")
    logger.info(f"  scores: {score_stats}")
    
    # 2024环境对照
    logger.info(f"\n{'='*55}")
    logger.info(f"📋 2024年环境切换对照 (CSI300)")
    logger.info(f"{'='*55}")
    bt_start_idx = len(warmup_dates)
    bt_total = len(warmup_dates) + len(bt_dates)
    
    # 构建2024对照表
    old_cls_reread = MAEnvClassifier()
    new_2024, old_2024, dates_2024 = [], [], []
    for i, ds in enumerate(bt_dates):
        if ds in csi300:
            old_cls_reread.update(csi300[ds][3])
        if "2024-01-01" <= ds <= "2024-12-31":
            dates_2024.append(ds)
            new_2024.append("N/A")  # will fill from csi_env
            old_2024.append(old_cls_reread.current_env)
    
    # 从 CSI300 分类器daily_log提取2024数据
    # daily_log在回测循环中填充，对应 bt_dates
    bt_env = csi_env.daily_log  # 和 bt_dates 等长
    log_map = {bt_dates[i]: bt_env[i] for i in range(min(len(bt_dates), len(bt_env)))}
    for i, ds in enumerate(dates_2024):
        if ds in log_map:
            new_2024[i] = log_map[ds]
    
    new_bull = sum(1 for e in new_2024 if e=="BULL")
    new_bear = sum(1 for e in new_2024 if e=="BEAR")
    new_osc = sum(1 for e in new_2024 if e=="OSCILLATE")
    old_bull = sum(1 for e in old_2024 if e=="BULL")
    old_bear = sum(1 for e in old_2024 if e=="BEAR")
    old_osc = sum(1 for e in old_2024 if e=="OSCILLATE")
    
    # 微盘股危机
    mc = [(d, n, o) for d,n,o in zip(dates_2024,new_2024,old_2024) if "2024-01-01"<=d<="2024-02-08"]
    mc_new_bear = sum(1 for _,n,_ in mc if n=="BEAR")
    mc_old_bear = sum(1 for _,_,o in mc if o=="BEAR")
    
    # 924前后
    sep = [(d, n, o) for d,n,o in zip(dates_2024,new_2024,old_2024) if "2024-09-15"<=d<="2024-10-15"]
    
    logger.info(f"  2024年总交易日: {len(dates_2024)}")
    logger.info(f"  新分类器(CSI300): BULL={new_bull}, OSC={new_osc}, BEAR={new_bear}")
    logger.info(f"  旧分类器(R1): BULL={old_bull}, OSC={old_osc}, BEAR={old_bear}")
    logger.info(f"  微盘股危机(1/2~2/8, {len(mc)}天): 新BEAR={mc_new_bear}, 旧BEAR={mc_old_bear}")
    logger.info(f"  924行情前后切换(9/15~10/15):")
    for d, n, o in sep:
        logger.info(f"    {d}: 新={n} 旧={o}")
    
    # 保存对照表
    comp_df = pd.DataFrame({"date": dates_2024, "new_env": new_2024, "old_env": old_2024})
    comp_df.to_csv(OUTPUT_DIR/"v31_env_comparison_2024.csv", index=False)
    logger.info(f"  ⬆ 环境对照表已保存")
    
    # 保存回测产出
    pd.DataFrame(trade_log).to_csv(OUTPUT_DIR/"v31_trades.csv", index=False)
    pd.DataFrame(equity_curve).to_csv(OUTPUT_DIR/"v31_equity.csv", index=False)
    metrics_save = {k:v for k,v in metrics.items() if isinstance(v,(int,float,str))}
    metrics_save.update({"win_rate": round(win_rate,1), "profit_factor": round(profit_factor,2)})
    with open(OUTPUT_DIR/"v31_summary.json","w") as f:
        json.dump(metrics_save, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - t_start
    logger.info(f"\n⏱️ 总耗时: {elapsed:.0f}s")
    logger.info(f"💾 产出: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_v31()
