#!/usr/bin/env python3
"""
v3.0 — 环境分类器升级 + 波动率标准化仓位
=============================================
改动：
  1. 三层环境分类器：MA20/60/120三线排列 + ADX(14)>22 + ATR比率1.5倍背离
  2. 波动率标准化仓位（vol_forecast = signal / atr_ratio）
  3. 其余同R1：TP/SL参数、800只池子、scores接入
"""

import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, '/home/quant/.openclaw/workspace')

OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/output/v30")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "v30_run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("v30")

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
# 改动1：三层环境分类器
# ============================================================
class AdvancedEnvClassifier:
    """三层环境分类器：MA三线排列 + ADX趋势强度 + ATR波动率阻隔"""
    
    def __init__(self):
        self.closes = []
        self.highs = []
        self.lows = []
        self.current_env = "OSCILLATE"
        # 缓存每日环境标签用于2024年对照分析
        self.daily_log = []
    
    def update(self, high: float, low: float, close: float):
        """用指数OHLC更新分类器"""
        if close <= 0 or high <= 0 or low <= 0:
            if close > 0:
                # 补齐估值
                high = max(high, close)
                low = min(low, close) if low > 0 else close * 0.99
            else:
                return
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.closes) < 120:
            return
        
        # ---- 层1：趋势方向（MA三线排列） ----
        ma20 = np.mean(self.closes[-20:])
        ma60 = np.mean(self.closes[-60:])
        ma120 = np.mean(self.closes[-120:])
        
        bull_trend = (ma20 > ma60) and (ma60 > ma120)
        bear_trend = (ma20 < ma60) and (ma60 < ma120)
        
        # ---- 层2：趋势强度（ADX） ----
        adx = self._calc_adx(14)
        strong_trend = adx > 22
        
        # ---- 层3：波动率阻隔（ATR比率 vs 其60日均线） ----
        atr_ratio = self._calc_atr_ratio(20)
        # 计算ATR比率的60日均值
        atr_ratios = [self._calc_atr_ratio(20, lookback=i) for i in range(min(60, len(self.closes)-20))]
        atr_sma = np.mean(atr_ratios[-60:]) if len(atr_ratios) >= 60 else np.mean(atr_ratios)
        high_vol = atr_ratio > atr_sma * 1.5
        
        # ---- 综合判断 ----
        if bull_trend and strong_trend:
            self.current_env = "BULL"
        elif bear_trend and strong_trend:
            self.current_env = "BEAR"
        elif high_vol:
            self.current_env = "OSCILLATE"
        else:
            # 弱趋势：按方向走
            self.current_env = "BULL" if bull_trend else "BEAR"

    def _calc_adx(self, period=14):
        """计算ADX(14)"""
        n = len(self.closes)
        if n < period * 2 + 2:
            return 0.0
        
        # 取最近 period*2+1 个数据点
        lookback = min(period * 3, n - 1)
        
        tr_list = []
        pdm_list = []
        ndm_list = []
        
        for i in range(n - lookback, n):
            hl = self.highs[i] - self.lows[i]
            hc = abs(self.highs[i] - self.closes[i-1])
            lc = abs(self.lows[i] - self.closes[i-1])
            tr_list.append(max(hl, hc, lc))
            
            up_move = self.highs[i] - self.highs[i-1]
            down_move = self.lows[i-1] - self.lows[i]
            
            if up_move > down_move and up_move > 0:
                pdm_list.append(up_move)
            else:
                pdm_list.append(0.0)
            
            if down_move > up_move and down_move > 0:
                ndm_list.append(down_move)
            else:
                ndm_list.append(0.0)
        
        # 简单EMA近似：取尾均值而非完整EMA
        tr14 = np.mean(tr_list[-period:]) if len(tr_list) >= period else np.mean(tr_list)
        pdm14 = np.mean(pdm_list[-period:]) if len(pdm_list) >= period else np.mean(pdm_list)
        ndm14 = np.mean(ndm_list[-period:]) if len(ndm_list) >= period else np.mean(ndm_list)
        
        di_plus = 100 * pdm14 / tr14 if tr14 > 0 else 0
        di_minus = 100 * ndm14 / tr14 if tr14 > 0 else 0
        
        if di_plus + di_minus == 0:
            return 0.0
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # ADX = EMA of DX, 这里简化为均值
        return dx  # 单点DX, 近似ADX
    
    def _calc_atr_ratio(self, period=20, lookback=0):
        """计算ATR/close比率"""
        n = len(self.closes)
        idx = n - 1 - lookback
        if idx < period + 1:
            return 0.01
        
        tr_vals = []
        for i in range(idx - period, idx + 1):
            hl = self.highs[i] - self.lows[i]
            hc = abs(self.highs[i] - self.closes[i-1])
            lc = abs(self.lows[i] - self.closes[i-1])
            tr_vals.append(max(hl, hc, lc))
        
        atr = np.mean(tr_vals)
        return atr / self.closes[idx] if self.closes[idx] > 0 else 0.01
    
    def get_env_for_date(self):
        """返回当前环境并记录到日志"""
        self.daily_log.append(self.current_env)
        return self.current_env

# ============================================================
# 工具函数
# ============================================================
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

def calc_stock_atr_ratio(stock_data_by_date, lookback=20):
    """
    计算单只股票的ATR/close比率。
    输入：该股票在多日内的一组数据列表 [(date, high, low, close), ...]
    """
    if len(stock_data_by_date) < lookback + 1:
        return 0.02  # 默认
    recent = stock_data_by_date[-(lookback+1):]
    tr_vals = []
    for i in range(1, len(recent)):
        _, h, l, p = recent[i]
        _, _, _, pp = recent[i-1]
        hl = h - l
        hc = abs(h - pp)
        lc = abs(l - pp)
        tr_vals.append(max(hl, hc, lc))
    atr = np.mean(tr_vals)
    _, _, _, close = recent[-1]
    if close <= 0:
        return 0.02
    return atr / close

# ============================================================
# 主回测
# ============================================================
def build_index_prices(data, all_dates, warmup_dates, bt_dates, top_stocks):
    """构建指数 OHLC（全市场均值）"""
    index_high, index_low, index_close = [], [], []
    for ds in warmup_dates + bt_dates:
        day = data.get(ds, {})
        # 用前800只计算
        candidates = [v for sym, v in day.items() if sym in top_stocks]
        if candidates:
            hs = [v.get("high", 0) for v in candidates if v.get("high", 0) > 0]
            ls = [v.get("low", 0) for v in candidates if v.get("low", 0) > 0]
            cs = [v.get("close", 0) for v in candidates if v.get("close", 0) > 0]
            index_high.append(np.mean(hs) if hs else (index_high[-1] if index_high else 10000))
            index_low.append(np.mean(ls) if ls else (index_low[-1] if index_low else 9000))
            index_close.append(np.mean(cs) if cs else (index_close[-1] if index_close else 10000))
        else:
            index_high.append(index_high[-1] if index_high else 10000)
            index_low.append(index_low[-1] if index_low else 9000)
            index_close.append(index_close[-1] if index_close else 10000)
    return index_high, index_low, index_close

def run_v30():
    t_start = time.time()
    logger.info(f"{'='*55}")
    logger.info(f"🔰 v3.0 — 三层环境分类器 + 波动率标准化仓位")
    logger.info(f"  分类器: MA三线+ADX(14)>22+ATR×1.5")
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
    
    # 4. 初始化
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion/config.yaml")))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    
    warmup_dates = [d for d in all_dates if d < "2020-01-01"]
    bt_dates = [d for d in all_dates if "2020-01-01" <= d <= "2026-12-31"]
    
    # 构建指数序列（前800只的OHLC均值）
    index_high, index_low, index_close = build_index_prices(
        data, all_dates, warmup_dates, bt_dates, top_stocks
    )
    
    # 高级环境分类器
    env_cls = AdvancedEnvClassifier()
    
    # Phase 1: Warm-up
    logger.info("Phase 1: Warm-up")
    total_idx_len = len(warmup_dates) + len(bt_dates)
    for i, ds in enumerate(warmup_dates):
        if i < total_idx_len:
            env_cls.update(index_high[i], index_low[i], index_close[i])
        day_data = data.get(ds, {})
        for stock in top_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=False)
                except: pass
        if (i+1) % 500 == 0:
            logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
    logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)}只")
    
    # === 状态 ===
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
    # 个股历变数据（用于ATR计算）
    stock_history = defaultdict(list)  # stock -> [(date, high, low, close), ...]
    
    # === 环境参数（同R1） ===
    def get_env_params(env: str):
        if env == "BULL":
            return {"tp1": 0.12, "tp2": 0.18, "trail": 0.88, "env_stop": None}
        elif env == "OSCILLATE":
            return {"tp1": 0.06, "tp2": 0.10, "trail": 0.94, "env_stop": None}
        else:
            return {"tp1": 0.03, "tp2": 0.05, "trail": None, "env_stop": 0.03}
    
    def signal_to_baseline(subtype: str) -> float:
        return {"third_point_buy": 0.6, "hard_divergence": 1.0}.get(subtype, 0.5)
    
    def env_cap(env: str) -> float:
        return {"BULL": 1.0, "OSCILLATE": 0.7, "BEAR": 0.5}.get(env, 1.0)
    
    # Phase 2: Backtest
    logger.info("Phase 2: Backtest v3.0...")
    for di, ds in enumerate(bt_dates):
        trade_date_dt = date.fromisoformat(ds)
        idx = len(warmup_dates) + di
        if idx < total_idx_len:
            env_cls.update(index_high[idx], index_low[idx], index_close[idx])
        env = env_cls.get_env_for_date()
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
        
        # B. 信号 + 过滤 + 新仓位公式
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
            
            # scores过滤
            fund_score = get_fund_score(sig.symbol, ds, scores_cache)
            if fund_score is None: fund_score = 50.0; score_stats["default"] += 1
            else: score_stats["used"] += 1
            if fund_score < FUND_SCORE_MIN: score_stats["rejected"] += 1; continue
            score_stats["passed"] += 1
            
            # 基础成交量过滤
            vols = vol_cache.get(sig.symbol, [])
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:])/20
                ev = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and ev < avg_vol * VOL_MIN_RATIO: continue
            
            # ============== 波动率标准化仓位（改动2） ==============
            base_signal = signal_to_baseline(sig.signal_subtype)
            
            # 个股ATR/close比率（需至少20天数据）
            stock_atr_ratio = calc_stock_atr_ratio(stock_history[sig.symbol])
            stock_atr_ratio = max(stock_atr_ratio, 0.005)  # ATR地板
            
            # vol_forecast = base_signal / atr_ratio
            vol_forecast = base_signal / max(stock_atr_ratio, 0.01)
            
            # 基本面乘数
            fund_mult = fund_score / 100.0
            
            # 目标仓位
            target_position_risk = 0.04  # 单只目标风险4%
            raw_amount = (cash * target_position_risk * vol_forecast * fund_mult) / max(stock_atr_ratio, 0.01)
            raw_amount = min(raw_amount, cash * 0.40)  # 单只上限40%
            
            # 环境上限帽
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
                "fund_score": fund_score,
                "env": env,
                "stock_atr_ratio": round(stock_atr_ratio, 4),
                "vol_forecast": round(vol_forecast, 2),
            }
            cash -= cost
            sig_stats[sig.signal_subtype] = sig_stats.get(sig.signal_subtype,0)+1
            trade_log.append({"date": ds, "symbol": sig.symbol, "action":"BUY",
                "price": round(entry_price,3), "signal": sig.signal_subtype,
                "fund_score": fund_score, "env": env,
                "atr_ratio": round(stock_atr_ratio, 4)})
        
        # C. 止盈止损（同R1）
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
            
            if ep["env_stop"] is not None:
                env_stop_price = pos["entry_price"] * (1 - ep["env_stop"])
                low = day_data.get(stock, {}).get("low", price)
                if low <= env_stop_price:
                    to_close.append((stock, "env_stop_loss", price, "close")); continue
            
            low = day_data.get(stock, {}).get("low", price)
            if low <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price, "close")); continue
            
            if ep["trail"] is not None and peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= ep["trail"]:
                    nd2 = get_next_trade_date(ds, all_dates)
                    sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                    to_close.append((stock, "trailing_stop", sp, "next_open"))
                    tp_stats["trailing_stop"] = tp_stats.get("trailing_stop",0)+1
                    continue
            
            if pnl_pct >= ep["tp2"]:
                pos["tp2_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock, "tp2_resonance", 0.50, sp, "next_open"))
                tp_stats["tp2"] = tp_stats.get("tp2",0)+1
                continue
            
            if pnl_pct >= ep["tp1"]:
                pos["tp1_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                sp = data[nd2][stock]["open"] if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock, "tp1_resonance", 0.30, sp, "next_open"))
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
            logger.info(f"  [{ds}]({di+1}/{len(bt_dates)}) NAV={equity:,.0f} pos={len(positions)} t={len(trade_log)} env={env}")
    
    # === 指标 ===
    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    buys = [t for t in trade_log if t["action"]=="BUY"]
    sells = [t for t in trade_log if t["action"]=="SELL"]
    wins = [t for t in sells if t.get("pnl_pct",0) > 0]
    win_rate = len(wins) / max(len(sells),1) * 100
    profit_factor = sum(t["pnl_pct"] for t in wins) / max(sum(abs(t["pnl_pct"]) for t in sells if t.get("pnl_pct",0) <= 0), 1)
    
    logger.info(f"\n{'='*55}")
    logger.info(f"📊 v3.0 全量结果")
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
    
    # 环境切换对照表（2024年）
    logger.info(f"\n{'='*55}")
    logger.info(f"📋 2024年环境切换对照")
    logger.info(f"{'='*55}")
    # 从env_cls.daily_log提取2024年标签
    # daily_log对应 warmup_dates + bt_dates 共 total_idx_len 条记录
    # 需要找到2024年的索引范围
    all_days = warmup_dates + bt_dates
    old_classifier = MAEnvClassifier()
    old_daily = []
    
    # 重新跑旧分类器用于对照
    for i, ds in enumerate(all_days):
        # 旧分类器只需要close
        old_classifier.update(index_close[i])
    # 旧分类器没记录历史，这里在回测循环中记录
    # 直接从回测结果看：bt_dates从2020-01-01开始
    # 我们需要2024年的索引
    
    pd = __import__('pandas')
    
    # 记录新旧分类器在2024年的比对
    # 新分类器每日环境从 env_cls.daily_log 获取
    # 旧分类器重新构造
    old_cls2 = MAEnvClassifier()
    new_envs = env_cls.daily_log  # 对应 warmup_dates + bt_dates
    new_2024 = []
    old_2024 = []
    dates_2024 = []
    
    for i, ds in enumerate(all_days):
        old_cls2.update(index_close[i])
        if ds >= "2024-01-01" and ds <= "2024-12-31":
            dates_2024.append(ds)
            new_2024.append(new_envs[i] if i < len(new_envs) else "N/A")
            old_2024.append(old_cls2.current_env)
    
    # 统计
    new_bear = sum(1 for e in new_2024 if e == "BEAR")
    old_bear = sum(1 for e in old_2024 if e == "BEAR")
    new_bull = sum(1 for e in new_2024 if e == "BULL")
    old_bull = sum(1 for e in old_2024 if e == "BULL")
    new_osc = sum(1 for e in new_2024 if e == "OSCILLATE")
    old_osc = sum(1 for e in old_2024 if e == "OSCILLATE")
    
    # 微盘股危机期(2024-01-02 ~ 2024-02-08)
    microcap_dates = [d for d in dates_2024 if d >= "2024-01-01" and d <= "2024-02-08"]
    microcap_new_bear = sum(1 for i,d in enumerate(dates_2024) 
                            if d >= "2024-01-01" and d <= "2024-02-08" and new_2024[i]=="BEAR")
    microcap_old_bear = sum(1 for i,d in enumerate(dates_2024)
                            if d >= "2024-01-01" and d <= "2024-02-08" and old_2024[i]=="BEAR")
    
    # 924行情前后(2024-09-01 ~ 2024-10-31)
    sep_dates = [(d, new_2024[i], old_2024[i]) for i,d in enumerate(dates_2024)
                 if d >= "2024-09-01" and d <= "2024-10-31"]
    
    logger.info(f"  2024年总交易日: {len(dates_2024)}")
    logger.info(f"  新分类器: BULL={new_bull}, OSCILLATE={new_osc}, BEAR={new_bear}")
    logger.info(f"  旧分类器: BULL={old_bull}, OSCILLATE={old_osc}, BEAR={old_bear}")
    logger.info(f"  微盘股危机(1/2~2/8): 新BEAR={microcap_new_bear}/{len(microcap_dates)}, 旧BEAR={microcap_old_bear}/{len(microcap_dates)}")
    logger.info(f"  924行情前后切换:")
    for sd in sep_dates:
        logger.info(f"    {sd[0]}: 新={sd[1]} 旧={sd[2]}")
    
    # 保存对照表
    comp_df = pd.DataFrame({
        "date": dates_2024,
        "new_env": new_2024,
        "old_env": old_2024,
    })
    comp_df.to_csv(OUTPUT_DIR / "v30_env_comparison_2024.csv", index=False)
    logger.info(f"  ⬆ 环境对照表已保存")
    
    # 保存产出
    pd.DataFrame(trade_log).to_csv(OUTPUT_DIR/"v30_trades.csv", index=False)
    pd.DataFrame(equity_curve).to_csv(OUTPUT_DIR/"v30_equity.csv", index=False)
    metrics_save = {k:v for k,v in metrics.items() if isinstance(v,(int,float,str))}
    metrics_save.update({"win_rate": round(win_rate,1), "profit_factor": round(profit_factor,2)})
    with open(OUTPUT_DIR/"v30_summary.json","w") as f:
        json.dump(metrics_save, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - t_start
    logger.info(f"\n⏱️ 总耗时: {elapsed:.0f}s")
    logger.info(f"💾 产出: {OUTPUT_DIR}")

# 引入旧分类器用于对照
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

if __name__ == "__main__":
    run_v30()
