#!/usr/bin/env python3
"""
v6ab Baseline — 完整记录版回测
=================================
基于 v6ab (动量排名过滤 15% + min_hold)，增加：
- 逐笔 trade log（买入原因、卖出原因完整字段）
- 逐年收益分析
- 完整策略报告（参数/仓位逻辑/止损止盈/环境分类器）

回测: 2010-07-01 → 2026-04-30
"""

import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

WORKSPACE = "/home/quant/.openclaw/workspace"
sys.path.insert(0, WORKSPACE)

OUT_DIR = Path(WORKSPACE) / "output" / "v6ab_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(),
        logging.FileHandler(str(OUT_DIR / "run.log"), mode="w", encoding="utf-8")])
log = logging.getLogger("v6ab_bl")

import pickle, pandas as pd
from scripts.task3_v21_backtest import (
    load_data, load_scores_parquet, get_next_trade_date,
    get_fund_score_v21 as get_fund_score, calc_metrics,
    INITIAL_CAPITAL, MAX_POSITIONS, COOLING_PERIOD_DAYS,
    TP_BREAKEVEN_LOCK, VOL_MIN_RATIO,
)

# ═══════════════════════════════════════════════════════════════
# 第一部分：完整参数定义（供报告使用）
# ═══════════════════════════════════════════════════════════════

# ---- 成本参数 ----
SLIPPAGE_BUY = 1.001       # 买入滑点 0.1%
SLIPPAGE_SELL = 0.999      # 卖出滑点 0.1%
VOL_CAP_RATIO = 0.05       # 单笔成交量上限（成交量×5%）
FIXED_STOP_LOSS = 0.08     # 固定止损 -8%
FUND_SCORE_MIN = 30.0      # 基本面评分最低门槛

# ---- 仓位控制参数 ----
INITIAL_CAP = INITIAL_CAPITAL       # 初始资金 1,000,000
MAX_POS = MAX_POSITIONS             # 最大持仓 5 只
COOLING_DAYS = COOLING_PERIOD_DAYS # 止损冷却期 10 天
POSITION_SIZE_PCT = 0.04           # 每笔风险预算比例 4%
MAX_SINGLE_POS = 0.40              # 单票最大仓位 40%

# ---- 动量过滤参数 ----
MOMENTUM_WINDOW = 20        # 动量计算窗口（20个交易日）
MOMENTUM_PERCENTILE = 15    # 拒绝底部 15%
MOMENTUM_UPDATE_FREQ = 20   # 每 20 天更新排名

# ---- 信号权重 ----
SIGNAL_WEIGHTS = {
    "third_point_buy": 0.6,    # 第三类买点
    "hard_divergence": 1.0,    # 盘整背驰
    "default": 0.5,            # 其他信号
}

# ---- 止盈止损参数 ----
# 按 CSI300 宏观看盘环境分类
ENV_PARAMS = {
    "BULL": {
        "tp1": 0.12,      # 第一止盈 12%
        "tp2": 0.18,      # 第二止盈 18%
        "trail": 0.88,    # 移动止盈回撤到峰值的 88%
        "env_stop": None,  # 无环境止损
    },
    "OSCILLATE": {
        "tp1": 0.06,      # 第一止盈 6%
        "tp2": 0.10,      # 第二止盈 10%
        "trail": 0.94,    # 移动止盈回撤到峰值的 94%
        "env_stop": None,  # 无环境止损
    },
    "BEAR": {
        "tp1": 0.03,      # 第一止盈 3%
        "tp2": 0.05,      # 第二止盈 5%
        "trail": None,     # 无移动止盈
        "env_stop": 0.03,  # 环境止损 -3%
    },
}

# 盈亏平衡锁
BREAKEVEN_LOCK = TP_BREAKEVEN_LOCK   # +5% 后止损移至成本价
# 持仓60天紧缩规则
HOLD_DAYS_TIGHTEN = 60               # 持仓 60 天
PNL_TIGHTEN_THRESHOLD = 0.05         # PnL < 5% 时收紧止损
STOP_TIGHTEN_RATIO = 0.85            # 止损收紧到入场价的 85%

# ---- 成交量过滤 ----
VOLUME_FILTER_RATIO = VOL_MIN_RATIO   # 当日量 > 20日均量 × 0.8

# ═══════════════════════════════════════════════════════════════
# 第二部分：环境分类器（CSI300 宏观看盘）
# ═══════════════════════════════════════════════════════════════

def load_csi300():
    """加载沪深300日线数据"""
    df = pd.read_csv(str(Path(WORKSPACE) / "data" / "index_sh000300.csv"))
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return {r['date']: (r['open'], r['high'], r['low'], r['close']) for _, r in df.iterrows()}

class CSI300EnvClassifier:
    """
    CSI300 大盘环境分类器
    
    逻辑:
    1. 计算 MA20, MA60, MA120
    2. BULL: MA20 > MA60 > MA120 (多头排列) 且 ADX>22 (趋势明确) 且非高波动
    3. BEAR: MA20 < MA60 < MA120 (空头排列) 且 ADX>22 (趋势明确) 且非高波动
    4. OSCILLATE: 震荡市 / 高波动期 / 趋势不明
    
    ADX 用于区分趋势市 vs 震荡市
    波动率带用于在高波动期强制回到 OSCILLATE
    """
    def __init__(self):
        self.dates, self.closes, self.highs, self.lows = [], [], [], []
        self.current_env = "OSCILLATE"
        self.daily_log = []
    
    def update(self, ds, ohlc):
        o, h, l, c = ohlc
        if c <= 0:
            return
        self.dates.append(ds)
        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(l)
        if len(self.closes) < 120:
            return
        ma20 = np.mean(self.closes[-20:])
        ma60 = np.mean(self.closes[-60:])
        ma120 = np.mean(self.closes[-120:])
        bull = ma20 > ma60 > ma120
        bear = ma20 < ma60 < ma120
        adx = self._adx(14)
        trend_clear = adx > 22
        atr_ratio = self._atr(20) / c
        
        hv = False
        if len(self.closes) >= 200:
            ratios = []
            for i in range(20, min(120, len(self.closes)-1)+20):
                tr = max(self.closes[i] - self.closes[i-1], abs(self.closes[i] - self.closes[i-1]))
                ratios.append(tr / self.closes[i-1])
            if ratios:
                hv = atr_ratio > np.percentile(ratios, 85)
        
        if bull and trend_clear and not hv:
            self.current_env = "BULL"
        elif bear and trend_clear and not hv:
            self.current_env = "BEAR"
        elif hv:
            self.current_env = "OSCILLATE"
        else:
            self.current_env = "BULL" if bull else "BEAR"
    
    def _adx(self, p=14):
        n = len(self.closes)
        if n < p * 2 + 2:
            return 0.0
        lb = min(p * 3, n - 1)
        tr, pd, nd = [], [], []
        for i in range(n - lb, n):
            hl = self.highs[i] - self.lows[i]
            hc = abs(self.highs[i] - self.closes[i-1])
            lc = abs(self.lows[i] - self.closes[i-1])
            tr.append(max(hl, hc, lc))
            up = self.highs[i] - self.highs[i-1]
            down = self.lows[i-1] - self.lows[i]
            pd.append(up if up > down and up > 0 else 0.0)
            nd.append(down if down > up and down > 0 else 0.0)
        tp = np.mean(tr[-p:]) if len(tr) >= p else np.mean(tr)
        dp = 100 * np.mean(pd[-p:]) / tp if tp > 0 else 0
        dn = 100 * np.mean(nd[-p:]) / tp if tp > 0 else 0
        return 100 * abs(dp - dn) / (dp + dn) if dp + dn > 0 else 0.0
    
    def _atr(self, p=20):
        if len(self.closes) < p + 1:
            return self.closes[-1] * 0.02 if self.closes else 0
        tr = [max(self.closes[i] - self.closes[i-1], abs(self.closes[i] - self.closes[i-1])) for i in range(-p, 0)]
        return np.mean(tr) if tr else self.closes[-1] * 0.02

def get_env_params(env):
    """获取环境对应的止盈止损参数"""
    return ENV_PARAMS.get(env, ENV_PARAMS["OSCILLATE"])

# ═══════════════════════════════════════════════════════════════
# 第三部分：工具函数
# ═══════════════════════════════════════════════════════════════

def calc_stock_atr(sd):
    """计算个股 ATR (21天)"""
    if len(sd) < 21:
        return 0.02
    r = sd[-21:]
    tr = [max(r[i][1]-r[i][2], abs(r[i][1]-r[i-1][3]), abs(r[i][2]-r[i-1][3])) for i in range(1, len(r))]
    return np.mean(tr) / r[-1][3] if r[-1][3] > 0 else 0.02

def compute_momentum_ranking(daily_close, top_stocks):
    """计算全市场动量排名（每只股票的20日涨跌幅排名）"""
    momentum = {}
    for stock in top_stocks:
        prices = daily_close.get(stock, [])
        if len(prices) >= MOMENTUM_WINDOW + 1:
            ret = (prices[-1] - prices[-MOMENTUM_WINDOW - 1]) / max(prices[-MOMENTUM_WINDOW - 1], 0.01)
            momentum[stock] = ret
        else:
            momentum[stock] = 0.0
    if not momentum:
        return {}, {}
    values = sorted(momentum.values())
    n = len(values)
    percentile = {}
    for stock, val in momentum.items():
        rank = sum(1 for v in values if v < val)
        percentile[stock] = (rank / max(n, 1)) * 100
    return percentile, momentum

# ═══════════════════════════════════════════════════════════════
# 第四部分：主回测
# ═══════════════════════════════════════════════════════════════

def run_baseline():
    t0 = time.time()
    log.info("=" * 55)
    log.info("v6ab Baseline — 完整记录版回测")
    log.info(f"  动量: MOMENTUM_PERCENTILE={MOMENTUM_PERCENTILE}%")
    log.info(f"  最大持仓: {MAX_POS}")
    log.info(f"  回测: 2010-07-01 ~ 2026-04-30")
    log.info("=" * 55)
    
    # ---- 加载数据 (带缓存) ----
    DATA_CACHE = "/tmp/daily_data_cache.pkl"
    if os.path.exists(DATA_CACHE):
        log.info("加载缓存数据...")
        with open(DATA_CACHE, 'rb') as f:
            data = pickle.load(f)
    else:
        log.info("首次加载数据...")
        data = load_data(max_stocks=0)
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump(data, f)
    all_dates = sorted(data.keys())
    log.info(f"  数据: {all_dates[0]} ~ {all_dates[-1]}, {len(all_dates)} 天")
    
    scores_cache = load_scores_parquet()
    
    # ---- 股票池构建 (成交量前800) ----
    vol_dates = all_dates[:min(500, len(all_dates))]
    vr = defaultdict(list)
    for ds in vol_dates:
        for sym, bar in data.get(ds, {}).items():
            if bar.get("volume", 0) > 0:
                vr[sym].append(bar["volume"])
    avg_vols = {s: np.mean(v[-250:]) for s, v in vr.items() if len(v) >= 20}
    top_stocks = set(sorted(avg_vols, key=avg_vols.get, reverse=True)[:800])
    log.info(f"  股票池: {len(top_stocks)} 只")
    
    # ---- 回测区间划分 ----
    backtest_start = "2010-07-01"
    end_date = "2026-04-30"
    warmup_dates = [d for d in all_dates if d < backtest_start]
    bt_dates = [d for d in all_dates if backtest_start <= d <= end_date]
    log.info(f"  预热: {len(warmup_dates)} 天 | 回测: {len(bt_dates)} 天")
    
    # ---- 初始化引擎 ----
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path(WORKSPACE) / "chanfund_fusion" / "config.yaml"))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    
    # ---- CSI300 环境 ----
    csi300 = load_csi300()
    csi_env = CSI300EnvClassifier()
    
    # ---- 预热 ----
    log.info("预热中...")
    for i, ds in enumerate(warmup_dates):
        if ds in csi300:
            csi_env.update(ds, csi300[ds])
        for stock in top_stocks:
            if stock in data.get(ds, {}):
                try:
                    tech_engine.update(stock, data[ds][stock], False)
                except:
                    pass
        if (i + 1) % 500 == 0:
            log.info(f"  {ds} ({i+1}/{len(warmup_dates)})")
    log.info(f"✅ 预热完成 ({time.time()-t0:.0f}s)")
    
    # ---- 回测主循环 ----
    cash = INITIAL_CAP
    peak_equity = INITIAL_CAP
    positions = {}
    equity_curve = []
    cooling = {}
    trade_log = []  # 逐笔交易记录
    sig_stats = defaultdict(int)
    exit_stats = defaultdict(int)
    vol_cache = defaultdict(list)
    stock_history = defaultdict(list)
    daily_close = defaultdict(list)
    momentum_percentile = {}
    momentum_last_update = -1
    momentum_passed = 0
    momentum_rejected = 0
    env_counts = defaultdict(int)
    score_stats = {"used": 0, "default": 0, "rejected": 0, "passed": 0}
    vol_filter_rejected = 0
    
    for di, ds in enumerate(bt_dates):
        tdt = date.fromisoformat(ds)
        
        # 更新 CSI300 环境
        if ds in csi300:
            csi_env.update(ds, csi300[ds])
            csi_env.daily_log.append(csi_env.current_env)
        env = csi_env.current_env
        env_counts[env] += 1
        ep = get_env_params(env)
        
        day_data = data.get(ds, {})
        if not day_data:
            continue
        
        # 更新日线数据
        for stock in top_stocks:
            if stock in day_data:
                b = day_data[stock]
                daily_close[stock].append(b.get("close", 0))
                if len(daily_close[stock]) > 250:
                    daily_close[stock] = daily_close[stock][-250:]
        
        # 更新动量排名（每20天）
        if di - momentum_last_update >= MOMENTUM_UPDATE_FREQ:
            momentum_percentile, _ = compute_momentum_ranking(daily_close, top_stocks)
            momentum_last_update = di
        
        # 更新缠论引擎和历史窗口
        for stock in top_stocks:
            if stock in day_data:
                b = day_data[stock]
                try:
                    tech_engine.update(stock, b, True)
                except:
                    pass
                stock_history[stock].append((ds, b.get("high", 0), b.get("low", 0), b.get("close", 0)))
                if len(stock_history[stock]) > 250:
                    stock_history[stock] = stock_history[stock][-250:]
                if b.get("volume", 0) > 0:
                    vol_cache[stock].append(b["volume"])
                    if len(vol_cache[stock]) > 50:
                        vol_cache[stock] = vol_cache[stock][-50:]
        
        # ── 买入处理 ──
        signals = tech_engine.get_confirmed_signals(tdt)
        for sig in signals:
            # 仓位上限和重复检查
            if len(positions) >= MAX_POS:
                break
            if sig.symbol in positions or sig.symbol not in top_stocks:
                continue
            if sig.symbol in cooling and tdt <= cooling[sig.symbol]:
                continue
            if sig.symbol not in day_data:
                continue
            
            # 次日开盘价（使用滑点）
            nd = get_next_trade_date(ds, all_dates)
            if nd is None or sig.symbol not in data.get(nd, {}):
                continue
            entry_price = data[nd][sig.symbol]["open"]
            if entry_price <= 0:
                continue
            
            # 涨跌停检查
            pc = data.get(ds, {}).get(sig.symbol, {}).get("close", 0)
            if pc > 0 and (entry_price >= round(pc * 1.10, 2) or entry_price <= round(pc * 0.90, 2)):
                continue
            
            exec_price = entry_price * SLIPPAGE_BUY
            
            # 基本面评分
            fund_score = get_fund_score(sig.symbol, ds, scores_cache)
            if fund_score is None:
                fund_score = 50.0
                score_stats["default"] += 1
            else:
                score_stats["used"] += 1
            if fund_score < FUND_SCORE_MIN:
                score_stats["rejected"] += 1
                continue
            score_stats["passed"] += 1
            
            # 动量排名过滤
            pct = momentum_percentile.get(sig.symbol, 50.0)
            if pct < MOMENTUM_PERCENTILE:
                momentum_rejected += 1
                continue
            momentum_passed += 1
            
            # 成交量过滤
            vols = vol_cache.get(sig.symbol, [])
            if len(vols) >= 20:
                avg_v = sum(vols[-20:]) / 20
                if avg_v > 0 and day_data[sig.symbol].get("volume", 0) < avg_v * VOLUME_FILTER_RATIO:
                    vol_filter_rejected += 1
                    continue
            
            # ── 仓位计算 ──
            # 信号权重: third_point_buy=0.6, hard_divergence=1.0, other=0.5
            bs = SIGNAL_WEIGHTS.get(sig.signal_subtype, SIGNAL_WEIGHTS["default"])
            
            # 波动率调整
            stock_atr = max(calc_stock_atr(stock_history[sig.symbol]), 0.005)
            
            # 波动因子 = 信号权重 / 波动率
            vol_factor = bs / max(stock_atr, 0.01)
            
            # 基本面因子 = fund_score / 100
            fund_factor = fund_score / 100.0
            
            # 风险预算 = 现金 × 4% × 波动因子 × 基本面因子 / 波动率
            raw_amt = (cash * POSITION_SIZE_PCT * vol_factor * fund_factor) / max(stock_atr, 0.01)
            raw_amt = min(raw_amt, cash * MAX_SINGLE_POS)  # 单票上限 40%
            
            # 成交量约束
            daily_vol = data.get(nd, {}).get(sig.symbol, {}).get("volume", 0)
            max_shares_by_vol = int(daily_vol * VOL_CAP_RATIO / 100) * 100 if daily_vol > 0 else 9999999
            
            shares = int(raw_amt / exec_price / 100) * 100
            shares = min(shares, max_shares_by_vol)
            if shares <= 0:
                continue
            
            cost = shares * exec_price
            if cost > cash or cost <= 0:
                continue
            
            # 开仓
            positions[sig.symbol] = {
                "entry_price": entry_price,
                "exec_price": exec_price,
                "shares": shares,
                "entry_date": ds,
                "buy_exec_date": nd,
                "stop_price": entry_price * (1 - FIXED_STOP_LOSS),  # 初始止损
                "peak_price": entry_price,
                "signal_type": sig.signal_subtype,
                "tp1_done": False,
                "tp2_done": False,
                "fund_score": round(fund_score, 1),
                "env": env,
                "momentum_pct": round(pct, 1),
            }
            cash -= cost
            sig_stats[sig.signal_subtype] += 1
            
            # 买入记录
            trade_log.append({
                "date": ds,
                "symbol": sig.symbol,
                "action": "BUY",
                "price": round(exec_price, 3),
                "shares": shares,
                "signal": sig.signal_subtype,
                "fund_score": round(fund_score, 1),
                "env": env,
                "momentum_pct": round(pct, 1),
                "pnl_pct": None,
                "reason": None,
                "hold_days": None,
            })
        
        # ── 卖出处理 ──
        to_close = []
        to_reduce = []
        
        for stock, pos in list(positions.items()):
            if stock not in day_data:
                continue
            raw_price = day_data[stock].get("close", 0)
            if raw_price <= 0:
                raw_price = pos.get("_last_price", pos["entry_price"])
            pos["_last_price"] = raw_price
            price = raw_price * SLIPPAGE_SELL
            
            pos["peak_price"] = max(pos["peak_price"], raw_price)
            pnl = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
            hold = (tdt - date.fromisoformat(pos["entry_date"])).days
            
            # 盈亏平衡锁: +5% 后止损移到成本价
            if pnl >= BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            
            # 持仓60天无盈利: 收紧止损到 85%
            if hold >= HOLD_DAYS_TIGHTEN and abs(pnl) < PNL_TIGHTEN_THRESHOLD:
                pos["stop_price"] = min(pos["stop_price"], pos["entry_price"] * STOP_TIGHTEN_RATIO)
            
            hit = False
            
            # 1. 环境止损 (envy_stop)
            if ep["env_stop"] is not None:
                es_price = pos["entry_price"] * (1 - ep["env_stop"])
                if day_data.get(stock, {}).get("low", raw_price) <= es_price:
                    to_close.append((stock, "env_stop_loss", price, raw_price))
                    hit = True
            
            # 2. 硬止损 (stop_loss)
            if not hit and day_data.get(stock, {}).get("low", raw_price) <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price, raw_price))
                hit = True
            
            # 3. 移动止盈 (trailing_stop)
            if not hit and ep["trail"] is not None and peak_pnl >= BREAKEVEN_LOCK:
                if raw_price / pos["peak_price"] <= ep["trail"]:
                    nd2 = get_next_trade_date(ds, all_dates)
                    if nd2 and stock in data.get(nd2, {}):
                        sp = data[nd2][stock]["open"] * SLIPPAGE_SELL
                    else:
                        sp = price
                    to_close.append((stock, "trailing_stop", sp, raw_price))
                    hit = True
            
            # 4. 第二止盈 (tp2)
            if not hit and pnl >= ep["tp2"]:
                pos["tp2_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                if nd2 and stock in data.get(nd2, {}):
                    sp = data[nd2][stock]["open"] * SLIPPAGE_SELL
                else:
                    sp = price
                to_reduce.append((stock, "tp2", 0.50, sp, raw_price))
                hit = True
            
            # 5. 第一止盈 (tp1)
            if not hit and pnl >= ep["tp1"]:
                pos["tp1_done"] = True
                nd2 = get_next_trade_date(ds, all_dates)
                if nd2 and stock in data.get(nd2, {}):
                    sp = data[nd2][stock]["open"] * SLIPPAGE_SELL
                else:
                    sp = price
                to_reduce.append((stock, "tp1", 0.30, sp, raw_price))
        
        # 执行减仓
        for sym, reason, rpct, sp, rp in to_reduce:
            p = positions.get(sym)
            if not p:
                continue
            rs = int(p["shares"] * rpct)
            if rs <= 0 or p["shares"] <= rs:
                continue
            # 减仓成本 = 份额 × 滑点后价格 × (1+佣金)
            reduction_amount = rs * sp
            cash += reduction_amount
            p["shares"] -= rs
            exit_stats[f"reduce_{reason}"] += 1
        
        # 执行清仓
        for sym, reason, price, rp in to_close:
            p = positions.pop(sym, None)
            if not p:
                continue
            pnl_pct = (price - p["entry_price"]) / p["entry_price"] * 100
            cash += p["shares"] * price
            exit_stats[reason] += 1
            
            trade_log.append({
                "date": ds,
                "symbol": sym,
                "action": "SELL",
                "price": round(price, 3),
                "shares": p["shares"],
                "signal": p["signal_type"],
                "fund_score": p["fund_score"],
                "env": p.get("env", env),
                "momentum_pct": p.get("momentum_pct"),
                "pnl_pct": round(pnl_pct, 2),
                "reason": reason,
                "hold_days": hold,
            })
            
            # 止损后冷却期
            if reason == "stop_loss":
                cooling[sym] = tdt + timedelta(days=COOLING_DAYS)
        
        # 计算净值
        net_asset = cash + sum(
            p["shares"] * day_data.get(sym, {}).get("close", p["entry_price"])
            for sym, p in positions.items()
        )
        peak_equity = max(peak_equity, net_asset)
        dd = (net_asset / peak_equity - 1) * 100
        equity_curve.append({
            "date": ds,
            "nav": round(net_asset, 2),
            "dd": round(dd, 2),
        })
        
        if (di + 1) % 500 == 0 or di == len(bt_dates) - 1:
            log.info(
                f"  [{ds}]({di+1}/{len(bt_dates)}) "
                f"NAV={net_asset:,.0f} pos={len(positions)} "
                f"trades={len(trade_log)} env={env}"
            )
    
    # ══════════════════════════════════════════════════════════
    # 第五部分：统计分析
    # ══════════════════════════════════════════════════════════
    log.info("计算统计指标...")
    eq_df = pd.DataFrame(equity_curve)
    eq_df.to_csv(str(OUT_DIR / "equity.csv"), index=False)
    
    trades_df = pd.DataFrame(trade_log)
    trades_df.to_csv(str(OUT_DIR / "trades.csv"), index=False)
    
    buys = trades_df[trades_df["action"] == "BUY"]
    sells = trades_df[trades_df["action"] == "SELL"]
    n_buys = len(buys)
    n_sells = len(sells)
    
    rets = eq_df["nav"].pct_change().dropna()
    sharpe = np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(252)
    md = eq_df["dd"].min()
    tot = (eq_df.iloc[-1]["nav"] / INITIAL_CAP - 1) * 100
    n_years = len(bt_dates) / 252
    ann_ret = ((1 + tot / 100) ** (1 / max(n_years, 1)) - 1) * 100
    
    win_sells = sells[sells["pnl_pct"] > 0] if len(sells) > 0 else sells
    wr = len(win_sells) / max(len(sells), 1) * 100
    avg_win = win_sells["pnl_pct"].mean() if len(win_sells) > 0 else 0
    avg_loss = sells[sells["pnl_pct"] <= 0]["pnl_pct"].mean() if len(sells[sells["pnl_pct"] <= 0]) > 0 else 0
    wlr = abs(avg_win / max(avg_loss, 0.01)) if avg_loss < 0 else 0
    
    # 逐年收益
    yearly = {}
    for year in range(2010, 2027):
        y = str(year)
        yeq = eq_df[eq_df["date"].str.startswith(y)]
        if len(yeq) < 10:
            continue
        yret = (yeq.iloc[-1]["nav"] / yeq.iloc[0]["nav"] - 1) * 100
        ysel = sells[sells["date"].str.startswith(y)]
        ywin = ysel[ysel["pnl_pct"] > 0]
        yearly[y] = {
            "return_pct": round(yret, 2),
            "n_trades": len(ysel),
            "win_rate": round(len(ywin) / max(len(ysel), 1) * 100, 1),
            "max_dd": round(yeq["dd"].min(), 2),
        }
    
    # 退出原因统计
    exit_analysis = {}
    for reason in sells["reason"].unique():
        sub = sells[sells["reason"] == reason]
        pnls = sub["pnl_pct"]
        exit_analysis[reason] = {
            "count": len(sub),
            "avg_pnl": round(pnls.mean(), 2),
            "avg_hold_days": round(sub["hold_days"].mean(), 1),
            "win_rate": round((pnls > 0).sum() / max(len(sub), 1) * 100, 1),
        }
    
    # 信号类型分布
    signal_dist = dict(sig_stats)
    
    # 环境分布
    env_dist = dict(env_counts)
    
    # 保存分析 JSON
    analysis = {
        "total_return_pct": round(tot, 2),
        "ann_return_pct": round(ann_ret, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(md, 2),
        "win_rate_pct": round(wr, 1),
        "n_buys": n_buys,
        "n_sells": n_sells,
        "profit_loss_ratio": round(wlr, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "momentum_passed": momentum_passed,
        "momentum_rejected": momentum_rejected,
        "momentum_percentile": MOMENTUM_PERCENTILE,
        "vol_filter_rejected": vol_filter_rejected,
        "yearly_returns": yearly,
        "exit_reasons": exit_analysis,
        "signal_types": signal_dist,
        "env_distribution": env_dist,
        "fund_score_stats": score_stats,
        "initial_capital": INITIAL_CAP,
        "final_nav": round(eq_df.iloc[-1]["nav"], 2),
        "backtest_start": backtest_start,
        "backtest_end": end_date,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(str(OUT_DIR / "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    summary = {k: v for k, v in analysis.items() if isinstance(v, (int, float, str))}
    with open(str(OUT_DIR / "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # ── 打印结果 ──
    print(f"\n{'=' * 65}")
    print(f"📊 v6ab Baseline — 完整记录版")
    print(f"{'=' * 65}")
    print(f"  夏普:           {sharpe:.4f}")
    print(f"  总收益:         {tot:+.2f}%")
    print(f"  年化收益:       {ann_ret:+.2f}%")
    print(f"  最大回撤:       {md:.2f}%")
    print(f"  胜率:           {wr:.1f}%")
    print(f"  交易:           买入{n_buys} / 卖出{n_sells}")
    print(f"  盈亏比:         {wlr:.2f}")
    print(f"  平均盈利:       {avg_win:.2f}% | 平均亏损: {avg_loss:.2f}%")
    print(f"  动量过滤:       通过={momentum_passed} 拒绝={momentum_rejected}")
    print(f"  终值:           {eq_df.iloc[-1]['nav']:,.0f}")
    print(f"  耗时:           {time.time()-t0:.0f}s")
    
    print(f"\n{'─' * 55}")
    print("逐年收益")
    print(f"{'─' * 55}")
    print(f"{'年份':>6s}  {'收益':>8s}  {'交易':>6s}  {'胜率':>7s}  {'回撤':>8s}")
    for y, yd in sorted(yearly.items()):
        print(f"{y:>6s}  {yd['return_pct']:>+7.2f}%  {yd['n_trades']:>6d}  {yd['win_rate']:>6.1f}%  {yd['max_dd']:>7.2f}%")
    
    print(f"\n{'─' * 55}")
    print("退出原因")
    print(f"{'─' * 55}")
    for reason, data in sorted(exit_analysis.items(), key=lambda x: -x[1]["count"]):
        print(f"  {reason:<20s}  {data['count']:>6d}  avg_pnl={data['avg_pnl']:>+7.2f}%  "
              f"avg_hold={data['avg_hold_days']:>5.1f}d  wr={data['win_rate']:>5.1f}%")
    
    print(f"\n{'─' * 55}")
    print("信号类型")
    print(f"{'─' * 55}")
    for sig, cnt in sorted(signal_dist.items(), key=lambda x: -x[1]):
        print(f"  {sig:<20s}  {cnt:>6d}")
    
    print(f"\n{'─' * 55}")
    print("环境分布")
    print(f"{'─' * 55}")
    for e, cnt in sorted(env_dist.items()):
        print(f"  {e:<12s}  {cnt:>6d}")
    
    print(f"\n{'─' * 55}")
    print(f"💾 {OUT_DIR}")
    log.info(f"✅ v6ab Baseline 完成 ({time.time()-t0:.0f}s)")
    
    # ── 生成综合报告 ──
    generate_report(analysis, stock_history, trades_df, eq_df)
    log.info(f"✅ 报告已生成")

# ═══════════════════════════════════════════════════════════════
# 第六部分：综合报告生成
# ═══════════════════════════════════════════════════════════════

def generate_report(analysis, stock_history, trades_df, eq_df):
    """生成完整的策略分析报告（Markdown格式）"""
    
    report = f"""# v6ab Baseline 策略完整报告

> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
> 回测区间: {analysis['backtest_start']} ~ {analysis['backtest_end']}
> 运行耗时: {analysis['elapsed_seconds']}s

---

## 一、总体绩效

| 指标 | 值 |
|:----|:---:|
| 夏普比率 | {analysis['sharpe']} |
| 总收益率 | {analysis['total_return_pct']:+.2f}% |
| 年化收益率 | {analysis['ann_return_pct']:+.2f}% |
| 最大回撤 | {analysis['max_drawdown_pct']:.2f}% |
| 胜率 | {analysis['win_rate_pct']:.1f}% |
| 总交易 | 买入 {analysis['n_buys']} / 卖出 {analysis['n_sells']} |
| 盈亏比 | {analysis['profit_loss_ratio']:.2f} |
| 平均盈利 | {analysis['avg_win_pct']:.2f}% |
| 平均亏损 | {analysis['avg_loss_pct']:.2f}% |
| 初始资金 | {analysis['initial_capital']:,} |
| 最终净值 | {analysis['final_nav']:,.2f} |
| 终值增长率 | {(analysis['final_nav']/analysis['initial_capital']-1)*100:+.2f}% |

---

## 二、逐年收益

| 年份 | 收益 | 交易 | 胜率 | 最大回撤 |
|:---:|:---:|:---:|:---:|:--------:|
"""
    for y, yd in sorted(analysis['yearly_returns'].items()):
        report += f"| {y} | {yd['return_pct']:+.2f}% | {yd['n_trades']} | {yd['win_rate']:.1f}% | {yd['max_dd']:.2f}% |\n"
    
    report += f"""
---

## 三、退出原因分析

| 退出原因 | 笔数 | 占比 | 平均盈亏 | 平均持仓 | 胜率 |
|:--------|:---:|:---:|:-------:|:-------:|:---:|
"""
    total_sells = sum(d['count'] for d in analysis['exit_reasons'].values())
    for reason, data in sorted(analysis['exit_reasons'].items(), key=lambda x: -x[1]['count']):
        pct = data['count'] / max(total_sells, 1) * 100
        report += f"| {reason} | {data['count']} | {pct:.1f}% | {data['avg_pnl']:+.2f}% | {data['avg_hold_days']:.0f}d | {data['win_rate']:.1f}% |\n"
    
    report += f"""
---

## 四、完整参数配置

### 4.1 成本参数

| 参数 | 值 | 说明 |
|:----|:---:|:-----|
| SLIPPAGE_BUY | {SLIPPAGE_BUY} | 买入滑点 (0.1%) |
| SLIPPAGE_SELL | {SLIPPAGE_SELL} | 卖出滑点 (0.1%) |
| COMMISSION | 0.0003 (0.03%) | 佣金费率 |
| STAMP_TAX | 0.001 (0.1%) | 印花税 (卖出) |
| VOL_CAP_RATIO | {VOL_CAP_RATIO} | 单笔成交量上限比例 |

### 4.2 仓位控制参数

| 参数 | 值 | 说明 |
|:----|:---:|:-----|
| INITIAL_CAPITAL | {INITIAL_CAP:,} | 初始资金 |
| MAX_POSITIONS | {MAX_POS} | 最大同时持仓数 |
| POSITION_SIZE_PCT | {POSITION_SIZE_PCT} | 每笔风险预算 (占现金比例) |
| MAX_SINGLE_POS | {MAX_SINGLE_POS} | 单票最大仓位 (占现金比例) |
| COOLING_PERIOD_DAYS | {COOLING_DAYS} | 止损后冷却期 (交易日) |

### 4.3 仓位计算公式

```
信号权重 bs = {
    'third_point_buy': 0.6,   (第三类买点)
    'hard_divergence': 1.0,   (盘整背驰)
    'default': 0.5            (其他信号)
}

波动率 sa = max(ATR(21), 0.005)
波动因子 vf = bs / max(sa, 0.01)
基本面因子 fm = fund_score / 100

风险预算 ra = cash × {POSITION_SIZE_PCT} × vf × fm / max(sa, 0.01)
ra = min(ra, cash × {MAX_SINGLE_POS})    ← 单票上限

成交量约束 ms = 当日成交量 × {VOL_CAP_RATIO}
最终股数 sh = min(int(ra / 开盘价), ms)
```

### 4.4 止损止盈逻辑

```
环境分类 (CSI300):
  BULL:       tp1=12% tp2=18%  trail=88%  env_stop=None
  OSCILLATE:  tp1=6%  tp2=10%  trail=94%  env_stop=None
  BEAR:       tp1=3%  tp2=5%   trail=None env_stop=3%

退出优先级 (高→低):
  1. env_stop_loss  — 环境止损 (仅BEAR市场, -3%)
  2. stop_loss      — 硬止损 (-8%)
  3. trailing_stop  — 移动止盈 (盈利+5%后, 回撤触发)
  4. tp2            — 第二止盈 (减仓50%)
  5. tp1            — 第一止盈 (减仓30%)

动态调整:
  - 盈亏平衡锁: PnL >= +5% → 止损移至成本价
  - 持仓紧缩: 持仓 >= 60天 且 PnL < 5% → 止损收紧到入场价的 85%
```

### 4.5 动量过滤参数

| 参数 | 值 | 说明 |
|:----|:---:|:-----|
| MOMENTUM_WINDOW | {MOMENTUM_WINDOW} | 动量计算窗口 (交易日) |
| MOMENTUM_PERCENTILE | {MOMENTUM_PERCENTILE}% | 拒绝底部百分比 |
| MOMENTUM_UPDATE_FREQ | {MOMENTUM_UPDATE_FREQ} | 动量排名更新频率 (交易日) |

动量过滤: 每20个交易日对所有候选股票计算过去20天涨跌幅排名,
拒绝排名在底部 {MOMENTUM_PERCENTILE}% 的信号。

### 4.6 其他过滤条件

| 条件 | 说明 |
|:----|:-----|
| 基本面评分 | fund_score >= {FUND_SCORE_MIN} |
| 成交量 | 当日成交量 >= 20日均量 × {VOLUME_FILTER_RATIO} |
| 涨跌停检查 | 次日开盘价 <= 前日收盘 × 1.10 且 >= 前日收盘 × 0.90 |
| 冷却期 | 止损后 {COOLING_DAYS} 天内不买入同票 |

---

## 五、环境分类器 (CSI300)

### 5.1 分类逻辑

```
1. 均线条件:
   - MA20(20日均线)
   - MA60(60日均线)
   - MA120(120日均线)

2. 趋势判定:
   BULL: MA20 > MA60 > MA120 (多头排列)
   BEAR: MA20 < MA60 < MA120 (空头排列)

3. ADX 过滤:
   ADX(14) > 22 才算趋势市
   ADX <= 22 判为震荡

4. 波动率保护:
   ATR(20)/收盘价 > 85%分位 → 强制震荡
```

### 5.2 环境分布

| 环境 | 天数 | 占比 |
|:----|:---:|:---:|
"""
    for e, cnt in sorted(analysis.get('env_distribution', {}).items()):
        total_days = sum(analysis.get('env_distribution', {}).values())
        pct = cnt / max(total_days, 1) * 100
        report += f"| {e} | {cnt} | {pct:.1f}% |\n"
    
    report += f"""
---

## 六、信号类型分布

| 信号类型 | 次数 | 占比 |
|:--------|:---:|:---:|
"""
    total_sigs = sum(analysis.get('signal_types', {}).values())
    for sig, cnt in sorted(analysis['signal_types'].items(), key=lambda x: -x[1]):
        pct = cnt / max(total_sigs, 1) * 100
        report += f"| {sig} | {cnt} | {pct:.1f}% |\n"
    
    report += f"""
---

## 七、基本面评分统计

| 统计项 | 值 |
|:------|:---:|
| 使用真实评分 | {analysis.get('fund_score_stats', {}).get('used', 0)} |
| 使用默认评分(50) | {analysis.get('fund_score_stats', {}).get('default', 0)} |
| 评分拒单 | {analysis.get('fund_score_stats', {}).get('rejected', 0)} |
| 评分通过 | {analysis.get('fund_score_stats', {}).get('passed', 0)} |

---

## 八、动量过滤效果

| 统计项 | 值 |
|:------|:---:|
| 动量通过 | {analysis.get('momentum_passed', 0)} |
| 动量拒绝 | {analysis.get('momentum_rejected', 0)} |
| 拒绝率 | {analysis.get('momentum_rejected', 0)/max(analysis.get('momentum_passed', 0)+analysis.get('momentum_rejected', 0),1)*100:.1f}% |
| 成交量过滤拒绝 | {analysis.get('vol_filter_rejected', 0)} |
"""
    
    with open(str(OUT_DIR / "baseline_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"  报告已写入 baseline_report.md")

if __name__ == "__main__":
    run_baseline()
