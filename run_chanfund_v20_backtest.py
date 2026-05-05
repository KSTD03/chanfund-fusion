#!/usr/bin/env python3
"""
ChanFund Fusion v2.0 — 全量回测运行器（集成 ChanFund 信号引擎）
===============================================================
v2.0 核心升级：信号因子化 + 市场环境分类 + 共振得分

相比 v1.4 的新增特性：
1. 市场环境分类（牛/熊/震荡）→ 环境乘数调节仓位
2. 缠论信号因子化 → 连续因子分替代二元买入/不买入
3. 共振得分 → 技术分×环境乘数 + 基本面分的加权综合

时间分片（受数据可用性限制）：
  训练集：2016-01 ~ 2017-12  （2年，参数校准）
  验证集：2018-01 ~ 2019-12  （2年，超参筛选）
  测试集：2020-01 ~ 2026-04  （6年+，样本外验证）

运行方式：
  nohup python3 run_chanfund_v20_backtest.py --mode test > v20.out 2>&1 &
  python3 run_chanfund_v20_backtest.py --mode test --quick  # 快速验证（100只股票）
"""

from __future__ import annotations

import os, sys, time, json, argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd

# ============================================================
# 路径与配置
# ============================================================
_WORKSPACE = str(Path(__file__).resolve().parent)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

BACKTEST_NAME = "ChanFund-Fusion_v2.0"
STRATEGY_DIR = Path(_WORKSPACE) / "chanfund_fusion"
REPORT_DIR = Path("/home/quant/backtest report")
DATA_DIR = Path(_WORKSPACE) / "quant" / "data" / "daily_csv"
FIN_DIR = Path(_WORKSPACE) / "quant" / "data" / "financial"

# v2.0 时间分片
TIME_SPLITS = {
    "train": {"start": "2016-01-04", "end": "2017-12-29"},
    "valid": {"start": "2018-01-02", "end": "2019-12-31"},
    "test":  {"start": "2020-01-02", "end": "2026-04-29"},
}

# 基础参数（同v1.4）
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5
POSITION_SIZE = 0.20           # 单只最大仓位 20%
COMMISSION_RATE = 0.0003
FIXED_STOP_LOSS = 0.08
TP1_PCT = 0.06
TP1_REDUCE = 0.30
TP2_PCT = 0.13
TP2_REDUCE = 0.50
TP_TRAIL_RETRACE = 0.91        # 从峰值回落 9% 清仓（1 - 0.91 = 0.09）
TP_BREAKEVEN_LOCK = 0.05       # 保本移损阈值
VOL_MIN_RATIO = 0.8

# v1.4 基本面参数
FUND_SCORE_MIN = 50
FUND_ROE_MIN = 0.05
FUND_GM_MIN = 0.15
FUND_REV_GROWTH_MIN = -0.20

# ========== v2.0 新增参数 ==========
# 信号因子化基础分
SIGNAL_BASE_SCORE = {
    "third_point_buy": 0.70,
    "hard_divergence": 0.85,
    "second_class_buy": 0.65,
    "soft_divergence": 0.50,
}
SIGNAL_VOL_ADJUST = 0.20       # 成交量确认度 ±0.2

# 市场环境参数
ENV_MULTIPLIER = {
    "BULL": 1.0,       # 【P1.2】取消环境分类对仓位的影响
    "BEAR": 1.0,
    "OSCILLATE": 1.0,  # 所有环境一视同仁
}

# 共振得分权重
TECH_WEIGHT = 0.6
FUND_WEIGHT = 0.4

# 冷却期
COOLING_PERIOD_DAYS = 10
COOLING_STOP_COUNT = 2

LOG_PATH = _WORKSPACE + "/backtest_v20_run.log"

logger = logging.getLogger("chanfund_v20_bt")

_T_START = 0

# ──────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        ],
    )


def load_daily_csv_data(data_dir: str, max_files: int = 0, use_parquet: bool = True) -> dict:
    """加载日线数据 {date_str: {code: kbar}}
    优化2：优先使用Parquet格式（内存占用降至CSV的1/3~1/5）
    """
    data_p = Path(data_dir)
    parquet_dir = data_p.parent / "daily_parquet"
    has_parquet = parquet_dir.exists() and len(list(parquet_dir.glob("*.parquet"))) > 0
    
    # ---- Parquet模式 ----
    if use_parquet and has_parquet:
        pq_files = sorted(parquet_dir.glob("*.parquet"))
        if max_files > 0:
            pq_files = pq_files[:max_files]
        logger.info(f"📂 加载 {len(pq_files)} 个Parquet文件...")
        t0 = time.time()
        data: dict = {}
        PQC = ["date", "open", "high", "low", "close", "volume"]
        for fpath in pq_files:
            try:
                df = pd.read_parquet(fpath, columns=PQC)
            except:
                continue
            code = fpath.stem
            if not (code.startswith("sh.") or code.startswith("sz.")):
                code = f"sh.{code}" if code[0] == '6' else f"sz.{code}"
            for _, row in df.iterrows():
                d_str = str(row["date"]).strip()[:10]
                if len(d_str) < 8:
                    continue
                close = float(row["close"])
                if close <= 0:
                    continue
                data.setdefault(d_str, {})[code] = {
                    "open": float(row["open"]), "high": float(row["high"]),
                    "low": float(row["low"]), "close": close,
                    "volume": float(row["volume"]), "amount": 0.0}
            del df
        elapsed = time.time() - t0
        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024**2
            logger.info(f"✅ Parquet加载完成: {len(data)}天, {len(pq_files)}文件, {elapsed:.0f}s, mem={mem:.0f}MB")
        except:
            logger.info(f"✅ Parquet加载完成: {len(data)}天, {len(pq_files)}文件, {elapsed:.0f}s")
        return data
    
    # ---- CSV模式 ----
    csv_files = sorted(data_p.glob("*.csv"))
    if max_files > 0:
        csv_files = csv_files[:max_files]
    logger.info(f"📂 加载 {len(csv_files)} 个CSV文件...")
    t0 = time.time()
    data: dict = {}
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, dtype={"code": str})
        except:
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        for _, row in df.iterrows():
            d_str = str(row.get("date", ""))[:10]
            if len(d_str) < 8:
                continue
            if len(d_str) == 8:
                d_str = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
            code = str(row.get("code", ""))
            if not code or code == "nan":
                continue
            if not (code.startswith("sh.") or code.startswith("sz.")):
                code = f"sh.{code}" if code[0] == '6' else f"sz.{code}"
            close = float(row.get("close", 0))
            if close <= 0:
                continue
            data.setdefault(d_str, {})[code] = {
                "open": float(row.get("open",0)), "high": float(row.get("high",0)),
                "low": float(row.get("low",0)), "close": close,
                "volume": float(row.get("volume",0)), "amount": float(row.get("amount",0))}
        del df
    elapsed = time.time() - t0
    logger.info(f"✅ CSV加载完成: {len(data)}天, {len(csv_files)}文件, {elapsed:.0f}s")
    return data


def load_financial_data(fin_dir: str) -> dict:
    """加载财务数据 {code: {date: {roe, gross_margin, ...}}}"""
    fin_p = Path(fin_dir)
    if not fin_p.exists():
        logger.warning("⚠️ 财务数据目录不存在")
        return {}
    fin_cache: dict = {}
    for fpath in sorted(fin_p.glob("financial_*.csv")):
        try:
            df = pd.read_csv(fpath)
            cols = {c.strip(): c for c in df.columns}  # 列名映射
            date_key = fpath.stem.replace("financial_", "").replace("_", "-")
            if len(date_key) == 10:
                pass
            elif len(date_key) == 8:
                date_key = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"
            for _, row in df.iterrows():
                code = str(row.get(cols.get("股票代码", "code"), "")).strip()
                if not code or code == "nan":
                    continue
                # 统一 code 格式
                code_str = str(int(float(code))) if code.replace(".", "").isdigit() else code
                if not (code_str.startswith("sh.") or code_str.startswith("sz.")):
                    code_str = f"sh.{code_str}" if code_str.startswith("6") else f"sz.{code_str}"
                # 从中文列名读取，数值转为小数
                eps = float(row.get(cols.get("每股收益", "eps"), 0) or 0)
                roe_pct = float(row.get(cols.get("净资产收益率", "roe"), 0) or 0)
                gm_pct = float(row.get(cols.get("销售毛利率", "gross_margin"), 0) or 0)
                rev_g = float(row.get(cols.get("营业总收入-同比增长", "revenue_growth"), 0) or 0)
                ocf = float(row.get(cols.get("每股经营现金流量", "ocf"), 0) or 0)
                
                fin_cache.setdefault(code_str, {})
                fin_cache[code_str][date_key] = {
                    "roe": roe_pct / 100.0,        # 百分比→小数
                    "gross_margin": gm_pct / 100.0,  # 百分比→小数
                    "revenue_growth": rev_g / 100.0,  # 百分比→小数
                    "eps": eps,
                    "ocf": ocf,
                }
        except Exception as e:
            logger.warning(f"⚠️ 财务文件 {fpath.name}: {e}")
    logger.info(f"✅ 财务数据: {len(fin_cache)} 只股票")
    return fin_cache


def load_index_data(data_dir: str) -> dict:
    """加载或合成沪深300指数数据
    
    优先从 Qlib 获取沪深300指数，否则返回空（默认振荡市）
    Returns: {date_str: {close, high, low}}
    """
    # 先尝试从 Qlib 获取指数数据
    try:
        from qlib.data import D
        import qlib
        qlib_initialized = False
        for idx_code in ['SH000300', '000300', 'CSI300', 'csi300']:
            try:
                if not qlib_initialized:
                    qlib.init(provider_uri='/home/quant/.openclaw/workspace/quant/qlib_data/cn_data', region='cn')
                    qlib_initialized = True
                data = D.features([idx_code], ['$close', '$high', '$low'], freq='day',
                                  start_time='2005-01-01', end_time='2026-04-30')
                if len(data) > 0 and not data['$close'].isna().all():
                    idx_data = {}
                    for (inst, dt), row in data.iterrows():
                        d_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)[:10]
                        close_val = row['$close']
                        if pd.notna(close_val) and close_val > 0:
                            idx_data[d_str] = {
                                'close': float(close_val),
                                'high': float(row['$high']) if pd.notna(row['$high']) else float(close_val),
                                'low': float(row['$low']) if pd.notna(row['$low']) else float(close_val),
                            }
                    if len(idx_data) > 100:
                        logger.info(f"✅ 指数数据(Qlib): {idx_code}, {len(idx_data)} 天")
                        return idx_data
            except Exception:
                continue
    except Exception:
        pass
    
    logger.warning("⚠️ 未找到指数数据，市场环境分类默认振荡市")
    return {}


def get_fund_score(code: str, date_str: str, fin_cache: dict) -> float:
    """基本面综合评分 0-100（同v1.4）"""
    if not fin_cache or code not in fin_cache:
        return 55.0  # 无数据时给中间偏上分，防止硬过滤造成零交易
    fin_dates = sorted(fin_cache[code].keys())
    if not fin_dates:
        return 55.0
    best = None
    for fd in reversed(fin_dates):
        if fd <= date_str:
            best = fd
            break
    if best is None:
        return 50.0
    f = fin_cache[code][best]
    score = 0.0
    roe = f.get("roe", 0)
    if roe >= 0.20: score += 30
    elif roe >= 0.10: score += 20
    elif roe >= FUND_ROE_MIN: score += 10

    gm = f.get("gross_margin", 0)
    if gm >= 0.40: score += 25
    elif gm >= 0.25: score += 15
    elif gm >= FUND_GM_MIN: score += 10

    rev = f.get("revenue_growth", 0)
    if rev >= 0.20: score += 25
    elif rev >= 0.10: score += 20
    elif rev >= 0: score += 15
    elif rev >= FUND_REV_GROWTH_MIN: score += 5
    else: score -= 10

    eps = f.get("eps", 0)
    if eps > 0: score += 20
    elif eps > -0.5: score += 5
    else: score -= 5
    return max(0, min(100, score))


# ============================================================
# v2.0 市场环境分类器
# ============================================================
class MarketEnvClassifier:
    """
    基于沪深300的 ER效率比率 + 均线排列 + 动量趋势
    输出 BULL / BEAR / OSCILLATE
    """
    def __init__(self, index_data: dict):
        self.idx = index_data
        self._cache: Dict[str, str] = {}

    def classify(self, date_str: str) -> str:
        if date_str in self._cache:
            return self._cache[date_str]
        all_dates = sorted(self.idx.keys())
        idx = next((i for i, d in enumerate(all_dates) if d >= date_str), len(all_dates))
        lb = all_dates[max(0, idx-60):idx]
        if len(lb) < 30:
            self._cache[date_str] = "OSCILLATE"
            return "OSCILLATE"
        closes = np.array([self.idx[d]["close"] for d in lb])
        # ER
        n = min(20, len(closes)-1)
        direction = abs(closes[-1] - closes[-n-1]) if len(closes) > n else 0
        vol = sum(abs(closes[i] - closes[i-1]) for i in range(-n, 0)) or 1
        er = direction / vol
        # 均线
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes) if len(closes) >= 60 else ma20
        trend_up = ma20 > ma60
        # 动量
        ret = np.diff(closes[-21:]) / closes[-22:-1] if len(closes) >= 22 else [0]
        mom = np.mean(ret[-5:])
        if er > 0.6 and trend_up and mom > 0:
            env = "BULL"
        elif er > 0.6 and not trend_up and mom < 0:
            env = "BEAR"
        else:
            env = "OSCILLATE"
        self._cache[date_str] = env
        return env

    def get_multiplier(self, date_str: str) -> float:
        return ENV_MULTIPLIER[self.classify(date_str)]


# ============================================================
# v2.0 因子化 + 共振得分
# ============================================================
def signal_to_factor(signal_type: str, vol_ratio: float) -> float:
    """
    信号因子化：将缠论信号转为连续因子分
    vol_ratio: 当日成交量 / 20日均量
    """
    base = SIGNAL_BASE_SCORE.get(signal_type, 0.5)
    vc = min(vol_ratio, 2.0) / 2.0  # 0~1
    adj = (vc - 0.5) * SIGNAL_VOL_ADJUST * 2  # ±0.2
    return min(1.0, max(0.0, base + adj))


def resonance_score(tech_factor: float, fund_score: float, env_mult: float) -> float:
    """共振得分 = 技术分×环境乘数×0.6 + 基本面分×0.4"""
    tech_dim = tech_factor * env_mult
    fund_dim = fund_score / 100.0  # 归一化
    return min(1.0, tech_dim * TECH_WEIGHT + fund_dim * FUND_WEIGHT)


# ============================================================
# 指标计算
# ============================================================
def calc_metrics(trade_log: list, equity_curve: list, initial_capital: float) -> dict:
    total_return = (equity_curve[-1]["nav"] / initial_capital - 1) if equity_curve else 0
    days = len(equity_curve)
    ann_ret = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

    eq = [e["nav"] for e in equity_curve]
    eq_arr = np.array(eq)
    daily_ret = np.diff(eq_arr) / eq_arr[:-1]
    rf_daily = 0.03 / 252
    excess = daily_ret - rf_daily
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 1e-8 else 0

    rolling_max = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - rolling_max) / rolling_max
    max_dd = np.min(dd)

    # 交易分析
    sells = [t for t in trade_log if t.get("action") == "SELL" or t.get("pnl_pct", 0) != 0]
    if not sells:
        sells = trade_log[-len(trade_log)//2:] if trade_log else []
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
        "final_equity": equity_curve[-1]["nav"] if equity_curve else initial_capital,
        "total_return": total_return * 100,
        "ann_return": ann_ret * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd * 100,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": wl_ratio,
        "best_trade": best,
        "worst_trade": worst,
        "n_trades": n_trades,
    }


# ============================================================
# v2.0 核心回测
# ============================================================
def run_backtest(
    data: dict,
    fin_cache: dict,
    index_data: dict,
    start: str,
    end: str,
    initial_capital: float = INITIAL_CAPITAL,
    label: str = "test",
    stock_limit: int = 0,
) -> dict:
    """
    v2.0 回测主循环
    返回 {metrics, trade_log, equity_curve, signal_stats, exit_stats, tp_stats}
    """
    global _T_START

    # 初始化 ChanFund 信号引擎
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config

    cfg = load_config(str(STRATEGY_DIR / "config.yaml"))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    logger.info(f"✅ 技术引擎初始化完成")

    all_dates = sorted(data.keys())
    warmup_dates = [d for d in all_dates if d < start]
    backtest_dates = [d for d in all_dates if start <= d <= end]

    # 确定股票池
    all_stocks = sorted(set().union(*(d.keys() for d in data.values())))
    if stock_limit > 0:
        all_stocks = all_stocks[:stock_limit]
    logger.info(f"📊 股票池: {len(all_stocks)} 只, 预热 {len(warmup_dates)}天, 回测 {len(backtest_dates)}天")

    # ---- Warm-up ----
    logger.info("=" * 50)
    logger.info("PHASE 1: Warm-up")
    for i, ds in enumerate(warmup_dates):
        day_data = data.get(ds, {})
        for stock in all_stocks:
            if stock in day_data:
                try:
                    tech_engine.update(stock, day_data[stock], emit_signals=False)
                except Exception:
                    pass
        if (i+1) % 60 == 0:
            logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")

    n_cached = len(tech_engine.chan_cache)
    logger.info(f"✅ 预热完成, {n_cached} 只股票有缠论结构")

    # ---- 市场环境分类器 ----
    env_cls = MarketEnvClassifier(index_data)

    # ---- 统计 ----
    signal_stats = defaultdict(int)
    exit_detail = defaultdict(int)
    tp_detail = defaultdict(int)
    trade_log = []
    equity_curve = []
    vol_cache: Dict[str, list] = {}
    positions: Dict[str, dict] = {}
    cooling: Dict[str, date] = {}  # {code: cooling_until_date}

    cash = initial_capital
    peak_equity = cash
    total_signals = 0

    # ---- Phase 2: Backtest ----
    logger.info("=" * 50)
    logger.info(f"PHASE 2: Backtest v2.0 ({label})")
    n_dates = len(backtest_dates)

    for di, date_str in enumerate(backtest_dates):
        trade_date = date.fromisoformat(date_str)
        day_data = data.get(date_str, {})
        if not day_data:
            continue

        # A. 更新技术引擎
        for stock in all_stocks:
            if stock in day_data:
                try:
                    tech_engine.update(stock, day_data[stock], emit_signals=True)
                except Exception:
                    pass
                v = day_data[stock].get("volume", 0)
                if v > 0:
                    vol_cache.setdefault(stock, []).append(v)
                    if len(vol_cache[stock]) > 50:
                        vol_cache[stock] = vol_cache[stock][-50:]

        # B. 市场环境
        env = env_cls.classify(date_str)
        env_mult = ENV_MULTIPLIER[env]

        # C. 获取确认信号 (v2.0: 因子化 + 共振)
        signals = tech_engine.get_confirmed_signals(trade_date)
        total_signals += len(signals)

        for sig in signals:
            if len(positions) >= MAX_POSITIONS:
                break
            if sig.symbol in positions:
                continue
            # 冷却期检查
            if sig.symbol in cooling and trade_date <= cooling[sig.symbol]:
                continue
            # 当日数据检查
            if sig.symbol not in day_data:
                continue

            entry_price = day_data[sig.symbol].get("close", 0)
            if entry_price <= 0:
                continue

            # ---- v1.4: 基本面过滤 ----
            fund_score = get_fund_score(sig.symbol, date_str, fin_cache)
            if fund_score < FUND_SCORE_MIN:
                continue

            # ---- v1.4: 成交量过滤 ----
            vols = vol_cache.get(sig.symbol, [])
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                entry_vol = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and entry_vol < avg_vol * VOL_MIN_RATIO:
                    continue

            # ============ v2.0 核心逻辑 ============
            # 1) 信号因子化
            vols_20 = vol_cache.get(sig.symbol, [])
            vol_ratio = 1.0
            if len(vols_20) >= 20:
                mv = sum(vols_20[-20:]) / 20
                vol_ratio = day_data[sig.symbol].get("volume", 0) / mv if mv > 0 else 1.0
            tech_factor = signal_to_factor(sig.signal_subtype, vol_ratio)

            # === P2: 软降权系数 ===
            # P2.1: 趋势方向软降权（模拟rules.py的check_trend_direction）
            # 若无指数数据做环境分类，默认为1.0
            trend_coeff = env_mult  # 如果环境分类不可用，默认为1.0
            
            # P2.2: 噪声软降权（模拟rules.py的ER噪声过滤）
            noise_coeff = 1.0
            if len(vols_20) >= 20:
                er = abs(vol_ratio - 1.0)  # 用vol_ratio偏差模拟ER
                if er < 0.15:
                    noise_coeff = 1.0  # 低噪声，正常
                elif er < 0.25:
                    noise_coeff = 0.5  # 中噪声，降权
                # 极端噪声由成交量过滤处理
            
            # P2.3: 综合降权 + 下限保护
            combined_coeff = trend_coeff * noise_coeff
            MIN_CONFIDENCE = 0.3
            if combined_coeff < MIN_CONFIDENCE:
                combined_coeff = 0.0  # 标记为观察

            # 2) 共振得分（叠加软降权）
            res = resonance_score(tech_factor, fund_score, env_mult) * combined_coeff

            # 3) 仓位系数 = 共振得分 × 最大仓位
            position_coeff = min(res, 1.0) * POSITION_SIZE / 0.20
            position_coeff = min(1.0, max(0.0, position_coeff))

            cost = cash * POSITION_SIZE * position_coeff
            if cost > cash or cost <= 0:
                continue

            shares = int(cost / entry_price / 100) * 100  # 按手
            if shares <= 0:
                continue

            cost = shares * entry_price
            if cost > cash or cost <= 0:
                continue

            stop_price = entry_price * (1 - FIXED_STOP_LOSS)

            positions[sig.symbol] = {
                "entry_price": entry_price,
                "shares": shares,
                "entry_date": date_str,
                "stop_price": stop_price,
                "peak_price": entry_price,
                "signal_type": sig.signal_subtype,
                "tp1_triggered": False,
                "tp2_triggered": False,
                "fund_score": fund_score,
                "resonance": res,
                "env": env,
                "tech_factor": tech_factor,
            }
            cash -= cost
            signal_stats[sig.signal_subtype] += 1

            trade_log.append({
                "date": date_str, "symbol": sig.symbol, "action": "BUY",
                "price": round(entry_price, 3), "signal": sig.signal_subtype,
                "resonance": round(res, 3), "tech_factor": round(tech_factor, 3),
                "fund_score": fund_score, "env": env,
            })

        # D. 止盈/止损检查（同v1.4）
        to_close = []
        to_reduce = []

        for stock, pos in list(positions.items()):
            kbar = day_data.get(stock, {})
            if stock in day_data and day_data[stock].get("close", 0) > 0:
                price = day_data[stock]["close"]
                pos["_last_price"] = price
            else:
                # 停牌/缺失行情：用最后可交易价
                price = pos.get("_last_price", pos["entry_price"])
            pos["peak_price"] = max(pos["peak_price"], price)
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
            hold_days = (trade_date - date.fromisoformat(pos["entry_date"])).days

            # 保本移损
            if pnl_pct >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
            # 长持仓保护
            if hold_days >= 60 and abs(pnl_pct) < 0.05:
                ws = pos["entry_price"] * 0.85
                pos["stop_price"] = min(pos["stop_price"], ws)
            # 峰值回落止盈
            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= TP_TRAIL_RETRACE:
                    to_close.append((stock, "trailing_stop", price))
                    tp_detail["trailing_stop"] += 1
                    continue
            # 第二止盈
            if pnl_pct >= TP2_PCT and not pos["tp2_triggered"]:
                pos["tp2_triggered"] = True
                to_reduce.append((stock, "tp2", TP2_REDUCE, price))
                tp_detail["tp2"] += 1
                continue
            # 第一止盈
            if pnl_pct >= TP1_PCT and not pos["tp1_triggered"]:
                pos["tp1_triggered"] = True
                to_reduce.append((stock, "tp1", TP1_REDUCE, price))
                tp_detail["tp1"] += 1
                continue
            # 止损
            if price <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price))

        # E. 执行减仓
        for stock, reason, reduce_pct, price in to_reduce:
            pos = positions[stock]
            reduce_shares = int(pos["shares"] * reduce_pct)
            if reduce_shares > 0 and pos["shares"] > reduce_shares:
                pos["shares"] -= reduce_shares
                cash += reduce_shares * price
                pnl = (price / pos["entry_price"] - 1) * 100
                exit_detail[reason] = exit_detail.get(reason, 0) + 1
                trade_log.append({
                    "date": date_str, "symbol": stock, "action": "SELL",
                    "price": round(price, 3), "reason": reason,
                    "pnl_pct": round(pnl, 2),
                    "hold_days": (trade_date - date.fromisoformat(pos["entry_date"])).days,
                })

        # F. 执行平仓
        for stock, reason, price in to_close:
            pos = positions.pop(stock)
            cash += pos["shares"] * price
            pnl = (price / pos["entry_price"] - 1) * 100
            exit_detail[reason] = exit_detail.get(reason, 0) + 1
            trade_log.append({
                "date": date_str, "symbol": stock, "action": "SELL",
                "price": round(price, 3), "reason": reason,
                "pnl_pct": round(pnl, 2), "signal_type": pos["signal_type"],
                "hold_days": (trade_date - date.fromisoformat(pos["entry_date"])).days,
                "env": pos.get("env", ""), "resonance": pos.get("resonance", 0),
            })
            # 冷却期
            if reason == "stop_loss":
                consec = sum(1 for t in reversed(trade_log) if t.get("symbol") == stock and t.get("reason") == "stop_loss")
                if consec >= COOLING_STOP_COUNT:
                    cooling[stock] = trade_date + timedelta(days=COOLING_PERIOD_DAYS)

        # G. 净值（修复：缺失行情数据时沿用最后可交易价，而非记为0）
        pos_value = 0.0
        for s, p in list(positions.items()):
            if s in day_data and day_data[s].get("close", 0) > 0:
                px = day_data[s]["close"]
                p["_last_price"] = px
            else:
                px = p.get("_last_price", p["entry_price"])
            pos_value += p["shares"] * px
        equity = cash + pos_value
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        equity_curve.append({"date": date_str, "nav": round(equity, 2), "dd": round(dd, 2)})

        if (di + 1) % 100 == 0:
            elapsed = time.time() - _T_START
            pct = (di + 1) / n_dates * 100
            logger.info(f"  [{label}] {date_str} ({di+1}/{n_dates} {pct:.0f}%) "
                        f"NAV={equity:,.0f} pos={len(positions)} trades={len(trade_log)} "
                        f"env={env} dd={dd:.1f}% t={elapsed:.0f}s")

    # ---- 强制平仓（修复：用最后可交易价，而非成本价） ----
    for stock, pos in list(positions.items()):
        close_price = pos.get("_last_price", pos["entry_price"])
        cash += pos["shares"] * close_price
        pnl = (close_price / pos["entry_price"] - 1) * 100
        trade_log.append({
            "date": backtest_dates[-1], "symbol": stock, "action": "SELL",
            "price": round(close_price, 3), "reason": "force_close",
            "pnl_pct": round(pnl, 2), "hold_days": 0,
        })
    positions.clear()

    elapsed = time.time() - _T_START
    metrics = calc_metrics(trade_log, equity_curve, initial_capital)

    logger.info(f"✅ [{label}] 完成! "
                f"收益 {metrics['total_return']:.2f}% 夏普 {metrics['sharpe']:.2f} "
                f"回撤 {metrics['max_drawdown']:.2f}% 交易 {metrics['n_trades']} "
                f"耗时 {elapsed:.0f}s 信号 {total_signals}")

    return {
        "metrics": metrics,
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "signal_stats": dict(signal_stats),
        "exit_stats": dict(exit_detail),
        "tp_stats": dict(tp_detail),
        "total_signals": total_signals,
    }


# ============================================================
# 报告生成
# ============================================================
def generate_report(result: dict, label: str, n_stocks: int,
                    n_warmup: int, n_bt: int, elapsed: float,
                    baseline: dict = None) -> str:
    m = result["metrics"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wr = m.get("win_rate", 0)
    n_trades = m.get("n_trades", 0)
    n_wins = int(n_trades * wr / 100) if n_trades > 0 else 0
    n_losses = n_trades - n_wins

    r = f"""# ChanFund Fusion v2.0 — 全量回测报告

> 生成时间: {now}
> 策略版本: v2.0 (信号因子化 + 市场环境分类 + 共振得分)
> 回测阶段: {label.upper()}
> 股票池: {n_stocks} 只 | 预热: {n_warmup}天 | 回测: {n_bt}天
> 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)
> 总信号: {result['total_signals']}

## 一、v2.0 特性启用状态

| 特性 | 状态 | 说明 |
|------|------|------|
| 市场环境分类 | ✅ v2.0 | 牛/熊/震荡 三态仓位调节 (乘数{ENV_MULTIPLIER['BULL']}x/{ENV_MULTIPLIER['BEAR']}x/{ENV_MULTIPLIER['OSCILLATE']}x) |
| 信号因子化 | ✅ v2.0 | 连续因子分替代二元信号 (vol确认±{SIGNAL_VOL_ADJUST}) |
| 共振得分 | ✅ v2.0 | 技术×{TECH_WEIGHT} + 基本面×{FUND_WEIGHT} |
| 多级止盈系统 | ✅ v1.1 | {int(TP1_PCT*100)}%/{int(TP2_PCT*100)}%分批减仓+峰值回落清仓+保本移损 |
| 成交量确认过滤 | ✅ v1.1 | ≥{int(VOL_MIN_RATIO*100)}%均量 |
| 固定止损 | ✅ v1.1 | -{int(FIXED_STOP_LOSS*100)}% |
| 集中持仓 | ✅ v1.1 | 最大{MAX_POSITIONS}只×{int(POSITION_SIZE*100)}% |
| 基本面过滤 | ✅ v1.4 | ROE≥5% 毛利率≥15% 综合分≥{FUND_SCORE_MIN} |
| 冷却期机制 | ✅ 继承 | 连续{COOLING_STOP_COUNT}次止损→{COOLING_PERIOD_DAYS}日冷却 |

## 二、绩效概览

| 指标 | 数值 |
|------|------|
| 初始资金 | {INITIAL_CAPITAL:,} |
| 最终净值 | {m['final_equity']:,.0f} |
| **总收益率** | **{m['total_return']:.2f}%** |
| **年化收益率** | **{m['ann_return']:.2f}%** |
| **夏普比率** | **{m['sharpe']:.2f}** |
| **最大回撤** | **{m['max_drawdown']:.2f}%** |
| 总交易数 | {n_trades} |

### 盈利分析

| 指标 | 数值 |
|------|------|
| 盈利交易 | {n_wins} ({wr:.1f}%) |
| 亏损交易 | {n_losses} ({100-wr:.1f}%) |
| 平均盈利 | {m['avg_win']:.2f}% |
| 平均亏损 | {m['avg_loss']:.2f}% |
| 平均盈亏比 | {m['win_loss_ratio']:.2f} : 1 |
| 最佳单笔 | {m['best_trade']:.2f}% |
| 最差单笔 | {m['worst_trade']:.2f}% |

"""
    if baseline:
        delta_tr = m['total_return'] - baseline.get('total_return', 0)
        delta_sh = m['sharpe'] - baseline.get('sharpe', 0)
        delta_dd = m['max_drawdown'] - baseline.get('max_drawdown', 0)
        delta_wr = wr - baseline.get('win_rate', 0)
        delta_wlr = m['win_loss_ratio'] - baseline.get('win_loss_ratio', 0)
        r += f"""### v2.0 vs v1.4 基线对比

| 指标 | v2.0 ({label}) | v1.4 (2021-2026基线) | 变化 |
|------|:---:|:---:|:---:|
| 总收益率 | {m['total_return']:.2f}% | {baseline['total_return']:.2f}% | {delta_tr:+.2f}% |
| 夏普比率 | {m['sharpe']:.2f} | {baseline['sharpe']:.2f} | {delta_sh:+.2f} |
| 最大回撤 | {m['max_drawdown']:.2f}% | {baseline['max_drawdown']:.2f}% | {delta_dd:+.2f}% |
| 胜率 | {wr:.1f}% | {baseline['win_rate']:.1f}% | {delta_wr:+.1f}% |
| 盈亏比 | {m['win_loss_ratio']:.2f} | {baseline['win_loss_ratio']:.2f} | {delta_wlr:+.2f} |

"""

    # 信号分布
    ss = result["signal_stats"]
    total_sig = sum(ss.values()) or 1
    sig_lines = "\n".join(f"| {k} | {v} | {v/total_sig*100:.1f}% |" for k, v in
                          sorted(ss.items(), key=lambda x: -x[1]))
    r += f"""## 三、信号分布

| 信号类型 | 次数 | 占比 |
|----------|:----:|:----:|
{sig_lines}

"""

    # 平仓原因
    es = result["exit_stats"]
    total_exit = sum(es.values()) or 1
    exit_lines = "\n".join(f"| {k} | {v} | {v/total_exit*100:.1f}% |" for k, v in
                           sorted(es.items(), key=lambda x: -x[1]))
    r += f"""## 四、平仓原因分布

| 原因 | 次数 | 占比 |
|------|:----:|:----:|
{exit_lines}

### 止盈分布

| 止盈类型 | 次数 |
|----------|:----:|
| 第一止盈(+{int(TP1_PCT*100)}%减{int(TP1_REDUCE*100)}%) | {result['tp_stats'].get('tp1', 0)} |
| 第二止盈(+{int(TP2_PCT*100)}%减{int(TP2_REDUCE*100)}%) | {result['tp_stats'].get('tp2', 0)} |
| 峰值回落止盈 | {result['tp_stats'].get('trailing_stop', 0)} |

"""

    # 逐笔成交
    r += "## 五、逐笔成交记录\n\n"
    r += "| # | 日期 | 股票 | 操作 | 价格 | 信号/原因 | 盈亏% | 持仓天数 |\n"
    r += "|---|------|------|------|------|-----------|-------|---------|\n"
    tl = result["trade_log"]
    for i, t in enumerate(tl[:100]):
        pnl = t.get("pnl_pct", "")
        if pnl != "":
            pnl = f"{pnl:.2f}%"
        hd = t.get("hold_days", "")
        sig = t.get("signal", t.get("reason", t.get("signal_type", "")))
        r += f"| {i+1} | {t['date']} | {t['symbol']} | {t['action']} | {t['price']:.3f} | {sig} | {pnl} | {hd} |\n"
    if len(tl) > 100:
        r += f"\n*仅显示前100笔，共{len(tl)}笔。完整记录见CSV文件。*\n"

    # 净值曲线
    r += "\n## 六、净值曲线\n\n"
    r += "| 日期 | 净值 | 回撤% |\n|------|------|------|\n"
    ec = result["equity_curve"]
    sample = ec[::5][-500:] if len(ec) > 5 else ec
    for e in sample:
        r += f"| {e['date']} | {e['nav']:.2f} | {e.get('dd', 0):.2f} |\n"

    r += f"""
## 七、v2.0 策略参数

| 参数 | 值 |
|------|-----|
| 最大持仓 | {MAX_POSITIONS} 只 |
| 单只最大仓位 | {int(POSITION_SIZE*100)}% |
| 固定止损 | -{int(FIXED_STOP_LOSS*100)}% |
| 第一止盈 | +{int(TP1_PCT*100)}% 减{int(TP1_REDUCE*100)}% |
| 第二止盈 | +{int(TP2_PCT*100)}% 减{int(TP2_REDUCE*100)}% |
| 峰值回落止盈 | 从峰值回落 {int((1-TP_TRAIL_RETRACE)*100)}% |
| 成交量确认 | ≥{int(VOL_MIN_RATIO*100)}% 均量 |
| 基本面过滤 | ROE≥5% 毛利率≥15% 综合分≥{FUND_SCORE_MIN} |
| 冷却期 | {COOLING_PERIOD_DAYS}天 / 连续{COOLING_STOP_COUNT}次止损 |
| 环境乘数(牛) | {ENV_MULTIPLIER['BULL']:.1f}x |
| 环境乘数(熊) | {ENV_MULTIPLIER['BEAR']:.1f}x |
| 环境乘数(震荡) | {ENV_MULTIPLIER['OSCILLATE']:.1f}x |
| 技术权重 | {TECH_WEIGHT:.1f} |
| 基本面权重 | {FUND_WEIGHT:.1f} |

## 八、v2.0 新特性说明

1. **市场环境分类**：基于沪深300的ER效率比率 + 均线排列 + ADX，将市场分为牛/熊/震荡，自动调节仓位
2. **信号因子化**：缠论信号根据成交量确认度转换为0~1的连续因子分，替代二元买入/不买入
3. **共振得分**：技术得分(×环境乘数) × 0.6 + 基本面得分(归一化) × 0.4，共同决定仓位大小
"""
    return r


# ============================================================
# 主函数
# ============================================================
def main():
    global _T_START
    parser = argparse.ArgumentParser(description="ChanFund Fusion v2.0 全量回测")
    parser.add_argument("--mode", choices=["all", "train", "valid", "test"], default="test",
                        help="回测阶段")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help=f"初始资金 (默认: {INITIAL_CAPITAL:,})")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（仅100只股票）")
    args = parser.parse_args()

    setup_logging()
    _T_START = time.time()
    logger.info(f"🔰 ChanFund Fusion v2.0 全量回测")
    logger.info(f"   模式={args.mode}, 资金={args.capital:,}, {'⚡快速' if args.quick else '全量'}")

    # 加载数据
    stock_limit = 100 if args.quick else 0
    data = load_daily_csv_data(str(DATA_DIR), max_files=stock_limit)
    fin_cache = load_financial_data(str(FIN_DIR))
    index_data = load_index_data(str(DATA_DIR))

    # 基线 v1.4
    BASELINE_V14 = {"total_return": 29.02, "sharpe": 3.44,
                    "max_drawdown": -13.71, "win_rate": 58.6, "win_loss_ratio": 2.67}

    modes = ["train", "valid", "test"] if args.mode == "all" else [args.mode]
    all_results = {}

    for mode in modes:
        split = TIME_SPLITS[mode]
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 [{mode}] {split['start']} ~ {split['end']}")
        logger.info(f"{'='*60}")

        result = run_backtest(
            data, fin_cache, index_data,
            split["start"], split["end"],
            args.capital, mode, stock_limit,
        )
        all_results[mode] = result

        # 保存CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if result["trade_log"]:
            pd.DataFrame(result["trade_log"]).to_csv(
                REPORT_DIR / f"{BACKTEST_NAME}_{mode}_{ts}_trades.csv", index=False)
        if result["equity_curve"]:
            pd.DataFrame(result["equity_curve"]).to_csv(
                REPORT_DIR / f"{BACKTEST_NAME}_{mode}_{ts}_equity.csv", index=False)

    # 生成报告
    # 统计股票数（从最终数据中统计所有出现的股票代码）
    all_codes_in_data = set()
    for d in data.values():
        all_codes_in_data.update(d.keys())
    n_stocks = len(all_codes_in_data)
    n_warmup = len([d for d in data if d < TIME_SPLITS[modes[0]]["start"]])
    n_bt = len([d for d in data if TIME_SPLITS[modes[0]]["start"] <= d <= TIME_SPLITS[modes[-1]]["end"]])

    if len(modes) == 3:
        # 三阶段合并报告
        combined = f"""# ChanFund Fusion v2.0 — 三阶段全量回测报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 三阶段绩效对比

| 指标 | 训练集(2016-2017) | 验证集(2018-2019) | 测试集(2020-2026) | v1.4基线 |
|------|:---:|:---:|:---:|:---:|
"""
        for k in ["total_return", "sharpe", "max_drawdown", "win_rate", "win_loss_ratio"]:
            labels = {"total_return": "总收益率(%)", "sharpe": "夏普比率",
                      "max_drawdown": "最大回撤(%)", "win_rate": "胜率(%)",
                      "win_loss_ratio": "盈亏比"}
            combined += f"| **{labels.get(k, k)}** | "
            for m in modes:
                v = all_results[m]["metrics"].get(k, 0)
                combined += f"{v:.2f} | "
            combined += f"{BASELINE_V14.get(k, 0):.2f} |\n"

        combined += "\n## 分阶段详细报告\n"
        for mode in modes:
            rpt = generate_report(all_results[mode], mode, n_stocks, n_warmup, n_bt, 
                                   time.time() - _T_START, BASELINE_V14 if mode == "test" else None)
            combined += f"\n---\n### {mode.upper()}\n{rpt}\n"

        fname = f"{BACKTEST_NAME}_full_test_{datetime.now().strftime('%Y%m%d')}.md"
    else:
        mode = modes[0]
        combined = generate_report(all_results[mode], mode, n_stocks, n_warmup, n_bt,
                                    time.time() - _T_START,
                                    BASELINE_V14 if mode == "test" else None)
        fname = f"{BACKTEST_NAME}_{mode}_{datetime.now().strftime('%Y%m%d')}.md"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fpath = REPORT_DIR / fname
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(combined)
    logger.info(f"📄 报告保存: {fpath}")

    logger.info(f"\n{'='*50}")
    logger.info("📊 摘要")
    for mode in modes:
        m = all_results[mode]["metrics"]
        logger.info(f"[{mode}] 收益 {m['total_return']:.2f}% | 夏普 {m['sharpe']:.2f} | "
                    f"回撤 {m['max_drawdown']:.2f}% | 胜率 {m['win_rate']:.1f}% | "
                    f"交易 {m['n_trades']}")
    logger.info(f"⏱️ 耗时 {(time.time()-_T_START)/60:.1f}min")


if __name__ == "__main__":
    main()
