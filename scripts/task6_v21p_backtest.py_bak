#!/usr/bin/env python3
"""
Task 3: v2.1 全量回测 — 全量基本面评分接入
===========================================
核心改动：将原来的 fund_score 默认值 55 替换为 scores.parquet 中的真实评分。
技术信号、风控参数、止盈止损保持 v2.0 不变。

回测窗口：TEST 期（2020-01 至 2026-04）
股票池：4441 只
约束：所有买入使用次日开盘价，卖出按规则区分。
"""

import os, sys, time, json, copy, argparse
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
LOG_PATH = _WORKSPACE + "/task3_v21_backtest.log"

_T_START = 0
logger = logging.getLogger("task3_v21")

# ============================================================
# 配置（与 v2.0 一致，仅 FUND_WEIGHT 调整以用真实基本面评分）
# ============================================================
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5
POSITION_SIZE = 0.20
FIXED_STOP_LOSS = 0.08
TP1_PCT = 0.06
TP1_REDUCE = 0.30
TP2_PCT = 0.13
TP2_REDUCE = 0.50
TP_TRAIL_RETRACE = 0.91
TP_BREAKEVEN_LOCK = 0.05
VOL_MIN_RATIO = 0.8
COOLING_PERIOD_DAYS = 10
COOLING_STOP_COUNT = 2

# v2.0 参数
SIGNAL_BASE_SCORE = {
    "third_point_buy": 0.70, "hard_divergence": 0.85,
    "second_class_buy": 0.65, "soft_divergence": 0.50,
}
SIGNAL_VOL_ADJUST = 0.20
TECH_WEIGHT = 0.6
FUND_WEIGHT = 0.4
FUND_DEFAULT_SCORE = 50.0   # 无数据时用中性分（与v2.0的55不同）
FUND_SCORE_MIN = 30.0       # v2.2 优化后值（原v2.0是50）

# Test 期
TEST_START = "2020-01-02"
TEST_END = "2026-04-29"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
    )


def load_data(max_stocks: int = 0) -> dict:
    """加载日线 Parquet 数据"""
    all_pq = Path(DATA_DIR) / "_all.parquet"
    if not all_pq.exists():
        logger.error(f"❌ 无数据: {all_pq}")
        return {}

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
    logger.info(f"✅ 数据: {len(data)}天, {time.time()-t0:.0f}s")
    return data


def load_scores_parquet() -> dict:
    """
    加载 scores.parquet 转为 {code: {end_date: fund_score}} 格式
    用于替代 v2.0 的 fin_cache 基本面评分
    """
    sp = Path(_WORKSPACE) / "financial_data" / "scores.parquet"
    if not sp.exists():
        logger.error(f"❌ scores.parquet 不存在: {sp}")
        return {}
    
    t0 = time.time()
    df = pd.read_parquet(sp)
    logger.info(f"✅ scores.parquet: {len(df)} 行, {df['ts_code'].nunique()} 只股票")
    
    scores_cache: dict = {}
    for _, row in df.iterrows():
        code_str = str(row["ts_code"])
        if not (code_str.startswith("sh.") or code_str.startswith("sz.")):
            code_str = f"sh.{code_str}" if code_str[0] == '6' else f"sz.{code_str}"
        end_date = row["end_date"]  # 已经是 str
        score = float(row["fund_score"])
        if score > 0:
            scores_cache.setdefault(code_str, {})[end_date] = score
    
    logger.info(f"✅ scores索引: {len(scores_cache)} 只股票, {time.time()-t0:.1f}s")
    return scores_cache


def get_fund_score_v21(code: str, date_str: str, scores_cache: dict) -> float:
    """
    从 scores.parquet 获取基本面评分。
    查询 <= date_str 的最新一个 end_date 的 score。
    如果无数据，返回 FUND_DEFAULT_SCORE (50)。
    """
    if code not in scores_cache or not scores_cache[code]:
        return FUND_DEFAULT_SCORE
    
    dates = sorted(scores_cache[code].keys())
    best = None
    for d in reversed(dates):
        if d <= date_str:
            best = d
            break
    if best is None:
        return FUND_DEFAULT_SCORE
    
    return scores_cache[code][best]


def get_next_trade_date(date_str: str, all_dates: list) -> Optional[str]:
    idx = next((i for i, d in enumerate(all_dates) if d > date_str), None)
    return all_dates[idx] if idx is not None and idx < len(all_dates) else None


def signal_to_factor(signal_type: str, vol_ratio: float) -> float:
    base = SIGNAL_BASE_SCORE.get(signal_type, 0.5)
    vc = min(vol_ratio, 2.0) / 2.0
    adj = (vc - 0.5) * SIGNAL_VOL_ADJUST * 2
    return min(1.0, max(0.0, base + adj))


def resonance_score_v21(tech_factor: float, fund_score: float) -> float:
    """共振得分 = 技术分×0.6 + 归一化基本面分×0.4"""
    return min(1.0, tech_factor * TECH_WEIGHT + (fund_score / 100.0) * FUND_WEIGHT)


def calc_metrics(trade_log: list, equity_curve: list, initial_capital: float) -> dict:
    if not equity_curve:
        return {"final_equity": initial_capital, "total_return": 0, "ann_return": 0,
                "sharpe": 0, "max_drawdown": 0, "win_rate": 0, "avg_win": 0,
                "avg_loss": 0, "win_loss_ratio": 0, "best_trade": 0, "worst_trade": 0,
                "n_trades": 0, "monthly_returns": {}, "yearly_returns": {}}
    
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
    
    # 月度和年度收益率
    monthly = {}
    yearly = {}
    if equity_curve:
        prev_date = equity_curve[0]["date"]
        prev_nav = initial_capital
        for e in equity_curve[1:]:
            ym = e["date"][:7]
            y = e["date"][:4]
            ret = (e["nav"] - prev_nav) / prev_nav
            monthly[ym] = monthly.get(ym, 1.0) * (1 + ret) - 1 if ym != prev_date[:7] else monthly.get(ym, 0) + ret
            yearly[y] = yearly.get(y, 1.0) * (1 + ret) - 1 if y != prev_date[:4] else yearly.get(y, 0) + ret
            prev_date = e["date"]
            prev_nav = e["nav"]
    
    return {
        "final_equity": round(equity_curve[-1]["nav"], 2) if equity_curve else initial_capital,
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
        "monthly_returns": monthly,
        "yearly_returns": yearly,
    }


def run_backtest_v21(data: dict, scores_cache: dict, all_dates: list,
                     start: str, end: str, initial_capital: float = INITIAL_CAPITAL,
                     max_stocks: int = 0, use_skip_list: bool = False,
                     skip_list: set = None) -> dict:
    """
    v2.1 核心回测（使用 scores.parquet 真实基本面评分）
    所有买入使用次日开盘价
    
    返回: {metrics, trade_log, equity_curve, signal_stats, exit_stats, tp_stats, monthly_returns}
    """
    global _T_START
    
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    
    cfg = load_config(str(Path(_WORKSPACE) / "chanfund_fusion" / "config.yaml"))
    tech_engine = TechSignalEngine(cfg.get("tech", {}))
    
    warmup_dates = [d for d in all_dates if d < start]
    backtest_dates = [d for d in all_dates if start <= d <= end]
    
    all_stocks = sorted(set().union(*(d.keys() for d in data.values())))
    if max_stocks > 0:
        all_stocks = all_stocks[:max_stocks]
    if skip_list:
        all_stocks = [s for s in all_stocks if s not in skip_list]
    
    logger.info(f"📊 股票池: {len(all_stocks)} 只, 预热 {len(warmup_dates)}天, 回测 {len(backtest_dates)}天")
    
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
    
    n_cached = len(tech_engine.chan_cache)
    logger.info(f"✅ 预热完成, {n_cached} 只股票有缠论结构")
    
    # ---- 状态 ----
    cash = initial_capital
    peak_equity = cash
    positions: Dict[str, dict] = {}
    equity_curve = []
    cooling: Dict[str, date] = {}
    trade_log = []
    vol_cache: Dict[str, list] = {}
    signal_stats = defaultdict(int)
    exit_detail = defaultdict(int)
    tp_detail = defaultdict(int)
    
    # ---- Phase 2: Backtest ----
    logger.info("Phase 2: Backtest v2.1")
    
    for di, date_str in enumerate(backtest_dates):
        trade_date = date.fromisoformat(date_str)
        day_data = data.get(date_str, {})
        if not day_data: continue
        
        # A. 更新技术引擎
        for stock in all_stocks:
            if stock in day_data:
                try: tech_engine.update(stock, day_data[stock], emit_signals=True)
                except: pass
                v = day_data[stock].get("volume", 0)
                if v > 0:
                    vol_cache.setdefault(stock, []).append(v)
                    if len(vol_cache[stock]) > 50:
                        vol_cache[stock] = vol_cache[stock][-50:]
        
        # B. 获取信号
        signals = tech_engine.get_confirmed_signals(trade_date)
        
        for sig in signals:
            if len(positions) >= MAX_POSITIONS: break
            if sig.symbol in positions: continue
            if sig.symbol in cooling and trade_date <= cooling[sig.symbol]: continue
            if sig.symbol not in day_data: continue
            
            # 成交价：次日开盘价  --------------------------------------------
            next_date = get_next_trade_date(date_str, all_dates)
            if next_date is None or sig.symbol not in data.get(next_date, {}):
                continue
            entry_price = data[next_date][sig.symbol]["open"]
            if entry_price <= 0: continue
            
            # 基本面评分：scores.parquet  ------------------------------------
            fund_score = get_fund_score_v21(sig.symbol, date_str, scores_cache)
            if fund_score < FUND_SCORE_MIN: continue
            
            # 成交量过滤
            vols = vol_cache.get(sig.symbol, [])
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                entry_vol = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and entry_vol < avg_vol * VOL_MIN_RATIO: continue
            
            # 因子化 + 共振
            vols_20 = vol_cache.get(sig.symbol, [])
            vol_ratio = 1.0
            if len(vols_20) >= 20:
                mv = sum(vols_20[-20:]) / 20
                vol_ratio = day_data[sig.symbol].get("volume", 0) / mv if mv > 0 else 1.0
            tech_factor = signal_to_factor(sig.signal_subtype, vol_ratio)
            res = resonance_score_v21(tech_factor, fund_score)
            
            # 仓位
            position_coeff = min(res, 1.0) * POSITION_SIZE / 0.20
            position_coeff = min(1.0, max(0.0, position_coeff))
            cost = cash * POSITION_SIZE * position_coeff
            if cost > cash or cost <= 0: continue
            shares = int(cost / entry_price / 100) * 100
            if shares <= 0: continue
            cost = shares * entry_price
            if cost > cash or cost <= 0: continue
            
            stop_price = entry_price * (1 - FIXED_STOP_LOSS)
            positions[sig.symbol] = {
                "entry_price": entry_price, "shares": shares,
                "entry_date": date_str,
                "buy_exec_date": next_date,  # 实际成交日
                "stop_price": stop_price, "peak_price": entry_price,
                "signal_type": sig.signal_subtype,
                "tp1_triggered": False, "tp2_triggered": False,
                "fund_score": fund_score, "resonance": res,
            }
            cash -= cost
            signal_stats[sig.signal_subtype] += 1
            
            trade_log.append({
                "date": date_str, "symbol": sig.symbol, "action": "BUY",
                "price": round(entry_price, 3), "signal": sig.signal_subtype,
                "fund_score": fund_score, "resonance": round(res, 3),
                "buy_price_type": "next_open",
            })
        
        # C. 止盈/止损
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
            if hold_days >= 60 and abs(pnl_pct) < 0.05:
                pos["stop_price"] = min(pos["stop_price"], pos["entry_price"] * 0.85)
            
            # 盘中止损
            low = day_data.get(stock, {}).get("low", price)
            if low <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price, "close"))
                continue
            
            # 峰值回落（收盘确认 → 次日开盘卖出）
            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= TP_TRAIL_RETRACE:
                    nd = get_next_trade_date(date_str, all_dates)
                    sp = data[nd][stock]["open"] if nd and stock in data.get(nd, {}) else price
                    to_close.append((stock, "trailing_stop", sp, "next_open"))
                    tp_detail["trailing_stop"] += 1
                    continue
            
            # TP2（收盘确认 → 次日开盘卖出）
            if pnl_pct >= TP2_PCT and not pos["tp2_triggered"]:
                pos["tp2_triggered"] = True
                nd = get_next_trade_date(date_str, all_dates)
                sp = data[nd][stock]["open"] if nd and stock in data.get(nd, {}) else price
                to_reduce.append((stock, "tp2", TP2_REDUCE, sp, "next_open"))
                tp_detail["tp2"] += 1
                continue
            
            # TP1（收盘确认 → 次日开盘卖出）
            if pnl_pct >= TP1_PCT and not pos["tp1_triggered"]:
                pos["tp1_triggered"] = True
                nd = get_next_trade_date(date_str, all_dates)
                sp = data[nd][stock]["open"] if nd and stock in data.get(nd, {}) else price
                to_reduce.append((stock, "tp1", TP1_REDUCE, sp, "next_open"))
                tp_detail["tp1"] += 1
                continue
        
        # 执行减仓
        for sym, reason, rpct, sp, pt in to_reduce:
            pos = positions[sym]
            r_shares = int(pos["shares"] * rpct)
            if r_shares > 0 and pos["shares"] > r_shares:
                pos["shares"] -= r_shares
                cash += r_shares * sp
                pnl = (sp / pos["entry_price"] - 1) * 100
                exit_detail[reason] += 1
                trade_log.append({
                    "date": date_str, "symbol": sym, "action": "SELL",
                    "price": round(sp, 3), "reason": reason,
                    "pnl_pct": round(pnl, 2),
                    "hold_days": (trade_date - date.fromisoformat(pos["entry_date"])).days,
                    "sell_price_type": pt,
                })
        
        # 执行平仓
        for sym, reason, sp, pt in to_close:
            if sym not in positions: continue
            pos = positions.pop(sym)
            cash += pos["shares"] * sp
            pnl = (sp / pos["entry_price"] - 1) * 100
            exit_detail[reason] += 1
            trade_log.append({
                "date": date_str, "symbol": sym, "action": "SELL",
                "price": round(sp, 3), "reason": reason,
                "pnl_pct": round(pnl, 2),
                "hold_days": (trade_date - date.fromisoformat(pos["entry_date"])).days,
                "sell_price_type": pt,
            })
            if reason == "stop_loss":
                consec = sum(1 for t in reversed(trade_log) if t.get("symbol") == sym and t.get("reason") == "stop_loss")
                if consec >= COOLING_STOP_COUNT:
                    cooling[sym] = trade_date + timedelta(days=COOLING_PERIOD_DAYS)
        
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
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        equity_curve.append({"date": date_str, "nav": round(equity, 2), "dd": round(dd, 2)})
        
        if (di+1) % 200 == 0:
            elapsed = time.time() - _T_START
            logger.info(f"  [{date_str}] ({di+1}/{len(backtest_dates)}) NAV={equity:,.0f} pos={len(positions)} trades={len(trade_log)} dd={dd:.1f}% t={elapsed:.0f}s")
    
    # 强制平仓
    for stock, pos in list(positions.items()):
        cp = pos.get("_last_price", pos["entry_price"])
        cash += pos["shares"] * cp
        pnl = (cp / pos["entry_price"] - 1) * 100
        trade_log.append({
            "date": backtest_dates[-1], "symbol": stock, "action": "SELL",
            "price": round(cp, 3), "reason": "force_close", "pnl_pct": round(pnl, 2),
            "sell_price_type": "close",
        })
    positions.clear()
    
    metrics = calc_metrics(trade_log, equity_curve, initial_capital)
    
    logger.info(f"✅ v2.1 完成! 收益 {metrics['total_return']:.2f}% | 夏普 {metrics['sharpe']:.2f} | "
                f"回撤 {metrics['max_drawdown']:.2f}% | 交易 {metrics['n_trades']}")
    
    return {
        "metrics": metrics,
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "signal_stats": dict(signal_stats),
        "exit_stats": dict(exit_detail),
        "tp_stats": dict(tp_detail),
    }


def main():
    parser = argparse.ArgumentParser(description="Task 3: v2.1 全量基本面回测")
    parser.add_argument("--quick", action="store_true", help="快速模式（500只）")
    parser.add_argument("--max-stocks", type=int, default=0, help="股票数限制")
    parser.add_argument("--batch", action="store_true", help="分批次模式")
    args = parser.parse_args()
    
    global _T_START
    _T_START = time.time()
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("🔰 Task 3: v2.1 全量回测 — 全量基本面评分接入")
    logger.info(f"   模式={'⚡快速(500只)' if args.quick else '全量(4441只)'} "
                f"{'分批次' if args.batch else ''}")
    logger.info("=" * 60)
    
    # 1. 加载数据
    n_stocks = 500 if args.quick else 0
    data = load_data(max_stocks=n_stocks)
    if not data:
        logger.error("❌ 数据加载失败")
        return
    
    # 2. 加载 scores.parquet
    scores_cache = load_scores_parquet()
    if not scores_cache:
        logger.error("❌ scores.parquet 加载失败")
        return
    
    all_dates = sorted(data.keys())
    
    # 3. 运行回测
    max_s = 500 if args.quick else 0
    result = run_backtest_v21(data, scores_cache, all_dates,
                              TEST_START, TEST_END, INITIAL_CAPITAL, max_s)
    
    m = result["metrics"]
    elapsed = time.time() - _T_START
    
    logger.info(f"\n{'='*55}")
    logger.info("📊 v2.1 回测结果")
    logger.info(f"{'='*55}")
    logger.info(f"  最终净值: {m['final_equity']:,.0f}")
    logger.info(f"  总收益率: {m['total_return']:.2f}%")
    logger.info(f"  年化收益率: {m['ann_return']:.2f}%")
    logger.info(f"  夏普比率: {m['sharpe']:.2f}")
    logger.info(f"  最大回撤: {m['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {m['win_rate']:.1f}%")
    logger.info(f"  盈亏比: {m['win_loss_ratio']:.2f}")
    logger.info(f"  交易次数: {m['n_trades']}")
    logger.info(f"  ⏱️ 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    
    # 逐年收益
    logger.info(f"\n  逐年收益:")
    for y in sorted(m.get("yearly_returns", {}).keys()):
        logger.info(f"    {y}: {m['yearly_returns'][y]*100:.2f}%")
    
    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_quick{'_'+str(max_s) if max_s else ''}"
    
    csv_path = REPORT_DIR / f"task3_v21{suffix}_{ts}_trades.csv"
    pd.DataFrame(result["trade_log"]).to_csv(csv_path, index=False)
    csv_eq = REPORT_DIR / f"task3_v21{suffix}_{ts}_equity.csv"
    pd.DataFrame(result["equity_curve"]).to_csv(csv_eq, index=False)
    
    summary = {
        "metrics": {k: v for k, v in m.items() if k not in ("monthly_returns", "yearly_returns")},
        "signal_stats": result["signal_stats"],
        "exit_stats": result["exit_stats"],
        "tp_stats": result["tp_stats"],
        "params": {"n_stocks": max_s or len(data), "start": TEST_START, "end": TEST_END,
                   "use_scores_parquet": True, "buy_next_open": True},
        "elapsed_s": round(elapsed),
    }
    sum_path = REPORT_DIR / f"task3_v21{suffix}_{ts}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"✅ 成交CSV: {csv_path}")
    logger.info(f"✅ 净值CSV: {csv_eq}")
    logger.info(f"✅ 摘要: {sum_path}")
    
    # 对比 v2.0
    logger.info(f"\n{'='*55}")
    logger.info("  vs v2.0 (原始)")
    logger.info(f"{'='*55}")
    v2 = {"total_return": 8.27, "ann_return": 1.32, "sharpe": -0.16,
          "max_drawdown": -11.23, "win_rate": 64.2, "n_trades": 497}
    for k in ["total_return", "ann_return", "sharpe", "max_drawdown", "win_rate", "n_trades"]:
        v1 = v2.get(k, 0)
        v2v = round(m.get(k, 0), 2)
        d = round(v2v - v1, 2)
        logger.info(f"  {k:<20} v2.0={v1:<10} v2.1={v2v:<10} Δ={d:+.2f}")


if __name__ == "__main__":
    main()
