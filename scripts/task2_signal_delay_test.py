#!/usr/bin/env python3
"""
Task 2: 信号延迟 / 真实可成交价格测试
====================================
将 v2.0 回测逐笔成交记录中的 BUY 成交价替换为次日开盘价，
按照规则重新计算每笔交易的盈亏及净值曲线。

规则：
- BUY: 成交价 → 次日开盘价
- SELL (止损): 价格触碰止损价 → 当日收盘价
- SELL (止盈 TP1/TP2/trailing_stop): 收盘后确认 → 次日开盘价
- 止损价基于新的买入成本重新计算
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
LOG_PATH = _WORKSPACE + "/task2_delay_test.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
)
logger = logging.getLogger("task2_delay")

# ---- 参数（与 v2.0 一致） ----
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
COOLING_PERIOD_DAYS = 10
COOLING_STOP_COUNT = 2


def load_daily_parquet(max_stocks: int = 0) -> dict:
    """加载日线数据 {date_str: {code: {open,high,low,close,volume}}}"""
    all_pq = Path(DATA_DIR) / "_all.parquet"
    if not all_pq.exists():
        logger.error(f"❌ Parquet文件不存在: {all_pq}")
        return {}

    logger.info(f"📂 加载Parquet: {all_pq}")
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
            if len(d_str) < 8:
                continue
            close = float(row["close"])
            if close <= 0:
                continue
            data.setdefault(d_str, {})[c] = {
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": close,
                "volume": float(row["volume"]), "amount": 0.0,
            }
    del all_df
    logger.info(f"✅ Parquet加载完成: {len(data)}天, {time.time()-t0:.0f}s")
    return data


def load_v2_trade_log() -> list:
    """加载已有的 v2.0 回测成交记录（从最近的回测CSV）"""
    report_dir = Path("/home/quant/backtest report")
    csv_files = sorted(report_dir.glob("ChanFund-Fusion_v2.0_*_trades.csv"))
    if not csv_files:
        # 尝试从 backtest_results 找
        for p in [Path(_WORKSPACE) / "backtest_results" / "V2.2" / "v2.2_optimal.out",
                   REPORT_DIR]:
            if p.exists():
                logger.info(f"Looking for CSV in: {p}")
    if csv_files:
        latest = csv_files[-1]
        logger.info(f"📄 加载成交记录: {latest}")
        df = pd.read_csv(latest)
        return df.to_dict("records")
    
    logger.warning("⚠️ 未找到已有成交CSV，尝试从备份日志重建...")
    # 从回测输出的摘要信息看，v2.0 结果存在，但逐笔CSV没有保存
    # 需要从回测运行日志中提取，或者重新跑一次快速回测获取
    return []


def get_next_trade_date(date_str: str, data: dict) -> Optional[str]:
    """获取下一交易日"""
    all_dates = sorted(data.keys())
    idx = next((i for i, d in enumerate(all_dates) if d > date_str), None)
    if idx is not None and idx < len(all_dates):
        return all_dates[idx]
    return None


def recalc_with_delayed_prices(trade_log: list, data: dict, initial_capital: float) -> dict:
    """
    使用延迟价格重新计算交易盈亏与净值曲线。
    
    规则：
    1. BUY: 成交价 → 次日开盘价 (next_open)
    2. SELL (stop_loss): 价格触碰止损 → 当日收盘价 (close)
    3. SELL (tp1/tp2/trailing_stop): 收盘确认 → 次日开盘价 (next_open)
    4. 止损价基于新的买入成本调整: new_stop = new_entry * (1 - FIXED_STOP_LOSS)
    
    返回同 v2.0 格式的 {metrics, trade_log, equity_curve}
    """
    # 用每日数据模拟净值曲线
    # 首先，重建持仓变化序列
    # 将原始 trade_log 按日期排序，并插入延迟价格
    
    delayed_log = []
    # 按股票+日期建立索引
    trades_by_date: Dict[str, list] = defaultdict(list)
    for t in trade_log:
        trades_by_date[t["date"]].append(t)
    
    all_dates = sorted(data.keys())
    
    # 状态追踪
    cash = initial_capital
    positions: Dict[str, dict] = {}
    equity_curve = []
    cooling: Dict[str, date] = {}
    exit_detail = defaultdict(int)
    
    for date_str in all_dates:
        trade_date = date.fromisoformat(date_str)
        day_data = data.get(date_str, {})
        
        # 处理买入
        if date_str in trades_by_date:
            for t in trades_by_date[date_str]:
                if t.get("action") == "BUY":
                    symbol = t["symbol"]
                    entry_price_orig = float(t["price"])
                    original_signal = t.get("signal", "")
                    
                    # 获取次日开盘价
                    next_date = get_next_trade_date(date_str, data)
                    if next_date is None or symbol not in data.get(next_date, {}):
                        # 无法获取次日数据，跳过
                        continue
                    
                    next_open = data[next_date][symbol]["open"]
                    if next_open <= 0:
                        continue
                    
                    if len(positions) >= MAX_POSITIONS:
                        continue
                    if symbol in positions:
                        continue
                    if symbol in cooling and trade_date <= cooling[symbol]:
                        continue
                    
                    # 用次日开盘价买入
                    entry_price = next_open
                    shares = int(INITIAL_CAPITAL * POSITION_SIZE / entry_price / 100) * 100
                    if shares <= 0:
                        continue
                    cost = shares * entry_price
                    if cost > cash:
                        continue
                    
                    stop_price = entry_price * (1 - FIXED_STOP_LOSS)
                    
                    positions[symbol] = {
                        "entry_price": entry_price,
                        "shares": shares,
                        "entry_date": date_str,
                        "stop_price": stop_price,
                        "peak_price": entry_price,
                        "signal_type": original_signal,
                        "tp1_triggered": False,
                        "tp2_triggered": False,
                    }
                    cash -= cost
                    
                    delayed_log.append({
                        "date": date_str, "symbol": symbol, "action": "BUY",
                        "price": round(entry_price, 3), "signal": original_signal,
                        "original_price": round(entry_price_orig, 3),
                        "sell_price_type": "next_open",
                    })
        
        # 处理卖出（止盈/止损）
        to_close = []
        to_reduce = []
        
        for symbol, pos in list(positions.items()):
            if symbol in day_data:
                price = day_data[symbol]["close"]
                open_price = day_data[symbol]["open"]
                pos["_last_price"] = price
            else:
                price = pos.get("_last_price", pos["entry_price"])
                open_price = price
            
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
            
            # 盘中止损（用最低价判断）
            low = day_data.get(symbol, {}).get("low", price)
            if low <= pos["stop_price"]:
                # 止损触发，用当日收盘价卖出
                to_close.append((symbol, "stop_loss", price, "close"))
                continue
            
            # 峰值回落止盈（收盘后确认）
            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= TP_TRAIL_RETRACE:
                    next_date = get_next_trade_date(date_str, data)
                    if next_date and symbol in data.get(next_date, {}):
                        sell_price = data[next_date][symbol]["open"]
                    else:
                        sell_price = price
                    to_close.append((symbol, "trailing_stop", sell_price, "next_open"))
                    continue
            
            # 第二止盈（收盘后确认）
            if pnl_pct >= TP2_PCT and not pos["tp2_triggered"]:
                pos["tp2_triggered"] = True
                next_date = get_next_trade_date(date_str, data)
                if next_date and symbol in data.get(next_date, {}):
                    sell_price = data[next_date][symbol]["open"]
                else:
                    sell_price = price
                to_reduce.append((symbol, "tp2", TP2_REDUCE, sell_price, "next_open"))
                continue
            
            # 第一止盈（收盘后确认）
            if pnl_pct >= TP1_PCT and not pos["tp1_triggered"]:
                pos["tp1_triggered"] = True
                next_date = get_next_trade_date(date_str, data)
                if next_date and symbol in data.get(next_date, {}):
                    sell_price = data[next_date][symbol]["open"]
                else:
                    sell_price = price
                to_reduce.append((symbol, "tp1", TP1_REDUCE, sell_price, "next_open"))
                continue
            
            # 止损检查（已在前面用low检查过）
        
        # 执行减仓
        for symbol, reason, reduce_pct, price, price_type in to_reduce:
            pos = positions[symbol]
            reduce_shares = int(pos["shares"] * reduce_pct)
            if reduce_shares > 0 and pos["shares"] > reduce_shares:
                pos["shares"] -= reduce_shares
                cash += reduce_shares * price
                pnl = (price / pos["entry_price"] - 1) * 100
                exit_detail[reason] += 1
                delayed_log.append({
                    "date": date_str, "symbol": symbol, "action": "SELL",
                    "price": round(price, 3), "reason": reason,
                    "pnl_pct": round(pnl, 2),
                    "hold_days": hold_days,
                    "sell_price_type": price_type,
                    "original_entry": round(pos["entry_price"], 3),
                })
        
        # 执行平仓
        for symbol, reason, price, price_type in to_close:
            if symbol not in positions:
                continue
            pos = positions.pop(symbol)
            cash += pos["shares"] * price
            pnl = (price / pos["entry_price"] - 1) * 100
            exit_detail[reason] += 1
            delayed_log.append({
                "date": date_str, "symbol": symbol, "action": "SELL",
                "price": round(price, 3), "reason": reason,
                "pnl_pct": round(pnl, 2),
                "hold_days": (trade_date - date.fromisoformat(pos["entry_date"])).days,
                "sell_price_type": price_type,
                "original_entry": round(pos["entry_price"], 3),
            })
            if reason == "stop_loss":
                consec = sum(1 for t in reversed(delayed_log) if t.get("symbol") == symbol and t.get("reason") == "stop_loss")
                if consec >= COOLING_STOP_COUNT:
                    cooling[symbol] = trade_date + timedelta(days=COOLING_PERIOD_DAYS)
        
        # 计算净值
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
    for symbol, pos in list(positions.items()):
        close_price = pos.get("_last_price", pos["entry_price"])
        cash += pos["shares"] * close_price
        pnl = (close_price / pos["entry_price"] - 1) * 100
        delayed_log.append({
            "date": all_dates[-1], "symbol": symbol, "action": "SELL",
            "price": round(close_price, 3), "reason": "force_close",
            "pnl_pct": round(pnl, 2), "hold_days": 0,
            "sell_price_type": "close",
        })
    positions.clear()
    
    # 计算指标
    metrics = calc_metrics(delayed_log, equity_curve, initial_capital)
    
    return {
        "metrics": metrics,
        "trade_log": delayed_log,
        "equity_curve": equity_curve,
        "exit_stats": dict(exit_detail),
    }


def calc_metrics(trade_log: list, equity_curve: list, initial_capital: float) -> dict:
    """同 v2.0 calc_metrics"""
    if not equity_curve:
        return {"final_equity": initial_capital, "total_return": 0, "ann_return": 0,
                "sharpe": 0, "max_drawdown": 0, "win_rate": 0, "avg_win": 0,
                "avg_loss": 0, "win_loss_ratio": 0, "best_trade": 0, "worst_trade": 0, "n_trades": 0}
    
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
        "final_equity": equity_curve[-1]["nav"],
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


def main():
    logger.info("=" * 60)
    logger.info("🔰 Task 2: 信号延迟 / 真实可成交价格测试")
    logger.info("=" * 60)
    
    # 加载数据
    data = load_daily_parquet(max_stocks=4441)
    if not data:
        logger.error("❌ 数据加载失败")
        return
    
    # 从保存的CSV中加载（已在之前v2.0回测时保存）
    csv_files = sorted(Path("/home/quant/backtest report").glob("ChanFund-Fusion_v2.0_*_trades.csv"))
    equity_files = sorted(Path("/home/quant/backtest report").glob("ChanFund-Fusion_v2.0_*_equity.csv"))
    
    if not csv_files:
        logger.error("❌ 未找到成交CSV，需要先跑一次 v2.0 全量回测")
        return None
    
    latest_csv = csv_files[-1]
    logger.info(f"✅ 加载成交CSV: {latest_csv.name}")
    trade_log_df = pd.read_csv(latest_csv)
    trade_log = trade_log_df.to_dict("records")
    
    v2_equity = None
    if equity_files:
        latest_eq = equity_files[-1]
        v2_eq_df = pd.read_csv(latest_eq)
        v2_equity = v2_eq_df.to_dict("records")
    
    if not trade_log:
        logger.error("❌ 无法获取成交记录")
        return
    
    logger.info(f"📊 原始成交记录: {len(trade_log)} 条")
    
    # 重新计算延迟价格版本
    logger.info("⚙️ 正在计算延迟成交价格...")
    result = recalc_with_delayed_prices(trade_log, data, INITIAL_CAPITAL)
    
    m = result["metrics"]
    
    logger.info(f"\n{'='*55}")
    logger.info("📊 Task 2 结果")
    logger.info(f"{'='*55}")
    logger.info(f"  v2.0 (原始): 收益 8.27% | 夏普 -0.16 | 回撤 -11.23%")
    logger.info(f"  延迟成交后:  收益 {m['total_return']:.2f}% | 夏普 {m['sharpe']:.2f} | "
                f"回撤 {m['max_drawdown']:.2f}% | 交易 {m['n_trades']}")
    delta = m['total_return'] - 8.27
    logger.info(f"  收益差: {delta:+.2f}% {'⚠️ 超过2%阈值, 前视偏差严重' if abs(delta) > 2 else '✅ 在2%范围内'}")
    
    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORT_DIR / f"task2_delay_{ts}_trades.csv"
    pd.DataFrame(result["trade_log"]).to_csv(csv_path, index=False)
    csv_eq = REPORT_DIR / f"task2_delay_{ts}_equity.csv"
    pd.DataFrame(result["equity_curve"]).to_csv(csv_eq, index=False)
    
    # 保存指标摘要
    summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
    summary_path = REPORT_DIR / f"task2_delay_{ts}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ 成交CSV: {csv_path}")
    logger.info(f"✅ 净值CSV: {csv_eq}")
    logger.info(f"✅ 指标摘要: {summary_path}")
    
    return result


if __name__ == "__main__":
    result = main()
    if result:
        m = result["metrics"]
        v2_orig = {"total_return": 8.27, "sharpe": -0.16, "max_drawdown": -11.23, "win_rate": 64.2, "n_trades": 497}
        logger.info(f"\n{'='*55}")
        logger.info("  Task 2 完成 — 对比摘要")
        logger.info(f"{'='*55}")
        logger.info(f"  {'指标':<20} {'v2.0原始':<15} {'延迟成交':<15} {'变化':<15}")
        logger.info(f"  {'-'*65}")
        for k in ["total_return", "ann_return", "sharpe", "max_drawdown", "win_rate", "n_trades"]:
            v1 = v2_orig.get(k, 0)
            v2 = round(m.get(k, 0), 2)
            delta = round(v2 - v1, 2) if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else 0
            label = {"total_return": "总收益率(%)", "ann_return": "年化收益率(%)",
                     "sharpe": "夏普比率", "max_drawdown": "最大回撤(%)",
                     "win_rate": "胜率(%)", "n_trades": "交易次数"}.get(k, k)
            logger.info(f"  {label:<20} {v1:<15} {v2:<15} {delta:+.2f}")
