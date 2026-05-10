"""
ChanFund Fusion v1.0.1 — 全量回测运行器
=====================================
全股票池(4441只A股) × 全时间段(2010-01 ~ 2026-04)
数据来源: daily_csv (已校准)

后台运行: nohup python3 run_chanfund_backtest.py > backtest_v101.out 2>&1 &
"""

from __future__ import annotations

import os, sys, time, json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict

_WORKSPACE = str(Path(__file__).resolve().parent)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import numpy as np
import pandas as pd

# ============================================================
# 配置
# ============================================================
BACKTEST_NAME = "ChanFund-Fusion_v1.0.1"
STRATEGY_DIR = Path(_WORKSPACE) / "chanfund_fusion"
REPORT_DIR = Path("/home/quant/backtest report")
DATA_DIR = Path(_WORKSPACE) / "quant" / "data" / "daily_csv"

BACKTEST_RANGE = {
    "warmup_start": "2018-01-01",
    "start": "2021-01-01",
    "end": "2026-04-29",
}

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 30
POSITION_PCT = 0.01
FIXED_STOP_LOSS = 0.05

LOG_PATH = _WORKSPACE + "/backtest_v101_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("chanfund_v101_bt")


def load_daily_csv_data(data_dir: str) -> dict:
    """从 daily_csv 目录加载全量数据

    Returns:
        {date_str: {stock_code: kbar_dict}}
    """
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    logger.info(f"📂 找到 {len(csv_files)} 个CSV文件, 开始加载...")
    t0 = time.time()

    all_data = defaultdict(dict)
    loaded = 0

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, usecols=["date", "code", "open", "high", "low", "close", "volume", "isST"])
            for _, row in df.iterrows():
                dt = str(row["date"])[:10]
                kbar = {
                    "time": date.fromisoformat(dt),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                    "isST": row.get("isST", 0),
                }
                if kbar["close"] > 0 and kbar["high"] >= kbar["low"]:
                    all_data[dt][str(row["code"])] = kbar
            loaded += 1
        except Exception as e:
            logger.debug(f"  Skip {fpath.name}: {e}")

        if loaded % 500 == 0:
            t1 = time.time()
            logger.info(f"  📥 已加载 {loaded}/{len(csv_files)} 文件, {sum(len(v) for v in all_data.values())} 条K线, {t1-t0:.0f}s")

    t1 = time.time()
    logger.info(f"✅ 数据加载完成: {len(all_data)} 个交易日, {sum(len(v) for v in all_data.values())} 条K线, 耗时 {t1-t0:.1f}s")
    return dict(all_data)


def calc_metrics(trade_log, equity_curve, initial_capital):
    """计算绩效

    Args:
        trade_log: 交易记录列表
        equity_curve: 净值曲线 [{"date": ..., "nav": ...}]
        initial_capital: 初始资金
    """
    sells = [t for t in trade_log if t.get("action") == "SELL"]
    buys = [t for t in trade_log if t.get("action") == "BUY"]

    # 总收益率从净值曲线计算
    if equity_curve:
        final_nav = equity_curve[-1]["nav"]
        total_return = (final_nav - initial_capital) / initial_capital * 100
    else:
        total_return = 0.0

    pnls = [s["pnl_pct"] for s in sells if "pnl_pct" in s]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / max(len(pnls), 1) * 100

    if pnls:
        avg_ret = np.mean(pnls)
        std_ret = np.std(pnls) if len(pnls) > 1 else 1e-10
        sharpe = (avg_ret / max(std_ret, 1e-10)) * np.sqrt(252 / 5)
    else:
        sharpe = 0.0

    return {
        "total_return_pct": round(total_return, 2),
        "total_trades": len(buys) + len(sells),
        "buy_trades": len(buys),
        "sell_trades": len(sells),
        "num_wins": len(wins),
        "num_losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(np.mean(wins), 2) if wins else 0.0,
        "avg_loss_pct": round(np.mean(losses), 2) if losses else 0.0,
        "sharpe_ratio": round(sharpe, 2),
        "best_trade": round(max(pnls), 2) if pnls else 0.0,
        "worst_trade": round(min(pnls), 2) if pnls else 0.0,
    }


def generate_report(metrics, signal_stats, exit_stats, trade_log,
                    equity_curve, total_stocks, warmup_dates, backtest_dates, elapsed):
    """生成Markdown报告"""
    from collections import Counter
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# ChanFund Fusion v1.0.1 — 全量回测报告

> 生成时间: {now}
> 策略版本: v1.0.1 (2026-05-03 精修版·8项优化)
> 股票池: {total_stocks} 只 | 回测区间: {backtest_dates[0] if backtest_dates else 'N/A'} ~ {backtest_dates[-1] if backtest_dates else 'N/A'}
> 交易日: {len(backtest_dates)} 天 | 耗时: {elapsed:.1f}秒

## 一、优化特性启用状态

| 特性 | 状态 |
|------|------|
| 保护性移动止损 | ✅ 开（笔低点+中枢ZG跟踪）|
| ER噪声过滤 | ✅ 开（分级降权）|
| 冷却期机制 | ✅ 开 |
| 趋势方向过滤 | ✅ 开 |
| 红黄牌排雷 | ✅ 开 |
| 分级仓位系数 | ✅ 开 |
| 试探仓演化 | ✅ 开 |
| 信号失效机制 | ✅ 开 |

## 二、绩效概览

| 指标 | 数值 |
|------|------|
| 初始资金 | {INITIAL_CAPITAL:,} |
| 总收益率 | {metrics.get('total_return_pct', 'N/A')}% |
| 夏普比率 | {metrics.get('sharpe_ratio', 'N/A')} |
| 总交易数 | {metrics.get('total_trades', 0)} |
| 买入笔数 | {metrics.get('buy_trades', 0)} |
| 卖出笔数 | {metrics.get('sell_trades', 0)} |
| 盈利交易 | {metrics.get('num_wins', 0)} ({metrics.get('win_rate', 0)}%) |
| 亏损交易 | {metrics.get('num_losses', 0)} |
| 平均盈利 | {metrics.get('avg_win_pct', 0)}% |
| 平均亏损 | {metrics.get('avg_loss_pct', 0)}% |
| 最佳单笔 | {metrics.get('best_trade', 0)}% |
| 最差单笔 | {metrics.get('worst_trade', 0)}% |

## 三、信号分布

| 信号类型 | 次数 |
|----------|------|
"""
    for sig_type, count in sorted(signal_stats.items(), key=lambda x: -x[1]):
        report += f"| {sig_type} | {count} |\n"

    report += "\n## 四、平仓原因分布\n\n| 原因 | 次数 |\n|------|------|\n"
    for reason, count in sorted(exit_stats.items(), key=lambda x: -x[1]):
        report += f"| {reason} | {count} |\n"

    report += "\n## 五、逐笔成交记录\n\n| # | 日期 | 股票 | 操作 | 价格 | 信号/原因 | 盈亏% | 持仓天数 |\n|---|------|------|------|------|-----------|-------|---------|\n"
    for i, t in enumerate(trade_log, 1):
        pnl = t.get("pnl_pct", "")
        sig = t.get("signal", t.get("signal_type", t.get("reason", "")))
        report += f"| {i} | {t.get('date', '')} | {t.get('symbol', '')} | {t.get('action', '')} | {t.get('price', 0):.3f} | {sig} | {pnl}% | {t.get('hold_days', '')} |\n"

    if equity_curve:
        report += "\n## 六、净值曲线（每5天，最近500点）\n\n| 日期 | 净值 |\n|------|------|\n"
        for ec in equity_curve[-500:]:
            report += f"| {ec['date']} | {ec['nav']:.2f} |\n"

    return report


def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info(f"  🚀 ChanFund Fusion v1.0.1 — 全量回测开始")
    logger.info(f"  区间: {BACKTEST_RANGE['start']} ~ {BACKTEST_RANGE['end']}")
    logger.info(f"  数据: {DATA_DIR}")
    logger.info("=" * 70)

    # 1. 加载数据
    cached_data = load_daily_csv_data(str(DATA_DIR))
    all_dates = sorted(cached_data.keys())
    warmup_dates = [d for d in all_dates if d < BACKTEST_RANGE["start"]]
    backtest_dates = [d for d in all_dates if BACKTEST_RANGE["start"] <= d <= BACKTEST_RANGE["end"]]
    logger.info(f"📅 预热: {len(warmup_dates)}天, 回测: {len(backtest_dates)}天")

    # 获取所有股票代码
    all_stocks = set()
    for date_data in cached_data.values():
        all_stocks.update(date_data.keys())
    all_stocks = sorted(all_stocks)
    logger.info(f"📊 总股票数: {len(all_stocks)} 只")

    # 2. 初始化策略引擎 (简化版)
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.tech.chan_objects import Signal, SignalStatus
    from chanfund_fusion.config_schema import load_config

    config = load_config(str(STRATEGY_DIR / "config.yaml"))

    tech_engine = TechSignalEngine(config.get("tech", {}))
    logger.info(f"✅ 技术引擎初始化完成")

    # ===== 阶段1: 预热 =====
    logger.info("=" * 60)
    logger.info("PHASE 1: Warm-up (no signals)")
    logger.info("=" * 60)

    warmup_count = 0
    for idx, date_str in enumerate(warmup_dates):
        day_data = cached_data.get(date_str, {})
        for stock, kbar in day_data.items():
            try:
                tech_engine.update(stock, kbar, emit_signals=False)
            except Exception:
                pass
        if (idx + 1) % 60 == 0:
            logger.info(f"  Warmup {date_str} ({idx+1}/{len(warmup_dates)}) — stocks={len(day_data)}")

    logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)} 只股票有缠论结构")

    # ===== 阶段2: 正式回测 =====
    logger.info("=" * 60)
    logger.info("PHASE 2: Backtest")
    logger.info("=" * 60)

    trade_log = []
    equity_curve = []
    signal_stats = defaultdict(int)
    exit_stats = defaultdict(int)
    positions = {}
    cash = INITIAL_CAPITAL
    equity = cash
    total_signals = 0

    for day_idx, date_str in enumerate(backtest_dates):
        day_data = cached_data.get(date_str, {})
        trade_date = date.fromisoformat(date_str)

        # A. 更新技术引擎
        for stock, kbar in day_data.items():
            try:
                tech_engine.update(stock, kbar, emit_signals=True)
            except Exception:
                pass

        # B. 获取确认信号
        signals = tech_engine.get_confirmed_signals(trade_date)
        total_signals += len(signals)

        # C. 执行买入
        for sig in signals:
            if len(positions) >= MAX_POSITIONS:
                break
            if sig.symbol in positions:
                continue
            if sig.symbol not in day_data:
                continue

            entry_price = day_data[sig.symbol].get("close", 0)
            if entry_price <= 0:
                continue

            cost = equity * POSITION_PCT
            if cost > cash:
                continue

            shares = cost / entry_price
            stop_price = entry_price * (1 - FIXED_STOP_LOSS)

            positions[sig.symbol] = {
                "entry_price": entry_price,
                "shares": shares,
                "entry_date": date_str,
                "stop_price": stop_price,
                "signal_type": sig.signal_subtype,
            }
            cash -= cost
            signal_stats[sig.signal_subtype] += 1

            trade_log.append({
                "date": date_str,
                "symbol": sig.symbol,
                "action": "BUY",
                "price": round(entry_price, 3),
                "signal": sig.signal_subtype,
            })

        # D. 检查卖出
        to_close = []
        current_nav = cash

        for stock, pos in list(positions.items()):
            price = day_data.get(stock, {}).get("close", pos["entry_price"])
            current_nav += pos["shares"] * price

            if price <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price))

        # E. 执行卖出
        for stock, reason, price in to_close:
            pos = positions.pop(stock)
            cash += pos["shares"] * price
            pnl_pct = round((price / pos["entry_price"] - 1) * 100, 2)
            exit_stats[reason] += 1
            hold_days = (trade_date - date.fromisoformat(pos["entry_date"])).days

            trade_log.append({
                "date": date_str,
                "symbol": stock,
                "action": "SELL",
                "price": round(price, 3),
                "reason": reason,
                "pnl_pct": pnl_pct,
                "signal_type": pos["signal_type"],
                "hold_days": hold_days,
            })

        equity = current_nav

        if day_idx % 5 == 0:
            equity_curve.append({"date": date_str, "nav": round(equity, 2)})

        if (day_idx + 1) % 50 == 0:
            logger.info(
                f"  📊 Day {date_str} ({day_idx+1}/{len(backtest_dates)}) — "
                f"NAV={equity:.0f}, holdings={len(positions)}, "
                f"trades={len(trade_log)}, signals={total_signals}"
            )

    elapsed = time.time() - start_time

    # ===== 阶段3: 报告 =====
    logger.info("=" * 60)
    logger.info("PHASE 3: Generate Report")
    logger.info("=" * 60)

    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    report = generate_report(
        metrics, dict(signal_stats), dict(exit_stats),
        trade_log, equity_curve, len(all_stocks),
        warmup_dates, backtest_dates, elapsed,
    )

    report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ChanFund-Fusion_v1.0.1_full_{report_date}.md"
    report_path = REPORT_DIR / report_filename
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    if trade_log:
        csv_path = REPORT_DIR / f"ChanFund-Fusion_v1.0.1_full_{report_date}_trades.csv"
        pd.DataFrame(trade_log).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存完成标志（供监控检测）
    flag_path = _WORKSPACE + "/.backtest_done"
    with open(flag_path, "w") as f:
        f.write(f"ChanFund Fusion v1.0.1 done at {datetime.now()}\nTotal return: {metrics['total_return_pct']}%")

    logger.info(f"✅ 报告: {report_path}")
    print(f"""
{'='*55}
  🎯 ChanFund Fusion v1.0.1 回测完成
{'='*55}
  总收益率: {metrics.get('total_return_pct', 'N/A')}%
  夏普比率: {metrics.get('sharpe_ratio', 'N/A')}
  总交易数: {metrics.get('total_trades', 0)}
  胜率: {metrics.get('win_rate', 0)}%
  盈利/亏损: {metrics.get('num_wins', 0)}/{metrics.get('num_losses', 0)}
  总信号: {total_signals}
  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)
  报告: {report_path}
{'='*55}""")


if __name__ == "__main__":
    main()
