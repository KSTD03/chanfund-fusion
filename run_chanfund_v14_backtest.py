"""
ChanFund Fusion v1.0.1 — 全量回测运行器
=====================================
全股票池(4441只A股) × 全时间段(2010-01 ~ 2026-04)
数据来源: daily_csv (已校准)

后台运行: nohup python3 run_chanfund_backtest.py > backtest_v101.out 2>&1 &
"""

from __future__ import annotations

import os, sys, time, json, re
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
BACKTEST_NAME = "ChanFund-Fusion_v1.4"
STRATEGY_DIR = Path(_WORKSPACE) / "chanfund_fusion"
REPORT_DIR = Path("/home/quant/backtest report")
DATA_DIR = Path(_WORKSPACE) / "quant" / "data" / "daily_csv"
FIN_DIR = Path(_WORKSPACE) / "quant" / "data" / "financial"

BACKTEST_RANGE = {
    "warmup_start": "2018-01-01",
    "start": "2021-01-01",
    "end": "2026-04-29",
}

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5
POSITION_SIZE = 0.20
COMMISSION_RATE = 0.0003

FIXED_STOP_LOSS = 0.08
TP1_PCT = 0.10
TP1_REDUCE = 0.30
TP2_PCT = 0.20
TP2_REDUCE = 0.50
TP_TRAIL_RETRACE = 0.90
TP_BREAKEVEN_LOCK = 0.05
VOL_MIN_RATIO = 0.8

# 【v1.4】基本面过滤参数
FUND_SCORE_MIN = 60       # 最小基本面分(0-100)，低于此分不买入
FUND_ROE_MIN = 0.05       # ROE ≥ 5%
FUND_GM_MIN = 0.15        # 毛利率 ≥ 15%
FUND_REV_GROWTH_MIN = -0.20  # 营收增长 ≥ -20%（容忍下滑但不严重）

LOG_PATH = _WORKSPACE + "/backtest_v14_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("chanfund_v14_bt")


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


def load_financial_data(fin_dir: str) -> dict:
    """【v1.4】加载财务数据，返回 PIT 缓存

    Returns:
        {report_date_str: {stock_code_6digits: {field: value}}}
    """
    fin_csvs = sorted(Path(fin_dir).glob("financial_*.csv"))
    logger.info(f"📊 加载 {len(fin_csvs)} 期财务数据...")

    fin_cache = {}  # {report_date_str: {code: {field: value}}}

    for fpath in fin_csvs:
        try:
            df = pd.read_csv(fpath, dtype={"股票代码": str})
        except Exception as e:
            logger.warning(f"  Skip {fpath.name}: {e}")
            continue

        # 从文件名提取报告期
        match = re.search(r"(\d{8})", fpath.stem)
        if not match:
            continue
        report_date = match.group(1)[:4] + "-" + match.group(1)[4:6] + "-" + match.group(1)[6:]

        period_data = {}
        for _, row in df.iterrows():
            code = str(row["股票代码"]).zfill(6)[:6]

            # 提取关键财务字段
            eps = row.get("每股收益", None)
            rev = row.get("营业总收入-营业总收入", None)
            rev_growth = row.get("营业总收入-同比增长", None)
            profit = row.get("净利润-净利润", None)
            profit_growth = row.get("净利润-同比增长", None)
            bvps = row.get("每股净资产", None)
            roe = row.get("净资产收益率", None)
            ocf_ps = row.get("每股经营现金流量", None)
            gross_margin = row.get("销售毛利率", None)

            period_data[code] = {
                "eps": eps,
                "revenue": rev,
                "rev_growth": rev_growth,
                "profit": profit,
                "profit_growth": profit_growth,
                "bvps": bvps,
                "roe": roe,
                "ocf_ps": ocf_ps,
                "gross_margin": gross_margin,
                "report_date": report_date,
            }

        fin_cache[report_date[:7]] = period_data  # 用 "2024-12" 作为key
        logger.info(f"  ✅ {fpath.name}: {len(period_data)} 只 {report_date[:7]}")

    logger.info(f"✅ 财务数据加载完成: {len(fin_cache)} 期")
    return fin_cache


def get_fund_score(stock_code: str, trade_date_str: str, fin_cache: dict) -> float:
    """【v1.4】获取股票在指定日期的最新基本面分(0-100)"""
    if not fin_cache:
        return 100.0  # 无财务数据时默认通过

    # 找最新可用的财报（trade_date 之前的最近一期）
    year = int(trade_date_str[:4])
    month = int(trade_date_str[5:7])

    # 财报季映射：Q1=3月, Q2=6月, Q3=9月, Q4=12月
    report_keys = []
    for y in range(year, year - 3, -1):
        for m in ["-12", "-09", "-06", "-03"]:
            rk = f"{y}{m}"
            if rk in fin_cache:
                report_keys.append(rk)

    code_6 = stock_code.split(".")[-1][:6]

    for rk in report_keys:
        stock_fin = fin_cache[rk].get(code_6)
        if stock_fin is None:
            continue

        roe = stock_fin.get("roe")
        gm = stock_fin.get("gross_margin")
        rev_g = stock_fin.get("rev_growth")
        eps = stock_fin.get("eps")
        ocf = stock_fin.get("ocf_ps")

        score = 50.0  # 基础分

        # ROE评分
        if roe is not None and not np.isnan(roe):
            if roe >= 20:
                score += 20
            elif roe >= 10:
                score += 15
            elif roe >= 5:
                score += 10
            elif roe >= 0:
                score += 5
            else:
                score -= 10

        # 毛利率评分
        if gm is not None and not np.isnan(gm):
            if gm >= 40:
                score += 15
            elif gm >= 25:
                score += 10
            elif gm >= 15:
                score += 5
            else:
                score -= 5

        # 营收增长评分
        if rev_g is not None and not np.isnan(rev_g):
            if rev_g >= 20:
                score += 10
            elif rev_g >= 10:
                score += 7
            elif rev_g >= 0:
                score += 3
            elif rev_g >= -20:
                score -= 5
            else:
                score -= 15

        # EPS评分
        if eps is not None and not np.isnan(eps):
            if eps > 0:
                score += 5
            else:
                score -= 10

        # OCF评分
        if ocf is not None and not np.isnan(ocf):
            if ocf > 0:
                score += 5
            else:
                score -= 5

        return max(0, min(100, score))

    return 70.0  # 无财务数据时给中等分（不歧视新股）


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


def generate_report(metrics, signal_stats, exit_stats, tp_stats, trade_log,
                    equity_curve, total_stocks, warmup_dates, backtest_dates, elapsed,
                    fin_enabled=False):
    """生成Markdown报告 v1.4"""
    from collections import Counter
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fin_status = "✅ 启用" if fin_enabled else "❌ 无数据"

    report = f"""# ChanFund Fusion v1.4 — 全量回测报告

> 生成时间: {now}
> 策略版本: v1.4 (2026-05-03 · 新增基本面过滤)
> 股票池: {total_stocks} 只 | 回测区间: {backtest_dates[0] if backtest_dates else 'N/A'} ~ {backtest_dates[-1] if backtest_dates else 'N/A'}
> 交易日: {len(backtest_dates)} 天 | 耗时: {elapsed:.1f}秒

## 一、v1.4 优化特性启用状态

| 特性 | 状态 | 说明 |
|------|------|------|
| **基本面过滤** | ✅ **v1.4** | ROE+毛利率+营收增长+EPS+OCF综合评分≥{FUND_SCORE_MIN}分 |
| 多级止盈系统 | ✅ v1.1 | 10%/20%分批减仓+峰值回落清仓+保本移损 |
| 成交量确认过滤 | ✅ v1.1 | ≥20日均量×{VOL_MIN_RATIO*100:.0f}% |
| 固定8%止损 | ✅ v1.1 | 统一{FIXED_STOP_LOSS*100:.0f}%硬止损+保本移损 |
| 集中持仓 | ✅ v1.1 | 最大{MAX_POSITIONS}只×{POSITION_SIZE*100:.0f}% |
| ER噪声过滤 | ✅ 继承 | 分级降权 |
| 冷却期机制 | ✅ 继承 | 连续2次止损→10日冷却 |
| 趋势方向过滤 | ✅ 继承 | strict(EMA12/26+MA120) |
| 信号失效机制 | ✅ 继承 | 结构破坏标记作废 |

## 二、绩效概览

| 指标 | 数值 |
|------|------|
| 初始资金 | {INITIAL_CAPITAL:,} |
| 最终净值 | {equity_curve[-1]['nav']:.0f} |
| **总收益率** | **{metrics.get('total_return_pct', 'N/A')}%** |
| **夏普比率** | **{metrics.get('sharpe_ratio', 'N/A')}** |
| **最大回撤** | **{metrics.get('max_drawdown_pct', 'N/A')}%** |
| 总交易数 | {metrics.get('total_trades', 0)} |
| 买入笔数 | {metrics.get('buy_trades', 0)} |
| 卖出笔数 | {metrics.get('sell_trades', 0)} |

### 盈利分析

| 指标 | 数值 |
|------|------|
| 盈利交易 | {metrics.get('num_wins', 0)} ({metrics.get('win_rate', 0)}%) |
| 亏损交易 | {metrics.get('num_losses', 0)} ({(100-metrics.get('win_rate', 0)):.1f}%) |
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

    total_exits = sum(exit_stats.values())
    report += "\n## 四、平仓原因分布\n\n| 原因 | 次数 | 占比 |\n|------|:----:|:----:|\n"
    for reason, count in sorted(exit_stats.items(), key=lambda x: -x[1]):
        pct = count / max(total_exits, 1) * 100
        report += f"| {reason} | {count} | {pct:.1f}% |\n"

    report += "\n### 止盈分布\n\n| 止盈类型 | 次数 |\n|----------|:----:|\n"
    for t, c in sorted(tp_stats.items(), key=lambda x: -x[1]):
        report += f"| {t} | {c} |\n"

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
    logger.info(f"  🚀 ChanFund Fusion v1.4 — 全量回测开始")
    logger.info(f"  区间: {BACKTEST_RANGE['start']} ~ {BACKTEST_RANGE['end']}")
    logger.info(f"  数据: {DATA_DIR}")
    logger.info("=" * 70)

    # 1. 加载数据
    cached_data = load_daily_csv_data(str(DATA_DIR))
    all_dates = sorted(cached_data.keys())
    warmup_dates = [d for d in all_dates if d < BACKTEST_RANGE["start"]]
    backtest_dates = [d for d in all_dates if BACKTEST_RANGE["start"] <= d <= BACKTEST_RANGE["end"]]
    logger.info(f"📅 预热: {len(warmup_dates)}天, 回测: {len(backtest_dates)}天")

    all_stocks = set()
    for date_data in cached_data.values():
        all_stocks.update(date_data.keys())
    all_stocks = sorted(all_stocks)
    logger.info(f"📊 总股票数: {len(all_stocks)} 只")

    # 2. 【v1.4】加载财务数据
    fin_cache = load_financial_data(str(FIN_DIR))
    fin_enabled = len(fin_cache) > 0
    logger.info(f"✅ 基本面过滤: {'启用(' + str(len(fin_cache)) + '期)' if fin_enabled else '未启用'}")

    # 3. 初始化策略引擎
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

    # ===== 阶段2: 正式回测 (v1.4) =====
    logger.info("=" * 60)
    logger.info("PHASE 2: Backtest v1.4")
    logger.info("=" * 60)

    trade_log = []
    equity_curve = []
    signal_stats = defaultdict(int)
    exit_stats = defaultdict(int)
    tp_stats = defaultdict(int)  # 止盈分布
    positions = {}
    cash = INITIAL_CAPITAL
    peak_equity = cash
    max_dd = 0.0
    total_signals = 0

    # 成交量缓存
    vol_cache = {}

    for day_idx, date_str in enumerate(backtest_dates):
        day_data = cached_data.get(date_str, {})
        trade_date = date.fromisoformat(date_str)

        # A. 更新技术引擎 + 成交量缓存
        for stock, kbar in day_data.items():
            try:
                tech_engine.update(stock, kbar, emit_signals=True)
            except Exception:
                pass
            v = kbar.get("volume", 0)
            if v > 0:
                if stock not in vol_cache:
                    vol_cache[stock] = []
                vol_cache[stock].append(v)
                if len(vol_cache[stock]) > 50:
                    vol_cache[stock] = vol_cache[stock][-50:]

        # B. 获取确认信号
        signals = tech_engine.get_confirmed_signals(trade_date)
        total_signals += len(signals)

        # C. v1.4: 基本面过滤 + 成交量过滤 + 执行买入
        for sig in signals:
            if len(positions) >= MAX_POSITIONS:
                break
            if sig.symbol in positions:
                continue
            if sig.symbol not in day_data:
                continue

            # 【v1.4】基本面过滤
            if fin_enabled:
                fund_score = get_fund_score(sig.symbol, date_str, fin_cache)
                if fund_score < FUND_SCORE_MIN:
                    continue

            # 成交量过滤
            vols = vol_cache.get(sig.symbol, [])
            if len(vols) >= 20:
                avg_vol = sum(vols[-20:]) / 20
                entry_vol = day_data[sig.symbol].get("volume", 0)
                if avg_vol > 0 and entry_vol < avg_vol * VOL_MIN_RATIO:
                    continue

            entry_price = day_data[sig.symbol].get("close", 0)
            if entry_price <= 0:
                continue

            cost = cash * POSITION_SIZE
            if cost > cash:
                continue

            shares = cost / entry_price
            # v1.1: 浮动止损 = 8%（简化版引擎无需ATR）
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

        # D. v1.1: 止盈/止损检查
        to_close = []
        to_reduce = []  # (stock, reason, reduce_pct)
        current_nav = cash

        for stock, pos in list(positions.items()):
            kbar = day_data.get(stock, {})
            price = kbar.get("close", pos["entry_price"])
            current_nav += pos["shares"] * price

            # 更新峰值
            pos["peak_price"] = max(pos["peak_price"], price)

            # v1.1: 计算持有天数
            hold_days = (trade_date - date.fromisoformat(pos["entry_date"])).days
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            peak_pnl = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]

            # v1.1: 保本移损（+5%后止损移入成本）
            if pnl_pct >= TP_BREAKEVEN_LOCK and pos["stop_price"] < pos["entry_price"]:
                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])

            # v1.1: 长持仓保护
            if hold_days >= 60 and abs(pnl_pct) < 0.05:
                wide_stop = pos["entry_price"] * 0.85
                if wide_stop < pos["stop_price"]:
                    pos["stop_price"] = wide_stop

            # v1.1: 峰值回落止盈
            if peak_pnl >= TP_BREAKEVEN_LOCK:
                retrace = price / pos["peak_price"]
                if retrace <= TP_TRAIL_RETRACE:
                    to_close.append((stock, f"峰值回落{int((1-TP_TRAIL_RETRACE)*100)}%止盈", price))
                    tp_stats["峰值回落止盈"] += 1
                    continue

            # v1.1: 第二止盈
            if pnl_pct >= TP2_PCT and not pos["tp2_triggered"]:
                pos["tp2_triggered"] = True
                to_reduce.append((stock, f"第二止盈+{int(TP2_PCT*100)}%减{int(TP2_REDUCE*100)}%", TP2_REDUCE, price))
                tp_stats["第二止盈"] += 1
                continue

            # v1.1: 第一止盈
            if pnl_pct >= TP1_PCT and not pos["tp1_triggered"]:
                pos["tp1_triggered"] = True
                to_reduce.append((stock, f"第一止盈+{int(TP1_PCT*100)}%减{int(TP1_REDUCE*100)}%", TP1_REDUCE, price))
                tp_stats["第一止盈"] += 1
                continue

            # 止损检查
            if price <= pos["stop_price"]:
                to_close.append((stock, "stop_loss", price))

        # E. v1.1: 执行减仓（部分止盈）
        for stock, reason, reduce_pct, price in to_reduce:
            pos = positions[stock]
            reduce_shares = pos["shares"] * reduce_pct
            pos["shares"] -= reduce_shares
            cash += reduce_shares * price
            pnl_pct = round((price / pos["entry_price"] - 1) * 100, 2)
            hold_days = (trade_date - date.fromisoformat(pos["entry_date"])).days
            exit_stats[reason] += 1
            trade_log.append({
                "date": date_str, "symbol": stock, "action": "SELL",
                "price": round(price, 3), "reason": reason,
                "pnl_pct": pnl_pct, "hold_days": hold_days,
            })

        # F. 执行平仓
        for stock, reason, price in to_close:
            pos = positions.pop(stock)
            cash += pos["shares"] * price
            pnl_pct = round((price / pos["entry_price"] - 1) * 100, 2)
            exit_stats[reason] += 1
            hold_days = (trade_date - date.fromisoformat(pos["entry_date"])).days
            trade_log.append({
                "date": date_str, "symbol": stock, "action": "SELL",
                "price": round(price, 3), "reason": reason,
                "pnl_pct": pnl_pct, "signal_type": pos["signal_type"],
                "hold_days": hold_days,
            })

        # G. 记录净值
        pos_value = sum(p["shares"] * day_data.get(s, {}).get("close", p["entry_price"])
                        for s, p in positions.items())
        equity = cash + pos_value
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        max_dd = min(max_dd, dd)

        if day_idx % 5 == 0:
            equity_curve.append({"date": date_str, "nav": round(equity, 2)})

        if (day_idx + 1) % 50 == 0:
            logger.info(
                f"  📊 {date_str} ({day_idx+1}/{len(backtest_dates)}) — "
                f"NAV={equity:.0f}, holdings={len(positions)}, "
                f"trades={len(trade_log)}, dd={dd:.1f}%"
            )

    # 强制平仓剩余持仓
    for stock, pos in list(positions.items()):
        price = pos["entry_price"]
        cash += pos["shares"] * price
        trade_log.append({
            "date": backtest_dates[-1], "symbol": stock, "action": "SELL",
            "price": round(price, 3), "reason": "强制平仓",
            "pnl_pct": 0.0, "hold_days": 0,
        })
    positions.clear()

    elapsed = time.time() - start_time

    # ===== 阶段3: 报告 =====
    logger.info("=" * 60)
    logger.info("PHASE 3: Generate Report")
    logger.info("=" * 60)

    metrics = calc_metrics(trade_log, equity_curve, INITIAL_CAPITAL)
    metrics["max_drawdown_pct"] = round(max_dd, 2)

    report = generate_report(
        metrics, dict(signal_stats), dict(exit_stats), dict(tp_stats),
        trade_log, equity_curve, len(all_stocks),
        warmup_dates, backtest_dates, elapsed,
        fin_enabled=fin_enabled,
    )

    report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ChanFund-Fusion_v1.4_full_{report_date}.md"
    report_path = REPORT_DIR / report_filename
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    if trade_log:
        csv_path = REPORT_DIR / f"ChanFund-Fusion_v1.4_full_{report_date}_trades.csv"
        pd.DataFrame(trade_log).to_csv(csv_path, index=False, encoding="utf-8-sig")

    logger.info(f"✅ 报告: {report_path}")
    print(f"""
{'='*55}
  🎯 ChanFund Fusion v1.4 回测完成
{'='*55}
  总收益率: {metrics.get('total_return_pct', 'N/A')}%
  夏普比率: {metrics.get('sharpe_ratio', 'N/A')}
  最大回撤: {metrics.get('max_drawdown_pct', 'N/A')}%
  总交易数: {metrics.get('total_trades', 0)}
  胜率: {metrics.get('win_rate', 0)}%
  盈利/亏损: {metrics.get('num_wins', 0)}/{metrics.get('num_losses', 0)}
  总信号: {total_signals}
  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)
  报告: {report_path}
{'='*55}""")


if __name__ == "__main__":
    main()
