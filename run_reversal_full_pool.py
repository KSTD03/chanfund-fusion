#!/usr/bin/env python3
"""
超跌反转策略 — 全股票池全时间段回测 (v2 - 优化版)
==============================================
全量A股(4441只) × 2019~2026
两阶段：①逐股预计算 → ②事件驱动回测
"""

import os, sys, json, time, math, logging, signal
import numpy as np
import pandas as pd
from datetime import datetime, date
from collections import defaultdict, OrderedDict

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
DATA_DIR = os.path.join(WORKSPACE, "quant/data/daily_csv")
REPORT_DIR = "/home/quant/backtest report"
os.makedirs(REPORT_DIR, exist_ok=True)

LOG_PATH = os.path.join(WORKSPACE, "reversal_full_pool.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("reversal_bt")

# ════════════════════════════════════════════
# 策略参数
# ════════════════════════════════════════════
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5
POSITION_SIZE = 0.20
COMMISSION_RATE = 0.0003

PRICE_LOOKBACKS = [252, 504, 756]
CEILING_RATIOS = [0.5, 0.6, 0.7]
MIN_AVG_AMOUNT = 30_000_000
CONSOL_WINDOW = 60
MAX_AMPLITUDE = 0.15
VOLUME_BREAK_RATIO = 1.2
FAST_MA = 5
SLOW_MA = 20
CONFIRM_DAYS = 2
HARD_STOP = 0.08
TP1, TP1_REDUCE = 0.15, 0.30
TP2, TP2_REDUCE = 0.25, 0.40
START_DATE = "2019-01-01"
END_DATE = "2026-04-29"


# ════════════════════════════════════════════
# Phase 1: 单股预处理 + 提取关键行
# ════════════════════════════════════════════
def preprocess_stock(code, df):
    """预处理单只股票，返回(全量df, 候选日索引)"""
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
    df = df[mask].copy()
    if len(df) < 252:
        return None, None

    # --- 价格位置 ---
    df["in_price_zone"] = False
    for lb in PRICE_LOOKBACKS:
        high_n = df["high"].rolling(lb, min_periods=lb // 2).max()
        for cr in CEILING_RATIOS:
            df["in_price_zone"] = df["in_price_zone"] | (df["close"] / high_n.replace(0, np.nan) <= cr)

    # 三年高点
    df["high_3y"] = df["high"].rolling(756, min_periods=504).max()

    # --- 流动性 + 横盘 ---
    df["avg_amount_20"] = df["amount"].rolling(20).mean()
    df["enough_liquidity"] = df["avg_amount_20"] >= MIN_AVG_AMOUNT

    bt = df["high"].rolling(CONSOL_WINDOW, min_periods=CONSOL_WINDOW // 2).max()
    bb = df["low"].rolling(CONSOL_WINDOW, min_periods=CONSOL_WINDOW // 2).min()
    df["box_amplitude"] = (bt - bb) / bb.replace(0, np.nan)
    df["is_consolidating"] = df["box_amplitude"] <= MAX_AMPLITUDE
    df["box_top"] = bt
    df["box_bottom"] = bb
    df["box_mid"] = (bt + bb) / 2

    # --- 突破/回升 ---
    avg_vol = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / avg_vol.replace(0, np.nan)
    df["break_above"] = df["close"] > df["box_top"]
    df["break_below"] = df["close"] < df["box_bottom"]
    df["above_mid"] = df["close"] >= df["box_mid"]
    bh = (df["box_top"] - df["box_bottom"]).replace(0, np.nan)
    df["box_recovery"] = (df["close"] - df["box_bottom"]) / bh

    # --- 均线 ---
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["golden_cross"] = (df["ma5"] > df["ma20"]) & (df["ma5"].shift(1) <= df["ma20"].shift(1))
    df["ma5_above_ma20"] = df["ma5"] > df["ma20"]
    df["ma20_direction"] = df["ma20"].diff(3) > 0
    df["below_ma20"] = df["close"] < df["ma20"]

    # --- 压力位 ---
    pz = df["high_3y"] * 0.60
    df["hit_pressure"] = df["close"] >= pz

    return df, None


def extract_ohlcv(df):
    """从全量df提取精简OHLCV数据"""
    keep = ["date", "close", "high", "low", "volume", "amount"]
    result = df[keep].copy()
    result["code"] = df.get("code", "")
    return result


# ════════════════════════════════════════════
# Phase 2: 事件驱动回测
# ════════════════════════════════════════════
class Position:
    __slots__ = ("code", "volume", "cost_price", "current_price", "peak_price",
                 "entry_date", "entry_box_top", "entry_box_bottom",
                 "added", "reduced", "days_held", "entry_reason")
    def __init__(self, code, price, volume, date_str, bt, bb, reason=""):
        self.code = code
        self.volume = volume
        self.cost_price = price
        self.current_price = price
        self.peak_price = price
        self.entry_date = date_str
        self.entry_box_top = bt
        self.entry_box_bottom = bb
        self.added = False
        self.reduced = False
        self.days_held = 0
        self.entry_reason = reason

    @property
    def cost_basis(self):
        return self.volume * self.cost_price

    @property
    def market_value(self):
        return self.volume * self.current_price

    @property
    def unrealized_pnl_pct(self):
        return (self.current_price / self.cost_price - 1) * 100


def run_backtest_efficient(all_data, signal_map):
    """
    高效回测：
    - all_data: {code: DataFrame} 全量指标数据
    - signal_map: {date_str: [(code, entry_strength, reason), ...]} 预计算候选
    """
    t0 = time.time()

    # 构建交易日历
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df["date"].dt.strftime("%Y-%m-%d").values)
    sorted_dates = sorted(all_dates)
    logger.info(f"📅 总交易日: {len(sorted_dates)}")

    capital = INITIAL_CAPITAL
    positions = OrderedDict()
    peak_equity = INITIAL_CAPITAL
    max_dd = 0.0
    trades = []
    equity_curve = []
    total_candidates = 0

    logger.info("=" * 60)
    logger.info("PHASE 2: 回测开始")
    logger.info("=" * 60)

    for day_idx, date_str in enumerate(sorted_dates):
        # 1. 更新持仓
        pos_value = 0
        for code in list(positions.keys()):
            pos = positions[code]
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df["date"] == date_str]
            if row.empty:
                continue
            r = row.iloc[0]
            pos.current_price = float(r["close"])
            pos.peak_price = max(pos.peak_price, float(r["close"]))
            pos.days_held += 1
            pos_value += pos.market_value

        # 2. 止损检查
        for code in list(positions.keys()):
            pos = positions[code]
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df["date"] == date_str]
            if row.empty:
                continue
            r = row.iloc[0]
            close = float(r["close"])
            stop_reason = None

            if r["break_below"] and not pd.isna(r["box_bottom"]):
                stop_reason = f"箱体破位止损(跌破{r['box_bottom']:.2f})"
            elif pos.unrealized_pnl_pct <= -HARD_STOP * 100:
                stop_reason = f"固定比例止损(-{HARD_STOP*100:.0f}%)"
            elif r["below_ma20"] and pos.unrealized_pnl_pct <= -3:
                stop_reason = "跌破MA20且亏损止损"

            if stop_reason:
                sell_amt = pos.volume * close
                comm = max(sell_amt * COMMISSION_RATE, 5)
                pnl = sell_amt - pos.cost_basis
                capital += sell_amt - comm
                trades.append({
                    "date": date_str, "code": code, "action": "SELL",
                    "price": round(close, 3), "volume": pos.volume,
                    "reason": stop_reason,
                    "pnl_pct": round(pos.unrealized_pnl_pct, 2),
                    "pnl": round(pnl, 2), "hold_days": pos.days_held,
                    "trade_type": "close",
                    "entry_date": pos.entry_date, "entry_price": round(pos.cost_price, 3),
                })
                del positions[code]
                continue

        # 3. 止盈检查
        for code in list(positions.keys()):
            pos = positions[code]
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df["date"] == date_str]
            if row.empty:
                continue
            r = row.iloc[0]
            close = float(r["close"])
            profit_pct = pos.unrealized_pnl_pct

            if profit_pct >= TP2 * 100:
                if pos.reduced:
                    # 已减过 → 全清
                    sell_amt = pos.volume * close
                    comm = max(sell_amt * COMMISSION_RATE, 5)
                    pnl = sell_amt - pos.cost_basis
                    capital += sell_amt - comm
                    trades.append({
                        "date": date_str, "code": code, "action": "SELL",
                        "price": round(close, 3), "volume": pos.volume,
                        "reason": f"第二止盈+{TP2*100:.0f}%清仓",
                        "pnl_pct": round(profit_pct, 2), "pnl": round(pnl, 2),
                        "hold_days": pos.days_held, "trade_type": "close",
                        "entry_date": pos.entry_date, "entry_price": round(pos.cost_price, 3),
                    })
                    del positions[code]
                    continue
                else:
                    reduce_vol = int(pos.volume * TP2_REDUCE / 100) * 100
                    if reduce_vol > 0:
                        sell_amt = reduce_vol * close
                        comm = max(sell_amt * COMMISSION_RATE, 5)
                        pnl = sell_amt - (reduce_vol * pos.cost_price)
                        capital += sell_amt - comm
                        pos.volume -= reduce_vol
                        pos.reduced = True
                        trades.append({
                            "date": date_str, "code": code, "action": "SELL",
                            "price": round(close, 3), "volume": reduce_vol,
                            "reason": f"第二止盈+{TP2*100:.0f}%减{TP2_REDUCE*100:.0f}%",
                            "pnl_pct": round(profit_pct, 2), "pnl": round(pnl, 2),
                            "hold_days": pos.days_held, "trade_type": "reduce",
                            "entry_date": pos.entry_date, "entry_price": round(pos.cost_price, 3),
                        })
                    continue

            if not pos.reduced and profit_pct >= TP1 * 100:
                reduce_vol = int(pos.volume * TP1_REDUCE / 100) * 100
                if reduce_vol > 0:
                    sell_amt = reduce_vol * close
                    comm = max(sell_amt * COMMISSION_RATE, 5)
                    pnl = sell_amt - (reduce_vol * pos.cost_price)
                    capital += sell_amt - comm
                    pos.volume -= reduce_vol
                    pos.reduced = True
                    trades.append({
                        "date": date_str, "code": code, "action": "SELL",
                        "price": round(close, 3), "volume": reduce_vol,
                        "reason": f"第一止盈+{TP1*100:.0f}%减{TP1_REDUCE*100:.0f}%",
                        "pnl_pct": round(profit_pct, 2), "pnl": round(pnl, 2),
                        "hold_days": pos.days_held, "trade_type": "reduce",
                        "entry_date": pos.entry_date, "entry_price": round(pos.cost_price, 3),
                    })

        # 4. 终极离场
        for code in list(positions.keys()):
            pos = positions[code]
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df["date"] == date_str]
            if row.empty:
                continue
            r = row.iloc[0]
            close = float(r["close"])
            exit_reason = None
            if r["below_ma20"]:
                exit_reason = "跌破20日线趋势走弱清仓"
            elif r["hit_pressure"]:
                exit_reason = "压力位全部止盈"
            if exit_reason and pos.volume > 0:
                sell_amt = pos.volume * close
                comm = max(sell_amt * COMMISSION_RATE, 5)
                pnl = sell_amt - pos.cost_basis
                capital += sell_amt - comm
                trades.append({
                    "date": date_str, "code": code, "action": "SELL",
                    "price": round(close, 3), "volume": pos.volume,
                    "reason": exit_reason,
                    "pnl_pct": round(pos.unrealized_pnl_pct, 2),
                    "pnl": round(pnl, 2), "hold_days": pos.days_held,
                    "trade_type": "close",
                    "entry_date": pos.entry_date, "entry_price": round(pos.cost_price, 3),
                })
                del positions[code]

        # 5. 加仓
        for code in list(positions.keys()):
            pos = positions[code]
            if pos.added or pos.reduced:
                continue
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df["date"] == date_str]
            if row.empty:
                continue
            r = row.iloc[0]
            if pos.days_held >= CONFIRM_DAYS and r["above_mid"] and r["ma20_direction"] and r["ma5_above_ma20"]:
                add_value = capital * 0.15
                add_vol = int(add_value / float(r["close"]) / 100) * 100
                if add_vol > 0 and add_vol * float(r["close"]) < capital * 0.9:
                    add_amt = add_vol * float(r["close"])
                    comm = max(add_amt * COMMISSION_RATE, 5)
                    capital -= add_amt + comm
                    pos.cost_price = (pos.cost_basis + add_amt) / (pos.volume + add_vol)
                    pos.volume += add_vol
                    pos.added = True
                    trades.append({
                        "date": date_str, "code": code, "action": "BUY",
                        "price": round(float(r["close"]), 3), "volume": add_vol,
                        "reason": "加仓(站稳中轨+MA20↑+MA5>MA20)",
                        "trade_type": "add",
                    })

        # 6. 入场（从signal_map读当天候选）
        if len(positions) < MAX_POSITIONS:
            candidates = signal_map.get(date_str, [])
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                for code, strength, reason in candidates:
                    if len(positions) >= MAX_POSITIONS:
                        break
                    if code in positions:
                        continue

                    df = all_data.get(code)
                    if df is None:
                        continue
                    row = df[df["date"] == date_str]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    close = float(r["close"])

                    invest = capital * POSITION_SIZE
                    vol = int(invest / close / 100) * 100
                    if vol <= 0:
                        continue
                    amt = vol * close
                    comm = max(amt * COMMISSION_RATE, 5)
                    if amt + comm > capital:
                        continue
                    capital -= amt + comm

                    bt = float(r["box_top"]) if not pd.isna(r["box_top"]) else close
                    bb = float(r["box_bottom"]) if not pd.isna(r["box_bottom"]) else close

                    pos = Position(code, close, vol, date_str, bt, bb, reason)
                    positions[code] = pos
                    trades.append({
                        "date": date_str, "code": code, "action": "BUY",
                        "price": round(close, 3), "volume": vol,
                        "reason": reason, "trade_type": "initial",
                        "box_top": round(bt, 3), "box_bottom": round(bb, 3),
                    })
                    total_candidates += 1

        # 7. 记录净值
        pv = sum(p.market_value for p in positions.values())
        equity = capital + pv
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        max_dd = min(max_dd, dd)

        if day_idx % 50 == 0:
            logger.info(f"  📊 {date_str} ({day_idx+1}/{len(sorted_dates)}) — "
                        f"NAV={equity:.0f}, holdings={len(positions)}, "
                        f"trades={len(trades)}, dd={dd:.1f}%")

        equity_curve.append({
            "date": date_str, "nav": round(equity, 2),
            "capital": round(capital, 2), "pos_value": round(pv, 2),
            "holdings": len(positions), "drawdown_pct": round(dd, 2),
        })

    # 强制平仓
    logger.info("🔄 强制平仓...")
    for code in list(positions.keys()):
        pos = positions[code]
        df = all_data.get(code)
        close = float(df.iloc[-1]["close"]) if df is not None and len(df) else pos.cost_price
        sell_amt = pos.volume * close
        comm = max(sell_amt * COMMISSION_RATE, 5)
        pnl = sell_amt - pos.cost_basis
        capital += sell_amt - comm
        trades.append({
            "date": END_DATE, "code": code, "action": "SELL",
            "price": round(close, 3), "volume": pos.volume,
            "reason": "回测结束强制平仓",
            "pnl_pct": round((close / pos.cost_price - 1) * 100, 2),
            "pnl": round(pnl, 2), "hold_days": pos.days_held,
            "trade_type": "close",
            "entry_date": pos.entry_date, "entry_price": round(pos.cost_price, 3),
        })
    positions.clear()

    elapsed = time.time() - t0
    return trades, equity_curve, total_candidates, max_dd, elapsed


# ════════════════════════════════════════════
# Phase 3: 生成报告
# ════════════════════════════════════════════
def generate_report(trades, equity_curve, total_candidates, total_stocks, max_dd, elapsed):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    buys = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]
    closes = [t for t in sells if t.get("trade_type") == "close" and t.get("pnl_pct") is not None]
    reduces = [t for t in sells if t.get("trade_type") == "reduce" and t.get("pnl_pct") is not None]
    all_exits = closes + reduces

    pnls = [t["pnl_pct"] for t in all_exits]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0

    win_amts = [t["pnl"] for t in all_exits if t.get("pnl_pct", 0) > 0]
    loss_amts = [t["pnl"] for t in all_exits if t.get("pnl_pct", 0) <= 0]

    df_eq = pd.DataFrame(equity_curve)
    start_nav = INITIAL_CAPITAL
    end_nav = df_eq["nav"].iloc[-1]
    total_return = (end_nav - start_nav) / start_nav * 100
    years = len(df_eq) / 252
    annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    daily_ret = df_eq["nav"].pct_change().dropna()
    sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
    sortino = np.sqrt(252) * daily_ret.mean() / daily_ret[daily_ret < 0].std() if len(daily_ret[daily_ret < 0]) > 0 else 0

    avg_hold = np.mean([t["hold_days"] for t in all_exits]) if all_exits else 0
    median_hold = np.median([t["hold_days"] for t in all_exits]) if all_exits else 0

    # 离场分类
    stop_losses = [t for t in all_exits if "止损" in t.get("reason", "")]
    take_profits = [t for t in all_exits if "止盈" in t.get("reason", "")]
    final_exits = [t for t in all_exits if "MA20" in t.get("reason", "") or "压力" in t.get("reason", "")]
    force_closes = [t for t in all_exits if "强制" in t.get("reason", "")]
    other = [t for t in all_exits if t not in stop_losses + take_profits + final_exits + force_closes]

    # 字符串
    avg_win_s = f"{np.mean(wins):.2f}% ({np.mean(win_amts):,.0f}元)" if wins else "N/A"
    avg_loss_s = f"{np.mean(losses):.2f}% ({np.mean(loss_amts):,.0f}元)" if losses else "N/A"
    pf = abs(sum(win_amts) / sum(loss_amts)) if sum(loss_amts) != 0 else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
    best = max(all_exits, key=lambda x: x["pnl_pct"]) if all_exits else None
    worst = min(all_exits, key=lambda x: x["pnl_pct"]) if all_exits else None
    best_s = f"{best['code']} | {best['date']} | {best['pnl_pct']}% | {best['hold_days']}天" if best else "N/A"
    worst_s = f"{worst['code']} | {worst['date']} | {worst['pnl_pct']}% | {worst['hold_days']}天" if worst else "N/A"

    # 信号分布
    entry_reasons = defaultdict(int)
    for t in buys:
        r = t.get("reason", "unknown")
        entry_reasons[r] += 1

    report = f"""# 超跌反转策略 — 全股票池回测报告

> **生成时间**: {now}
> **策略**: 超跌反转（多周期价格位置 + 横盘箱体突破 + 分批止盈）
> **股票池**: {total_stocks} 只 A 股
> **回测区间**: {equity_curve[0]['date'] if equity_curve else 'N/A'} ~ {equity_curve[-1]['date'] if equity_curve else 'N/A'}
> **交易日**: {len(equity_curve)} 天
> **耗时**: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)

---

## 一、策略参数

| 参数 | 值 |
|------|-----|
| 初始资金 | {INITIAL_CAPITAL:,} |
| 最大持仓数 | {MAX_POSITIONS} 只 |
| 单只仓位 | {POSITION_SIZE*100:.0f}% |
| 选股:价格位置 | 当前价 ≤ 1/2/3年高点 × 50%/60%/70% |
| 选股:流动性 | 日均成交额 ≥ {MIN_AVG_AMOUNT/1e8:.1f}亿 |
| 横盘判定 | {CONSOL_WINDOW}日振幅 ≤ {MAX_AMPLITUDE*100:.0f}% |
| 入场信号 | 站上中轨/突破上轨/回升箱体 + 放量{VOLUME_BREAK_RATIO:.0f}倍 + 金叉MA5>MA20 |
| 加仓条件 | 站上中轨 + MA20↑ + MA5>MA20(确认{CONFIRM_DAYS}天) |
| 硬止损 | -{HARD_STOP*100:.0f}% / 箱体破位 |
| 第一止盈 | +{TP1*100:.0f}%减{TP1_REDUCE*100:.0f}% |
| 第二止盈 | +{TP2*100:.0f}%减{TP2_REDUCE*100:.0f}% |
| 终极离场 | 跌破MA20 / 压力位(三年高点×60%) |
| 手续费 | 万{COMMISSION_RATE*10000:.0f} |

---

## 二、绩效概览

| 指标 | 数值 |
|------|------|
| 初始资金 | {INITIAL_CAPITAL:,} |
| 最终净值 | {end_nav:,.0f} |
| **总收益率** | **{total_return:.2f}%** |
| **年化收益率** | **{annual_return:.2f}%** |
| **夏普比率** | **{sharpe:.2f}** |
| **索提诺比率** | **{sortino:.2f}** |
| **最大回撤** | **{max_dd:.2f}%** |
| 总交易次数 | {len(trades)} |
| 买入笔数 | {len(buys)} |
| 卖出笔数 | {len(sells)} |

### 盈利分析

| 指标 | 数值 |
|------|------|
| 盈利交易 | {len(wins)} ({win_rate:.1f}%) |
| 亏损交易 | {len(losses)} ({(100-win_rate):.1f}%) |
| 平均盈利 | {avg_win_s} |
| 平均亏损 | {avg_loss_s} |
| 总盈利 | {sum(win_amts):,.0f}元 |
| 总亏损 | {sum(loss_amts):,.0f}元 |
| 盈亏比 | {pf_s} |
| 平均持仓天数 | {avg_hold:.1f} 天 |
| 中位持仓天数 | {median_hold:.0f} 天 |

### 最佳/最差交易

| | 股票 | 日期 | 盈亏 | 持仓天数 |
|---|------|------|------|---------|
| **最佳** | {best_s} |
| **最差** | {worst_s} |

---

## 三、平仓原因分布

| 原因 | 次数 | 占比 |
|------|:----:|:----:|
| 止损(箱体破位/硬止损/破MA20) | {len(stop_losses)} | {len(stop_losses)/max(len(all_exits),1)*100:.1f}% |
| 止盈(分批减仓) | {len(take_profits)} | {len(take_profits)/max(len(all_exits),1)*100:.1f}% |
| 终极离场(MA20/压力位) | {len(final_exits)} | {len(final_exits)/max(len(all_exits),1)*100:.1f}% |
| 回测结束强制平仓 | {len(force_closes)} | {len(force_closes)/max(len(all_exits),1)*100:.1f}% |
| 其他 | {len(other)} | {len(other)/max(len(all_exits),1)*100:.1f}% |
| **合计** | **{len(all_exits)}** | **100%** |

---

## 四、信号分布

| 信号类型 | 次数 | 占比 |
|----------|:----:|:----:|
"""
    for reason, count in sorted(entry_reasons.items(), key=lambda x: -x[1]):
        report += f"| {reason} | {count} | {count/len(buys)*100:.1f}% |\n"

    report += f"""
| 加仓信号 | {len([t for t in buys if t.get('trade_type')=='add'])} | — |
| 总候选信号 | {total_candidates} | — |

---

## 五、逐笔交易记录

### 5.1 全部买入记录

| # | 日期 | 股票 | 价格 | 数量 | 类型 | 信号原因 | 箱体上轨 | 箱体下轨 |
|---|------|------|------|------|------|----------|---------|---------|
"""
    for i, t in enumerate(buys, 1):
        report += f"| {i} | {t['date']} | {t['code']} | {t['price']} | {t['volume']} | {t.get('trade_type','')} | {t.get('reason','')} | {t.get('box_top','—')} | {t.get('box_bottom','—')} |\n"

    report += """
### 5.2 全部卖出记录

| # | 日期 | 股票 | 价格 | 数量 | 类型 | 原因 | 盈亏% | 盈亏额 | 持仓天数 | 入场日 | 入场价 |
|---|------|------|------|------|------|------|-------|--------|---------|--------|--------|
"""
    for i, t in enumerate(sells, 1):
        pnl_s = f"{t.get('pnl_pct','')}%" if t.get('pnl_pct') is not None else "—"
        pnl_a = f"{t.get('pnl',0):,.0f}" if t.get('pnl') is not None else "—"
        report += f"| {i} | {t['date']} | {t['code']} | {t['price']} | {t.get('volume','')} | {t.get('trade_type','')} | {t.get('reason','')} | {pnl_s} | {pnl_a} | {t.get('hold_days','')} | {t.get('entry_date','')} | {t.get('entry_price','')} |\n"

    report += """
---

## 六、净值曲线（每50天采样）

| 日期 | 净值 | 现金 | 持仓市值 | 持仓数 | 回撤% |
|------|------|------|---------|:------:|:-----:|
"""
    for ec in equity_curve[::50]:
        report += f"| {ec['date']} | {ec['nav']:.0f} | {ec['capital']:.0f} | {ec['pos_value']:.0f} | {ec['holdings']} | {ec['drawdown_pct']}% |\n"
    report += "| ... | ... | ... | ... | ... | ... |\n"
    for ec in equity_curve[-10:]:
        report += f"| {ec['date']} | {ec['nav']:.0f} | {ec['capital']:.0f} | {ec['pos_value']:.0f} | {ec['holdings']} | {ec['drawdown_pct']}% |\n"

    return report


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("超跌反转策略 — 全股票池回测 v2")
    logger.info("=" * 60)

    # Phase 1: 逐股预处理 + 提取信号
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 1: 逐股预处理 + 候选信号提取")
    logger.info("=" * 60)

    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    logger.info(f"📂 共发现 {len(csv_files)} 只股票CSV")

    all_data = {}  # {code: DataFrame} - 存全量指标
    signal_map = defaultdict(list)  # {date_str: [(code, strength, reason), ...]}
    total_valid = 0
    t1 = time.time()

    for i, fname in enumerate(csv_files):
        code = fname.replace(".csv", "")
        fpath = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(
                fpath,
                usecols=["date", "open", "high", "low", "close", "volume", "amount"],
                dtype={"date": str, "open": float, "high": float, "low": float,
                       "close": float, "volume": float, "amount": float},
            )
        except Exception:
            continue

        processed_df = preprocess_stock(code, df)[0]
        if processed_df is None or len(processed_df) < 252:
            continue

        # 提取信号: 遍历每行，筛选候选
        cand_count = 0
        for _, row in processed_df.iterrows():
            if not (row["in_price_zone"] and row["enough_liquidity"] and row["is_consolidating"]):
                continue

            d = row["date"].strftime("%Y-%m-%d")
            close = float(row["close"])
            # 计算信号强度
            strength = 0.0
            reasons = []

            # 信号1: 站上中轨 + 放量 + 金叉
            if row["above_mid"] and row["volume_ratio"] >= VOLUME_BREAK_RATIO and row["golden_cross"]:
                strength = row["volume_ratio"] * 2.0
                reasons.append("站上中轨+放量+金叉")

            # 信号2: 突破上轨 + 放量
            if row["break_above"] and row["volume_ratio"] >= VOLUME_BREAK_RATIO:
                strength = max(strength, row["volume_ratio"] * 1.8)
                reasons.append("突破上轨+放量")

            # 信号3: 从下轨回升≥25% + 金叉
            if not pd.isna(row["box_recovery"]) and row["box_recovery"] >= 0.25 and row["golden_cross"]:
                strength = max(strength, row["volume_ratio"] * 1.5)
                reasons.append(f"回升{row['box_recovery']*100:.0f}%箱体+金叉")

            if strength > 0:
                reason = "+".join(reasons[:2])
                signal_map[d].append((code, strength, reason))
                cand_count += 1

        all_data[code] = processed_df
        total_valid += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t1
            total_cand = sum(len(v) for v in signal_map.values())
            logger.info(f"  📥 {i+1}/{len(csv_files)}, {total_valid}只有效, "
                        f"{total_cand}候选信号, {elapsed:.0f}s")

    t_phase1 = time.time() - t1
    total_cand = sum(len(v) for v in signal_map.values())
    logger.info(f"✅ Phase1完成: {total_valid}只, {total_cand}候选信号, {t_phase1:.0f}s")

    # Phase 2: 回测
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: 全量回测")
    logger.info("=" * 60)
    trades, equity_curve, tc, max_dd, bt_elapsed = run_backtest_efficient(all_data, signal_map)

    # Phase 3: 报告
    total_elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3: 生成报告")
    logger.info("=" * 60)

    report = generate_report(trades, equity_curve, tc, total_valid, max_dd, total_elapsed)

    report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"Reversal_FullPool_{report_date}.md"
    report_path = os.path.join(REPORT_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"✅ 报告: {report_path}")

    if trades:
        csv_path = os.path.join(REPORT_DIR, f"Reversal_FullPool_{report_date}_trades.csv")
        pd.DataFrame(trades).to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 交易记录: {csv_path}")

    if equity_curve:
        eq_csv = os.path.join(REPORT_DIR, f"Reversal_FullPool_{report_date}_equity.csv")
        pd.DataFrame(equity_curve).to_csv(eq_csv, index=False, encoding="utf-8-sig")

    # 摘要
    end_nav = equity_curve[-1]["nav"] if equity_curve else 0
    print(f"""
{'='*55}
  🎯 超跌反转策略 全股票池回测完成
{'='*55}
  有效股票: {total_valid} 只
  候选信号: {total_cand} 个
  总收益率: {(end_nav-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%
  总交易: {len(trades)} 笔
  最大回撤: {max_dd:.2f}%
  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}分)
  报告: {report_path}
{'='*55}""")
