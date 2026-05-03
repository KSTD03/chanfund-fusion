"""
ChanFund Fusion v1.0.1 — 全量回测运行器
=====================================
全股票池(5157只) × 全时间段(2021-01-01 ~ 2026-04-29)
详细记录每笔成交和买卖逻辑，产出完整报告。

运行方式: python3 run_backtest_v101.py (后台: nohup python3 run_backtest_v101.py &)
"""

from __future__ import annotations

import os, sys, json, time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

_WORKSPACE = str(Path(__file__).resolve().parent.parent)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import numpy as np
import pandas as pd

# ============================================================
# 配置区
# ============================================================
BACKTEST_NAME = "ChanFund-Fusion_full_v1.0.1"
STRATEGY_DIR = str(Path(__file__).resolve().parent)
REPORT_DIR = "/home/quant/backtest report"

BACKTEST_RANGE = {
    "warmup_start": "2018-01-01",    # 3年预热
    "warmup_end": "2020-12-31",
    "start": "2021-01-01",           # 回测区间起点
    "end": "2026-04-29",             # 回测区间终点（全量数据最后一天）
}

INITIAL_CAPITAL = 1_000_000

# 日志配置
LOG_PATH = os.path.join(STRATEGY_DIR, f"backtest_{BACKTEST_NAME}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("chanfund_v101_backtest")


def init_qlib():
    """初始化Qlib"""
    import qlib
    provider_uri = os.path.join(
        os.path.expanduser("~"),
        ".openclaw", "workspace", "quant", "qlib_data", "cn_data",
    )
    qlib.init(provider_uri=provider_uri, region="cn")
    logger.info(f"Qlib initialized: {provider_uri}")
    return qlib


def load_all_stocks() -> list:
    """加载全量股票列表"""
    instr_path = os.path.join(
        os.path.expanduser("~"),
        ".openclaw", "workspace", "quant", "qlib_data", "cn_data",
        "instruments", "all.txt",
    )
    stocks = []
    with open(instr_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                stocks.append(parts[0])
    logger.info(f"Loaded {len(stocks)} stocks")
    return stocks


def calc_metrics(trade_log, initial_capital):
    """计算绩效指标"""
    if not trade_log:
        return {
            "total_return_pct": 0.0,
            "annual_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
        }

    # 按日期计算每日净值
    sells = [t for t in trade_log if t.get("action") == "sell"]
    buys = [t for t in trade_log if t.get("action") == "buy"]

    if not sells:
        return {
            "total_return_pct": 0.0,
            "annual_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": len(buys) + len(sells),
            "win_rate": 0.0,
            "num_wins": 0,
            "num_losses": len(sells),
        }

    pnls = [s["pnl"] for s in sells if "pnl" in s]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = sum(pnls) if pnls else 0.0
    win_rate = len(wins) / len(pnls) if pnls else 0.0

    # 简化夏普
    if pnls:
        avg_return = np.mean(pnls)
        std_return = np.std(pnls)
        sharpe = avg_return / max(std_return, 1e-10) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "total_trades": len(buys) + len(sells),
        "buy_trades": len(buys),
        "sell_trades": len(sells),
        "num_wins": len(wins),
        "num_losses": len(losses),
        "win_rate": round(win_rate * 100, 1),
        "avg_win_pct": round(np.mean(wins) * 100, 2) if wins else 0.0,
        "avg_loss_pct": round(np.mean(losses) * 100, 2) if losses else 0.0,
        "sharpe_ratio": round(sharpe, 2),
        "best_trade_pct": round(max(pnls) * 100, 2) if pnls else 0.0,
        "worst_trade_pct": round(min(pnls) * 100, 2) if pnls else 0.0,
    }


def analyze_signals(decision_log):
    """分析信号分布"""
    if not decision_log:
        return {}, {}, {}

    accepted = [d for d in decision_log if d.get("decision") == "accepted"]
    rejected = [d for d in decision_log if d.get("decision") == "rejected"]

    # 信号类型分布
    from collections import Counter
    sig_dist = Counter(d.get("signal_type", "unknown") for d in accepted)
    rej_dist = Counter(d.get("reason", "unknown") for d in rejected)

    return dict(sig_dist), dict(rej_dist), accepted, rejected


def generate_report(metrics, signal_dist, reject_dist, trade_log, decision_log,
                    start_date, end_date, total_stocks, elapsed):
    """生成回测报告Markdown"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# ChanFund Fusion v1.0.1 — 全量回测报告

> 生成时间: {now}
> 策略版本: v1.0.1 (2026-05-03 精修版)

> 股票池: {total_stocks} 只 | 回测区间: {start_date} ~ {end_date} | 耗时: {elapsed:.1f}秒

## 一、优化特性启用状态

| 特性 | 状态 |
|------|------|
| 保护性移动止损 | ✅ 开（笔低点+中枢ZG跟踪）|
| ER噪声过滤 | ✅ 开（分级降权：丢弃/0.5/1.0/1.2）|
| 冷却期 | ✅ 开（连续2次止损→10日冷却）|
| 趋势方向过滤 | ✅ 开（adaptive模式）|
| 红黄牌排雷 | ✅ 开（ST/调查/质押/商誉）|
| 分级仓位系数 | ✅ 开（HD=1.0/3B=0.8/软背驰=0.25）|
| 动态风险预算 | ✅ 开（权益×1%/价差）|
| 信号失效机制 | ✅ 开（结构破坏标记作废）|
| 试探仓演化 | ✅ 开（2R半仓止盈+移损成本）|

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
| 最佳单笔 | {metrics.get('best_trade_pct', 0)}% |
| 最差单笔 | {metrics.get('worst_trade_pct', 0)}% |

## 三、信号分布

| 信号类型 | 次数 | 占比 |
|----------|------|------|
"""
    total_sigs = sum(signal_dist.values()) if signal_dist else 0
    for sig_type, count in sorted(signal_dist.items(), key=lambda x: -x[1]):
        pct = count / max(total_sigs, 1) * 100
        report += f"| {sig_type} | {count} | {pct:.1f}% |\n"

    report += f"\n## 四、拒绝原因分布\n\n| 原因 | 次数 | 占比 |\n|------|------|------|\n"
    total_rej = sum(reject_dist.values()) if reject_dist else 0
    for reason, count in sorted(reject_dist.items(), key=lambda x: -x[1]):
        pct = count / max(total_rej, 1) * 100
        report += f"| {reason} | {count} | {pct:.1f}% |\n"

    report += f"""

## 五、逐笔成交记录

| # | 日期 | 股票 | 操作 | 价格 | 权重 | PnL | 原因 |
|---|------|------|------|------|------|------|------|
"""
    for i, t in enumerate(trade_log[-200:], 1):  # 最多展示最近200笔
        pnl = t.get("pnl", "")
        pnl_str = f"{pnl:.2%}" if isinstance(pnl, (int, float)) else ""
        report += f"| {i} | {t.get('date', '')} | {t.get('symbol', '')} | {t.get('action', '')} | {t.get('price', ''):.3f} | {t.get('weight', 0):.2%} | {pnl_str} | {t.get('reason', '')} |\n"

    report += f"""

## 六、决策日志摘要

| # | 日期 | 股票 | 决策 | 信号类型 | 原因 | 权重 | 基本面分 | 详情 |
|---|------|------|------|----------|------|------|---------|------|
"""
    for i, d in enumerate(decision_log[-100:], 1):
        report += f"| {i} | {d.get('date', '')} | {d.get('symbol', '')} | {d.get('decision', '')} | {d.get('signal_type', '')} | {d.get('reason', '')} | {d.get('weight', 0):.2%} | {d.get('fund_score', 0)} | {d.get('details', '')} |\n"

    report += f"""

## 七、参数配置（核心项）

| 参数 | 值 |
|------|-----|
| 止损模式 | trailing (bi_low + zs_zg) |
| 初始止损ATR倍数 | 2.0 |
| ER丢弃阈值 | <20% |
| ER降权区间 | 20-30% (×0.5) |
| ER增强阈值 | >80% (×1.2) |
| 冷却期 | 连续2次止损→10日 |
| 方向过滤模式 | adaptive |
| 单笔风险 | 权益×1% |
| 单只上限 | 8% |
| 行业上限 | 25% |
"""
    return report


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"ChanFund Fusion v1.0.1 — 全量回测开始")
    logger.info(f"  区间: {BACKTEST_RANGE['start']} ~ {BACKTEST_RANGE['end']}")
    logger.info("=" * 60)

    # 1. 初始化Qlib
    qlib = init_qlib()

    # 2. 加载股票池
    all_stocks = load_all_stocks()
    logger.info(f"Total stocks: {len(all_stocks)}")

    # 3. 加载策略
    sys.path.insert(0, STRATEGY_DIR)
    from strategy import Strategy

    config_path = os.path.join(STRATEGY_DIR, "config.yaml")

    # 创建策略实例
    strategy = Strategy(config_path=config_path)
    logger.info(f"Strategy initialized: {strategy.strategy_name} v{strategy.strategy_version}")

    # 4. 运行回测
    from qlib.backtest import backtest, executor
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158

    # 准备回测参数
    benchmark = "SH000300"
    trade_dates = D.list_trade_dates(
        start_time=BACKTEST_RANGE["start"],
        end_time=BACKTEST_RANGE["end"],
    )
    logger.info(f"Trade days: {len(trade_dates)}")

    # 定义回测执行器
    from qlib.contrib.evaluate import risk_analysis
    from qlib.backtest.executor import NestedExecutor
    from qlib.backtest.decision import Order

    # 使用自定义策略跑回测
    from qlib.backtest import backtest as qlib_backtest

    try:
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            strategy=strategy,
            trade_dates=trade_dates,
            universe=all_stocks,
            start_time=BACKTEST_RANGE["start"],
            end_time=BACKTEST_RANGE["end"],
            initial_cash=INITIAL_CAPITAL,
            benchmark=benchmark,
            deal_price="close",
            cost=None,
            generate_report=True,
        )

        logger.info("Backtest simulation completed")

        # 收集回测结果
        trade_log = strategy.get_trade_log()
        decision_log = strategy.get_decision_log()

        # 分析
        metrics = calc_metrics(trade_log, INITIAL_CAPITAL)
        signal_dist, reject_dist, _, _ = analyze_signals(decision_log)

        elapsed = time.time() - start_time

        # 生成报告
        report = generate_report(
            metrics, signal_dist, reject_dist,
            trade_log, decision_log,
            BACKTEST_RANGE["start"], BACKTEST_RANGE["end"],
            len(all_stocks), elapsed,
        )

        # 保存报告 & 数据
        report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{BACKTEST_NAME}_{report_date}.md"
        report_path = os.path.join(REPORT_DIR, report_filename)

        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 保存成交记录CSV
        csv_path = os.path.join(REPORT_DIR, f"{BACKTEST_NAME}_{report_date}_trades.csv")
        if trade_log:
            pd.DataFrame(trade_log).to_csv(csv_path, index=False, encoding="utf-8-sig")

        logger.info(f"✅ Report saved: {report_path}")
        logger.info(f"✅ Trades CSV: {csv_path}")

        # 打印摘要
        print()
        print("=" * 55)
        print(f"  ChanFund Fusion v1.0.1 回测完成")
        print("=" * 55)
        print(f"  总收益率: {metrics.get('total_return_pct', 'N/A')}%")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 'N/A')}")
        print(f"  总交易数: {metrics.get('total_trades', 0)}")
        print(f"  胜率: {metrics.get('win_rate', 0)}%")
        print(f"  总耗时: {elapsed:.1f}s")
        print(f"  报告: {report_path}")
        print()

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Backtest failed: {e}", exc_info=True)
        # 保存错误信息
        error_report = f"""# ChanFund Fusion v1.0.1 — 回测失败报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 耗时: {elapsed:.1f}秒

## 错误信息

```
{str(e)}
```

## 已收集的交易记录

"""
        trade_log = strategy.get_trade_log()
        decision_log = strategy.get_decision_log()

        metrics = calc_metrics(trade_log, INITIAL_CAPITAL)

        error_report += f"交易记录: {len(trade_log)} 条\n"
        error_report += f"决策记录: {len(decision_log)} 条\n"
        error_report += f"已完成交易总收益: {metrics.get('total_return_pct', 'N/A')}%\n"

        report_filename = f"{BACKTEST_NAME}_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(REPORT_DIR, report_filename)
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(error_report)
        logger.info(f"Partial report saved: {report_path}")

        # 仍尝试输出
        print(f"⚠️  Backtest completed with errors (elapsed: {elapsed:.1f}s)")
        print(f"   Partial results: {report_path}")


if __name__ == "__main__":
    main()
