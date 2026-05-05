#!/usr/bin/env python3
"""500只中等规模回测（第一阶段验证）"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psutil
from run_chanfund_v20_backtest import (
    load_daily_csv_data, load_financial_data,
    TIME_SPLITS, REPORT_DIR, run_backtest, generate_report,
    BACKTEST_NAME, ENV_MULTIPLIER, FUND_SCORE_MIN,
    INITIAL_CAPITAL, MAX_POSITIONS, FIXED_STOP_LOSS,
)

mem0 = psutil.Process().memory_info().rss / 1024**3
print(f"[MEM] Start: {mem0:.2f} GB")

# 加载500只股票数据
data = load_daily_csv_data("quant/data/daily_csv", max_files=500)
mem1 = psutil.Process().memory_info().rss / 1024**3
print(f"[MEM] Data loaded: {mem1:.2f} GB (delta {(mem1-mem0)*1024:.0f} MB)")

fin_cache = load_financial_data("quant/data/financial")
index_data = {}

# 回测
from run_chanfund_v20_backtest import _T_START
_T_START = time.time()

result = run_backtest(
    data, fin_cache, index_data,
    TIME_SPLITS["test"]["start"], TIME_SPLITS["test"]["end"],
    INITIAL_CAPITAL, "500_test", 0,
)
mem2 = psutil.Process().memory_info().rss / 1024**3
print(f"[MEM] After backtest: {mem2:.2f} GB")

# 输出结果
m = result["metrics"]
print(f"\n{'='*50}")
print("  500只中规模回测结果")
print(f"{'='*50}")
print(f"总收益率:  {m['total_return']:.2f}%")
print(f"年化收益:  {m['ann_return']:.2f}%")
print(f"夏普比率:  {m['sharpe']:.2f}")
print(f"最大回撤:  {m['max_drawdown']:.2f}%")
print(f"胜率:      {m['win_rate']:.1f}%")
print(f"交易数:    {m['n_trades']}")
print(f"信号数:    {result['total_signals']}")
print(f"内存峰值:  {max(mem0,mem1,mem2):.3f} GB")

# 与v1.4基线对比
print(f"\n{'='*50}")
print("  与v1.4基线对比")
print(f"{'='*50}")
baseline = {"total_return": 29.02, "sharpe": 3.44, "max_drawdown": -13.71,
            "win_rate": 58.6, "n_trades": 406}
for k in baseline:
    v = m.get(k, 0)
    b = baseline[k]
    delta = v - b
    print(f"  {k:<12}: v2.0={v:.2f} vs v1.4={b:.2f} ({delta:+.2f})")

# 2025年8月断崖检查
print(f"\n{'='*50}")
print("  2025年7月25日-8月20日净值")
print(f"{'='*50}")
for eq in result["equity_curve"]:
    if "2025-07-25" <= eq["date"] <= "2025-08-20":
        print(f"  {eq['date']}: NAV={eq['nav']:.0f} dd={eq.get('dd',0):.1f}%")

# 保存报告
report_path = REPORT_DIR / f"{BACKTEST_NAME}_500_test_{time.strftime('%Y%m%d')}.md"
with open(report_path, "w") as f:
    f.write(generate_report(result, "500_TEST", len(data), 0, 0, time.time()-_T_START))
print(f"\n📄 报告: {report_path}")

elapsed = time.time() - _T_START
print(f"\n⏱️ 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
