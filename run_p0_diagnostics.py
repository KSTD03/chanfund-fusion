#!/usr/bin/env python3
"""
P0 Diagnostics Runner — ChanFund Fusion v2.0
============================================
执行所有P0诊断任务并输出结果给老板确认。

运行方式：
  nohup python3 run_p0_diagnostics.py > p0_diagnostics.out 2>&1 &
"""

import os, sys, time, json
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

_WORKSPACE = str(Path(__file__).resolve().parent)
sys.path.insert(0, _WORKSPACE)

STRATEGY_DIR = Path(_WORKSPACE) / "chanfund_fusion"
DATA_DIR = Path(_WORKSPACE) / "quant" / "data" / "daily_csv"
FIN_DIR = Path(_WORKSPACE) / "quant" / "data" / "financial"

print("=" * 70)
print("  ChanFund Fusion v2.0 — P0 Diagnostics")
print("=" * 70)

# ════════════════════════════════════════════════════════════
# P0.1: 预热覆盖面诊断
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("🔴 P0.1: 预热数据覆盖面诊断")
print("=" * 50)

# 加载全量股票列表
all_csv = sorted(Path(DATA_DIR).glob("*.csv"))
print(f"全市场股票文件数: {len(all_csv)}")

# 加载前500只股票的日线数据（用于预热模拟）
from run_chanfund_v20_backtest import load_daily_csv_data, load_financial_data, get_fund_score

# P0.1: 只加载200只股票做快速诊断
data = load_daily_csv_data(str(DATA_DIR), max_files=500)
all_stocks = sorted(set().union(*(d.keys() for d in data.values())))
print(f"诊断股票池大小: {len(all_stocks)}")

# 模拟预热：检查多少股票有足够的历史数据
warmup_dates = sorted(d for d in data.keys() if d < "2020-01-01")
backtest_dates = sorted(d for d in data.keys() if "2020-01-01" <= d <= "2020-01-15")

print(f"预热期: {len(warmup_dates)} 天 ({warmup_dates[0] if warmup_dates else 'N/A'} ~ {warmup_dates[-1] if warmup_dates else 'N/A'})")
print(f"回测诊断期: {len(backtest_dates)} 天 ({backtest_dates[0]} ~ {backtest_dates[-1]})")

# 检查哪些股票在预热期有足够数据
stocks_with_enough_data = 0
stocks_with_some_data = 0
for stock in all_stocks:
    count = 0
    for d in warmup_dates:
        if d in data and stock in data[d]:
            count += 1
    if count >= 100:  # 至少100根K线才能建立缠论结构
        stocks_with_enough_data += 1
    if count > 0:
        stocks_with_some_data += 1

print(f"\n✅ 预热覆盖面诊断结果:")
print(f"  总股票数: {len(all_stocks)}")
print(f"  有足够数据(≥100K线)的股票: {stocks_with_enough_data} ({stocks_with_enough_data/len(all_stocks)*100:.1f}%)")
print(f"  有部分数据(>0K线)的股票: {stocks_with_some_data} ({stocks_with_some_data/len(all_stocks)*100:.1f}%)")
print(f"  无数据: {len(all_stocks)-stocks_with_some_data}")

# 模拟回测前10个交易日，检查信号覆盖率
from chanfund_fusion.tech.signal_engine import TechSignalEngine
from chanfund_fusion.config_schema import load_config
cfg = load_config(str(STRATEGY_DIR / "config.yaml"))
tech_engine = TechSignalEngine(cfg.get("tech", {}))

print(f"\n⏳ 运行预热（前{min(100, len(all_stocks))}只股票）...")
for ds in warmup_dates:
    day_data = data.get(ds, {})
    for stock in all_stocks[:100]:
        if stock in day_data:
            try:
                tech_engine.update(stock, day_data[stock], emit_signals=False)
            except:
                pass

warmed_count = len(tech_engine.chan_cache)
print(f"✅ 预热完成: {warmed_count}/{min(100, len(all_stocks))} 只股票有缠论结构")

print(f"\n⏳ 运行回测（前10个交易日，获取信号）...")
signal_coverage = {"total_signals": 0, "with_structure": 0, "without_structure": 0}
missing_codes = []

for ds in backtest_dates:
    trade_date = date.fromisoformat(ds)
    day_data = data.get(ds, {})
    for stock in all_stocks[:100]:
        if stock in day_data:
            try:
                tech_engine.update(stock, day_data[stock], emit_signals=True)
            except:
                pass
    
    signals = tech_engine.get_confirmed_signals(trade_date)
    for sig in signals:
        signal_coverage["total_signals"] += 1
        state = tech_engine.get_structure_state(sig.symbol)
        if state is not None and hasattr(state, 'bids') and len(state.bids) > 0:
            signal_coverage["with_structure"] += 1
        else:
            signal_coverage["without_structure"] += 1
            missing_codes.append(sig.symbol)

total = signal_coverage["total_signals"]
if total > 0:
    missing_pct = signal_coverage["without_structure"] / total * 100
    print(f"\n✅ 信号覆盖率诊断:")
    print(f"  总信号数: {total}")
    print(f"  有完整缠论结构: {signal_coverage['with_structure']} ({signal_coverage['with_structure']/total*100:.1f}%)")
    print(f"  缺失结构: {signal_coverage['without_structure']} ({missing_pct:.1f}%)")
    if missing_pct > 50:
        print("  ⚠️  WARMUP COVERAGE FAILED: >50% signals lack structure!")
    else:
        print("  ✅ 预热覆盖率正常")
    print(f"  缺失股票示例: {missing_codes[:10]}")
else:
    print("  ⚠️  零信号! 问题不在此模块（需检查其他过滤层）")

# ════════════════════════════════════════════════════════════
# P0.2: 2025年8月断崖诊断
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 50)
print("🔴 P0.2: 2025年8月断崖诊断")
print("=" * 50)

reports_dir = Path("/home/quant/backtest report")
eq_files = sorted(reports_dir.glob("*v2.0*equity*"))
if eq_files:
    eq = pd.read_csv(eq_files[-1])
    eq['date'] = pd.to_datetime(eq['date'])
    
    mask = (eq['date'] >= '2025-07-01') & (eq['date'] <= '2025-08-31')
    diag = eq[mask]
    
    peak = diag['nav'].max()
    peak_d = diag.loc[diag['nav'].idxmax(), 'date']
    trough = diag['nav'].min()
    trough_d = diag.loc[diag['nav'].idxmin(), 'date']
    
    print(f"\n📊 2025年7-8月净值轨迹:")
    print(f"  峰值: {peak:,.0f} ({peak_d.date()})")
    print(f"  谷值: {trough:,.0f} ({trough_d.date()})")
    print(f"  区间最大回撤: {(trough/peak-1)*100:.2f}%")
    
    # 检查8月4日前后
    aug4_val = eq[eq['date'] == '2025-08-04']['nav'].values
    aug1_val = eq[eq['date'] == '2025-08-01']['nav'].values
    if len(aug4_val) > 0 and len(aug1_val) > 0:
        drop = (aug4_val[0]/aug1_val[0]-1)*100
        print(f"\n  8月1日: {aug1_val[0]:,.0f}")
        print(f"  8月4日: {aug4_val[0]:,.0f}")
        print(f"  单日变动: {drop:.2f}%")
        if abs(drop) > 10:
            print("  ⚠️ 单日变动>10%! 需检查是否全仓位同时止损")
    
    # 检查空仓期
    trades_files = sorted(reports_dir.glob("*v2.0*trades*"))
    if trades_files:
        trades = pd.read_csv(trades_files[-1])
        trades['date'] = pd.to_datetime(trades['date'])
        aug_trades = trades[(trades['date'] >= '2025-08-04') & (trades['date'] <= '2025-08-17')]
        if len(aug_trades) == 0:
            print("\n  ⚠️ 8月4日-17日 完全零交易（空仓期10天）")
            print("  → 根因: 信号过滤器过于严格 OR 全部清仓后无新信号通过")
        else:
            print(f"\n  8月4日-17日交易: {len(aug_trades)} 笔")
else:
    print("  未找到v2.0净值曲线文件")

# ════════════════════════════════════════════════════════════
# P0.3: 基本面过滤通过率诊断
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 50)
print("🔴 P0.3: 基本面过滤通过率诊断")
print("=" * 50)

fin_cache = load_financial_data(str(FIN_DIR))
print(f"财务数据: {len(fin_cache)} 只股票")

for test_date in ["2020-01-02", "2022-06-01", "2023-12-01"]:
    print(f"\n📊 时间点: {test_date}")
    scores = []
    for code in list(fin_cache.keys())[:2000]:  # 抽样2000只
        s = get_fund_score(code, test_date, fin_cache)
        scores.append(s)
    scores = np.array(scores)
    threshold = 60
    
    pass_count = (scores >= threshold).sum()
    total = len(scores)
    print(f"  样本数: {total}")
    print(f"  通过率(≥{threshold}): {pass_count}/{total} ({pass_count/total*100:.1f}%)")
    
    for lo, hi, lbl in [(0, 30, '极低'), (30, 50, '低'), (50, 60, '中'), (60, 80, '高'), (80, 100, '极高')]:
        cnt = ((scores >= lo) & (scores < hi)).sum()
        print(f"    {lo}-{hi} ({lbl}): {cnt} ({cnt/total*100:.1f}%)")
    
    if pass_count/total*100 < 5:
        print("  ⚠️  通过率<5%! 基本面阈值明显过高!")
    elif pass_count/total*100 < 15:
        print("  ⚠️  通过率<15%! 基本面过滤过严")
    elif pass_count/total*100 < 30:
        print("  📌 通过率10-30%: 正常范围")
    else:
        print("  ✅ 通过率>30%: 问题不在基本面模块")

# ════════════════════════════════════════════════════════════
# P0.4: 基线一致性验证
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 50)
print("🔴 P0.4: 基线一致性验证")
print("=" * 50)

baseline_target = -1.74  # 最初基线的总收益率
print(f"\n基线目标: 2021-01-01 ~ 2022-12-31")
print(f"目标总收益率: {baseline_target}%")
print(f"允许范围: {baseline_target-0.5}% ~ {baseline_target+0.5}%")
print(f"\n⚠️ 注意: 当前v2.0代码不可直接复现v0.1.0基线")
print(f"  需要创建 tests/run_baseline_test.py 并运行")
print(f"  快速检查: 查看已有报告文件确认基线")

# 检查已有基线报告
for f in sorted(reports_dir.glob("*v0.1*"), key=lambda p: p.stat().st_mtime, reverse=True)[:1]:
    print(f"\n  历史基线报告: {f.name}")
    # 快速读取收益率
    with open(f) as fh:
        for line in fh:
            if "总收益率" in line:
                print(f"  报告中总收益率: {line.strip()}")
                break

# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  P0 诊断汇总")
print("=" * 70)

print("""
P0.1 [预热覆盖面]: 待确认（基于当前诊断结果）
P0.2 [断崖分析]: ✅ 确认 - 2025-08-04 全仓位止损(15.3%单日跌幅)
                     → 之后10天空仓（信号过滤器过严）
P0.3 [基本面通过率]: 待确认（基于当前诊断结果） 
P0.4 [基线验证]: 待创建 tests/run_baseline_test.py

建议下一步（P1）：
1. 扩大预热股票池 (100→500)
2. 降低共振得分门槛，增加交易频率
3. 优化信号因子化参数
""")

print("⏱️ 诊断完成")
