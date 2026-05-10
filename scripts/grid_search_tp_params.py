#!/usr/bin/env python3
"""
Step 6: 止盈参数调优 — take_profit
================================
搜索空间: tp1 [6%, 7%] × tp2 [13%, 15%] × 峰值回落 [8%, 10%] × 保护期 [0, 5]

用法:
    python3 scripts/grid_search_tp_params.py
"""

import sys, os, time, json, itertools, subprocess, re
from pathlib import Path
from datetime import datetime
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
OPT_DIR = WORKSPACE / "optimization"
os.makedirs(OPT_DIR, exist_ok=True)

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

# 搜索空间
TP1_VALS = [0.06, 0.07]
TP2_VALS = [0.13, 0.15]
TRAIL_VALS = [0.08, 0.10]
PROTECT_VALS = [0, 5]


def run_500_test(tp1: float, tp2: float, trail: float, protect: int) -> dict:
    """运行500只回测"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE)
    
    cmd = [
        sys.executable, str(WORKSPACE / "run_chanfund_v20_backtest.py"),
        "--mode", "grid",
        "--max-stocks", "500",
        "--tp1", str(tp1),
        "--tp2", str(tp2),
        "--tp-trail", str(trail),
        "--tp-protect", str(protect),
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(WORKSPACE), env=env
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log("  ⏱️ 超时")
        return {}
    except Exception as e:
        log(f"  ❌ 运行失败: {e}")
        return {}
    
    metrics = {}
    patterns = [
        ("total_return", r"收益[：:\s]\s*([\-\d.]+)%"),
        ("ann_return", r"年化[收益率]*[：:\s]\s*([\-\d.]+)%"),
        ("sharpe", r"夏普[：:\s]\s*([\-\d.]+)"),
        ("max_drawdown", r"回撤[：:\s]\s*([\-\d.]+)%"),
        ("win_rate", r"胜率[：:\s]\s*([\d.]+)%"),
        ("n_trades", r"交易[数]*[：:\s]\s*(\d+)"),
        ("avg_profit", r"平均盈利[：:\s]\s*([\d.]+)%"),
        ("avg_loss", r"平均亏损[：:\s]\s*([\-\d.]+)%"),
    ]
    for key, pat in patterns:
        m = re.search(pat, output)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except:
                pass
    return metrics


def main():
    log("=" * 60)
    log("Step 6: 止盈参数调优")
    log(f"  tp1={TP1_VALS}, tp2={TP2_VALS}, trail={TRAIL_VALS}, protect={PROTECT_VALS}")
    total = len(TP1_VALS) * len(TP2_VALS) * len(TRAIL_VALS) * len(PROTECT_VALS)
    log(f"  总计 {total} 组")
    log("=" * 60)
    
    results = []
    grid = list(itertools.product(TP1_VALS, TP2_VALS, TRAIL_VALS, PROTECT_VALS))
    
    for i, (tp1, tp2, trail, protect) in enumerate(grid):
        log(f"\n[{i+1}/{total}] tp1={tp1:.0%} tp2={tp2:.0%} trail={trail:.0%} protect={protect}d")
        t0 = time.time()
        
        metrics = run_500_test(tp1, tp2, trail, protect)
        elapsed = time.time() - t0
        
        row = {
            "tp1": tp1, "tp2": tp2, "trail_retrace": trail, "protect_days": protect,
            "total_return": metrics.get("total_return", -999),
            "sharpe": metrics.get("sharpe", -999),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "n_trades": metrics.get("n_trades", 0),
            "avg_profit": metrics.get("avg_profit", 0),
            "avg_loss": metrics.get("avg_loss", 0),
            "duration_s": round(elapsed),
        }
        results.append(row)
        
        log(f"  → 回报={row['total_return']:.2f}% 夏普={row['sharpe']:.2f} "
            f"回撤={row['max_drawdown']:.2f}% 胜率={row['win_rate']:.1f}% "
            f"交易={row['n_trades']} [{elapsed:.0f}s]")
    
    df = pd.DataFrame(results)
    
    # 按夏普排序
    df_by_sharpe = df.sort_values("sharpe", ascending=False)
    df_by_sharpe.to_csv(OPT_DIR / "tp_param_results.csv", index=False)
    
    # 也按盈亏比排序
    df["profit_loss_ratio"] = df["avg_profit"] / (df["avg_loss"].abs() + 0.001)
    df_by_pl = df.sort_values("profit_loss_ratio", ascending=False)
    df_by_pl.to_csv(OPT_DIR / "tp_param_results_by_pl.csv", index=False)
    
    log(f"\n{'='*60}")
    log("📊 按夏普排序 (Top 5)")
    log(f"{'='*60}")
    for _, row in df_by_sharpe.head(5).iterrows():
        log(f"  tp1={row['tp1']:.0%} tp2={row['tp2']:.0%} trail={row['trail_retrace']:.0%} "
            f"prot={row['protect_days']:.0f}d → 回报={row['total_return']:.2f}% "
            f"夏普={row['sharpe']:.2f} 回撤={row['max_drawdown']:.2f}%")
    
    log(f"\n📊 按盈亏比排序 (Top 5)")
    log(f"{'='*60}")
    for _, row in df_by_pl.head(5).iterrows():
        log(f"  tp1={row['tp1']:.0%} tp2={row['tp2']:.0%} trail={row['trail_retrace']:.0%} "
            f"prot={row['protect_days']:.0f}d → 回报={row['total_return']:.2f}% "
            f"盈亏比={row['profit_loss_ratio']:.2f}")
    
    # 最优(按夏普)
    best = df_by_sharpe.iloc[0]
    log(f"\n🏆 最优(夏普): tp1={best['tp1']:.0%} tp2={best['tp2']:.0%} "
        f"trail={best['trail_retrace']:.0%} protect={best['protect_days']:.0f}d")
    
    with open(OPT_DIR / "best_tp_params.json", "w") as f:
        json.dump({
            "tp1_pct": best["tp1"],
            "tp2_pct": best["tp2"],
            "tp_trail_retrace": best["trail_retrace"],
            "tp_protect_days": best["protect_days"],
        }, f)


if __name__ == "__main__":
    main()
