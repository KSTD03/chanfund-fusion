#!/usr/bin/env python3
"""
Step 5: 因子门槛网格搜索 — fusion.buy_score_min
==============================================
搜索空间: buy_score_min [30, 40, 50]
          默认评分 [55, 60]

用法:
    python3 scripts/grid_search_factor_threshold.py
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
BUY_SCORE_MIN = [30, 40, 50]
DEFAULT_SCORES = [55, 60]


def run_500_test(buy_score_min: float, default_score: float) -> dict:
    """运行500只回测"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE)
    
    cmd = [
        sys.executable, str(WORKSPACE / "run_chanfund_v20_backtest.py"),
        "--mode", "grid",
        "--max-stocks", "500",
        "--buy-score-min", str(buy_score_min),
        "--default-score", str(default_score),
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
        ("total_return", r"收益[：:\s]+([\-\d.]+)%"),
        ("ann_return", r"年化[收益率]*[：:\s]+([\-\d.]+)%"),
        ("sharpe", r"夏普[：:\s]+([\-\d.]+)"),
        ("max_drawdown", r"回撤[：:\s]+([\-\d.]+)%"),
        ("win_rate", r"胜率[：:\s]+([\d.]+)%"),
        ("n_trades", r"交易[数]*[：:\s]+(\d+)"),
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
    log("Step 5: 因子门槛网格搜索")
    log(f"  buy_score_min={BUY_SCORE_MIN}, default_score={DEFAULT_SCORES}")
    log(f"  总计 {len(BUY_SCORE_MIN)*len(DEFAULT_SCORES)} 组")
    log("=" * 60)
    
    results = []
    grid = list(itertools.product(BUY_SCORE_MIN, DEFAULT_SCORES))
    total = len(grid)
    
    for i, (bsm, ds) in enumerate(grid):
        log(f"\n[{i+1}/{total}] buy_score_min={bsm} default_score={ds}")
        t0 = time.time()
        
        metrics = run_500_test(bsm, ds)
        elapsed = time.time() - t0
        
        row = {
            "buy_score_min": bsm, "default_score": ds,
            "total_return": metrics.get("total_return", -999),
            "sharpe": metrics.get("sharpe", -999),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "n_trades": metrics.get("n_trades", 0),
            "duration_s": round(elapsed),
        }
        results.append(row)
        
        log(f"  回报={row['total_return']:.2f}% 夏普={row['sharpe']:.2f} "
            f"回撤={row['max_drawdown']:.2f}% 胜率={row['win_rate']:.1f}% "
            f"交易={row['n_trades']} [{elapsed:.0f}s]")
    
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)
    df.to_csv(OPT_DIR / "factor_threshold_results.csv", index=False)
    
    log(f"\n{'='*60}")
    log("📊 排序 (按夏普)")
    log(f"{'='*60}")
    for _, row in df.iterrows():
        log(f"  buy={row['buy_score_min']:.0f} default={row['default_score']:.0f} "
            f"→ 回报={row['total_return']:.2f}% 夏普={row['sharpe']:.2f} "
            f"回撤={row['max_drawdown']:.2f}%")
    
    best = df.iloc[0]
    log(f"\n🏆 最优: buy_score_min={best['buy_score_min']:.0f} default_score={best['default_score']:.0f}")
    
    with open(OPT_DIR / "best_factor_threshold.json", "w") as f:
        json.dump({"buy_score_min": best["buy_score_min"], "default_score": best["default_score"]}, f)


if __name__ == "__main__":
    main()
