#!/usr/bin/env python3
"""
Step 4: 环境乘数网格搜索 — market_env.multipliers
=================================================
搜索空间: 牛[1.0, 1.2, 1.5] × 震荡[0.8, 1.0] × 熊[0.3, 0.6]
使用500只验证集加速

用法:
    python3 scripts/grid_search_env_multipliers.py
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
BULL_VALS = [1.0, 1.2, 1.5]
OSCILLATE_VALS = [0.8, 1.0]
BEAR_VALS = [0.3, 0.6]


def run_500_test(config_patch: dict) -> dict:
    """在500只子集上运行回测，返回指标"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE)
    
    cmd = [
        sys.executable, str(WORKSPACE / "run_chanfund_v20_backtest.py"),
        "--mode", "grid",
        "--max-stocks", "500",
        "--env-bull", str(config_patch["bull"]),
        "--env-oscillate", str(config_patch["oscillate"]),
        "--env-bear", str(config_patch["bear"]),
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
    
    # 解析结果
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
    log("Step 4: 环境乘数网格搜索")
    log(f"  牛={BULL_VALS}, 震荡={OSCILLATE_VALS}, 熊={BEAR_VALS}")
    log(f"  总计 {len(BULL_VALS)*len(OSCILLATE_VALS)*len(BEAR_VALS)} 组")
    log("=" * 60)
    
    results = []
    grid = list(itertools.product(BULL_VALS, OSCILLATE_VALS, BEAR_VALS))
    total = len(grid)
    
    for i, (bull, osc, bear) in enumerate(grid):
        log(f"\n[{i+1}/{total}] 牛={bull} 震荡={osc} 熊={bear}")
        t0 = time.time()
        
        patch = {"bull": bull, "oscillate": osc, "bear": bear}
        metrics = run_500_test(patch)
        
        elapsed = time.time() - t0
        sharpe = metrics.get("sharpe", -999)
        ret = metrics.get("total_return", -999)
        
        row = {
            "bull": bull, "oscillate": osc, "bear": bear,
            "total_return": ret, "sharpe": sharpe,
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "n_trades": metrics.get("n_trades", 0),
            "duration_s": round(elapsed),
        }
        results.append(row)
        
        log(f"  回报={ret:.2f}% 夏普={sharpe:.2f} 回撤={row['max_drawdown']:.2f}% "
            f"胜率={row['win_rate']:.1f}% 交易={row['n_trades']} [{elapsed:.0f}s]")
    
    # 结果输出
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)
    df.to_csv(OPT_DIR / "env_multiplier_results.csv", index=False)
    
    log(f"\n{'='*60}")
    log("📊 综合排序 (按夏普)")
    log(f"{'='*60}")
    for _, row in df.head(10).iterrows():
        log(f"  牛={row['bull']:.1f} 震荡={row['oscillate']:.1f} 熊={row['bear']:.1f} "
            f"→ 回报={row['total_return']:.2f}% 夏普={row['sharpe']:.2f} 回撤={row['max_drawdown']:.2f}%")
    
    best = df.iloc[0]
    log(f"\n🏆 最优: 牛={best['bull']:.1f} 震荡={best['oscillate']:.1f} 熊={best['bear']:.1f}")
    log(f"   夏普={best['sharpe']:.2f} 回报={best['total_return']:.2f}% 回撤={best['max_drawdown']:.2f}%")
    
    # 存最优配置
    best_path = OPT_DIR / "best_env_multipliers.json"
    with open(best_path, "w") as f:
        json.dump({"bull": best["bull"], "oscillate": best["oscillate"], "bear": best["bear"]}, f)
    log(f"📄 最优配置: {best_path}")


if __name__ == "__main__":
    main()
