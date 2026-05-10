#!/usr/bin/env python3
"""
Step 7: 最优组合全量回测
========================
读取 Step 4-6 的最优参数，生成 config_v2.2_optimal.yaml
然后跑全量回测

用法:
    python3 scripts/build_best_config.py
"""

import sys, os, json, subprocess, shutil
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).resolve().parent.parent
CONFIG_DIR = WORKSPACE / "chanfund_fusion"
OPT_DIR = WORKSPACE / "optimization"
BACKTEST_RESULTS = WORKSPACE / "backtest_results"
os.makedirs(BACKTEST_RESULTS, exist_ok=True)

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_opt(path: str) -> dict:
    if not os.path.exists(path):
        log(f"⚠️ 找不到: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    log("=" * 60)
    log("Step 7: 最优组合全量回测")
    log("=" * 60)
    
    # 读取最优参数
    env_best = load_opt(str(OPT_DIR / "best_env_multipliers.json"))
    factor_best = load_opt(str(OPT_DIR / "best_factor_threshold.json"))
    tp_best = load_opt(str(OPT_DIR / "best_tp_params.json"))
    
    log(f"  环境乘数: {env_best}")
    log(f"  因子门槛: {factor_best}")
    log(f"  止盈参数: {tp_best}")
    
    if not any([env_best, factor_best, tp_best]):
        log("❌ 无最优参数, 使用默认值")
        env_best = {"bull": 1.0, "oscillate": 1.0, "bear": 0.5}
        factor_best = {"buy_score_min": 50, "default_score": 55}
        tp_best = {"tp1_pct": 0.06, "tp2_pct": 0.13, "tp_trail_retrace": 0.08, "tp_protect_days": 0}
    
    # 从 v2.1_datafull.yaml 复制并修改
    base_conf = CONFIG_DIR / "config_v2.1_datafull.yaml"
    target_conf = CONFIG_DIR / "config_v2.2_optimal.yaml"
    
    with open(base_conf) as f:
        content = f.read()
    
    # 替换参数
    content = content.replace("version: \"2.1-datafull\"", "version: \"2.2-optimal\"")
    content = content.replace("tp1_pct: 0.06", f"tp1_pct: {tp_best.get('tp1_pct', 0.06)}")
    content = content.replace("tp2_pct: 0.13", f"tp2_pct: {tp_best.get('tp2_pct', 0.13)}")
    content = content.replace("tp_trail_retrace: 0.90", f"tp_trail_retrace: {tp_best.get('tp_trail_retrace', 0.08)}")
    content = content.replace("buy_score_min: 50", f"buy_score_min: {factor_best.get('buy_score_min', 50)}")
    content = content.replace("trending_weight: 1.0", f"trending_weight: {env_best.get('bull', 1.0)}")
    content = content.replace("oscillating_weight: 1.0", f"oscillating_weight: {env_best.get('oscillate', 1.0)}")
    
    with open(target_conf, "w") as f:
        f.write(content)
    log(f"📄 配置: {target_conf}")
    
    # 运行全量回测
    log("\n🚀 启动全量回测...")
    cmd = [
        sys.executable, str(WORKSPACE / "run_chanfund_v20_backtest.py"),
        "--config", str(target_conf),
        "--batched",
    ]
    
    log(f"  命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, cwd=str(WORKSPACE)
        )
        output = result.stdout + result.stderr
        print(output[-2000:] if len(output) > 2000 else output)
    except subprocess.TimeoutExpired:
        log("⏱️ 回测超时(3600s), 检查日志")
        with open(str(WORKSPACE / "v22_optimal.log"), "w") as f:
            f.write("TIMEOUT\n")
        return 1
    except Exception as e:
        log(f"❌ 回测失败: {e}")
        return 1
    
    # 提取结果保存
    with open(str(BACKTEST_RESULTS / "v2.2_optimal.log"), "w") as f:
        f.write(output)
    log("✅ 全量回测完成")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
