#!/usr/bin/env python3
"""Runner: 顺序执行 R1 → R2 → R3 三轮优化"""
import subprocess, sys, time, os
from pathlib import Path

SCRIPTS = [
    ("R1 (环境依赖TP/SL)", "/home/quant/.openclaw/workspace/scripts/round_r1_v25_param.py"),
    ("R2 (成交量过滤)",    "/home/quant/.openclaw/workspace/scripts/round_r2_volume_filter.py"),
    ("R3 (R1+R2组合)",     "/home/quant/.openclaw/workspace/scripts/round_r3_combined.py"),
]

total_start = time.time()
for name, script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"🔄 启动 {name}  {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    sys.stdout.flush()
    t0 = time.time()
    ret = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ {name} 完成 ({elapsed/60:.1f}min)  returncode={ret.returncode}")
    print(f"{'='*60}")
    sys.stdout.flush()
    if ret.returncode != 0:
        print(f"❌ {name} 失败，终止后续")
        break

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"🏁 三轮优化全部完成! 总耗时: {total_elapsed/60:.1f}min")
print(f"{'='*60}")
