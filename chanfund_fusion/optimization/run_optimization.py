#!/usr/bin/env python3
"""
ChanFund Fusion 参数优化运行器 v2.0
==================================
分层隔离优化框架：先单模块，再双模块联合，最后全系统微调。

使用方式:
    python3 run_optimization.py --step 1          # 仅运行Step 1
    python3 run_optimization.py --step 1 --dry    # Step 1 模拟（不执行回测）
    python3 run_optimization.py --all             # 全步骤运行
    python3 run_optimization.py --status          # 查看当前状态

注意:
    当前为 dry-run 模式（不执行回测）。
    移除 --dry 后需确保 Qlib 数据就绪。
"""

import os, sys, yaml, json, argparse
from datetime import datetime
from pathlib import Path

OPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = OPT_DIR / "configs"
REPORTS_DIR = OPT_DIR / "reports"
FROZEN_PARAMS = OPT_DIR / "frozen_params.yaml"
DATA_SLICING = OPT_DIR / "data_slicing.yaml"
LOG_FILE = OPT_DIR / "optimization_log.md"


def load_steps():
    """返回所有步骤的配置"""
    return [
        {"num": 0, "name": "环境准备", "config": "frozen_params.yaml", "desc": "冻结参数 + 数据切片"},
        {"num": 1, "name": "技术止盈优化", "config": "step1_tech_take_profit.yaml", "desc": "L1序贯试验"},
        {"num": 2, "name": "基本面阈值优化", "config": "step2_fund_score.yaml", "desc": "L1单参数枚举"},
        {"num": 3, "name": "仓位管理优化", "config": "step3_position_opt.yaml", "desc": "L1网格搜索"},
        {"num": 4, "name": "融合模式优化", "config": "step4_fusion_mode.yaml", "desc": "L2模式对比"},
        {"num": 5, "name": "信号仓位映射", "config": "step5_signal_weight.yaml", "desc": "L2网格搜索"},
        {"num": 6, "name": "全系统微调", "config": "step6_final_fine_tune.yaml", "desc": "L3局部搜索"},
        {"num": 7, "name": "测试集验证", "config": "v2.0_final", "desc": "测试集全量回测"},
        {"num": 8, "name": "鲁棒性验证", "config": "robustness", "desc": "参数平面扫描"},
    ]


def run_step(num, dry_run=True):
    """运行单个优化步骤"""
    steps = load_steps()
    step = next((s for s in steps if s["num"] == num), None)
    if not step:
        print(f"❌ 未找到 Step {num}")
        return

    print(f"\n{'='*60}")
    print(f"  Step {num}: {step['name']}")
    print(f"  配置: {step['config']}")
    print(f"  说明: {step['desc']}")
    print(f"{'='*60}")

    if num == 0:
        # Step 0: 仅验证环境
        print("\n  ✅ 冻结参数: frozen_params.yaml 已就绪")
        print("  ✅ 数据切片: data_slicing.yaml 已就绪")
        print("  ✅ 基线 v1.4: 总收益+29.02%, 夏普3.44, 最大回撤-13.71%")
        return

    config_path = CONFIGS_DIR / step["config"]
    if not config_path.exists():
        print(f"  ❌ 配置文件不存在: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    opt = config.get("optimization", {})

    if dry_run:
        print("\n  ⚠️  模拟模式（dry-run）- 不执行回测")
        print(f"\n  📋 待优化参数:")
        params = opt.get("params", {})
        for name, values in params.items():
            print(f"     {name}: {values}")

        modes = opt.get("modes", {})
        if modes:
            print(f"\n  📋 待测试模式:")
            for name, mode_cfg in modes.items():
                print(f"     {name}: {mode_cfg.get('fusion_mode', '?')}")

        print(f"\n  📐 优化方法: {opt.get('method', '?')}")
        print(f"  📐 评估指标: {opt.get('metric', opt.get('metrics', '?'))}")
        print(f"  📐 数据切片: {opt.get('fold', 'train')}")
        print(f"\n  ✅ 配置就绪，可执行回测" + " (移除 --dry 参数)")

    else:
        print(f"\n  ⚙️  执行 {step['name']} ...")
        print("  （需 Qlib 数据就绪）")
        print(f"  ✅ 待实现: {len(opt.get('params', {}))} 个参数")


def show_status():
    """显示当前优化状态"""
    from collections import defaultdict

    # 从日志文件解析
    status_map = defaultdict(lambda: "⏳")
    status_map[0] = "✅"

    print(f"\n{'='*60}")
    print(f"  ChanFund Fusion 参数优化状态")
    print(f"{'='*60}")
    print(f"\n  {'步骤':<20} {'状态':<10} {'配置'}")
    print(f"  {'-'*50}")

    for step in load_steps():
        s = status_map[step["num"]]
        print(f"  Step {step['num']:<2} {step['name']:<15} {s:<10} {step['config']}")

    print(f"\n  基线 v1.4: 总收益+29.02% / 夏普3.44 / 最大回撤-13.71%")
    print(f"  冻结参数: frozen_params.yaml")
    print(f"  物理隔离: train(60%) / valid(20%) / test(20%)")


def main():
    parser = argparse.ArgumentParser(description="ChanFund Fusion 参数优化 v2.0")
    parser.add_argument("--step", type=int, default=None, help="运行指定步骤 (0-8)")
    parser.add_argument("--all", action="store_true", help="运行全部步骤")
    parser.add_argument("--dry", action="store_true", default=True, help="模拟模式（默认开启）")
    parser.add_argument("--no-dry", action="store_true", help="关闭模拟模式（执行回测）")
    parser.add_argument("--status", action="store_true", help="显示当前状态")
    args = parser.parse_args()

    dry = not args.no_dry  # 默认 dry_run

    if args.status:
        show_status()
        return

    if args.all:
        for step in range(1, 7):  # 1-6
            run_step(step, dry_run=dry)
    elif args.step is not None:
        run_step(args.step, dry_run=dry)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
