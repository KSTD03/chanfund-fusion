#!/usr/bin/env python3
"""
Step 3: v2.0 vs v2.1 对比
========================
量化基本面数据补全的边际收益

用法:
    python3 scripts/compare_versions.py --baseline backtest_results/v2.0_full_summary.md --new backtest_results/v2.1_datafull_summary.md
"""

import sys, os, re, argparse
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def parse_summary(path: str) -> dict:
    """从回测报告提取指标"""
    if not os.path.exists(path):
        log(f"⚠️ 文件不存在: {path}")
        return {}
    
    with open(path) as f:
        text = f.read()
    
    patterns = {
        "total_return": r"总收益率[：:]\s*([\-\d.]+)%",
        "ann_return": r"年化[收益率]*[：:]\s*([\-\d.]+)%",
        "sharpe": r"夏普[比率]*[：:]\s*([\-\d.]+)",
        "max_drawdown": r"最大回撤[：:]\s*([\-\d.]+)%",
        "win_rate": r"胜率[：:]\s*([\d.]+)%",
        "n_trades": r"交易[数]*[：:]\s*(\d+)",
        "profit_trades": r"盈利交易[：:]\s*(\d+)",
        "loss_trades": r"亏损交易[：:]\s*(\d+)",
        "avg_profit": r"平均盈利[：:]\s*([\d.]+)%",
        "avg_loss": r"平均亏损[：:]\s*([\-\d.]+)%",
        "best_trade": r"最佳单笔[：:]\s*([\-\d.]+)%",
        "worst_trade": r"最差单笔[：:]\s*([\-\d.]+)%",
        "final_nav": r"最终净值[：:]\s*([\d,]+)",
    }
    
    metrics = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1).replace(",", "")
            try:
                metrics[key] = float(val)
            except:
                metrics[key] = val
    
    return metrics


def generate_report(v0: dict, v1: dict, output_path: str):
    """生成对比报告"""
    lines = [
        "# v2.0 vs v2.1 数据补全效果对比",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> v2.0: 仅baostock财务数据(2023Q4-2024Q4 5个季度)",  
        "> v2.1: Tushare全量财务数据(2005-2026 fina_indicator+income)",
        "",
        "## 核心指标对比",
        "",
        "| 指标 | v2.0 | v2.1-datafull | 变化 | 方向 |",
        "|------|:---:|:---:|:---:|:---:|",
    ]
    
    key_metrics = [
        ("total_return", "总收益率(%)"),
        ("ann_return", "年化收益率(%)"),
        ("sharpe", "夏普比率"),
        ("max_drawdown", "最大回撤(%)"),
        ("win_rate", "胜率(%)"),
        ("n_trades", "交易次数"),
        ("avg_profit", "平均盈利(%)"),
        ("avg_loss", "平均亏损(%)"),
    ]
    
    for key, label in key_metrics:
        v0_v = v0.get(key, None)
        v1_v = v1.get(key, None)
        if v0_v is None and v1_v is None:
            continue
        if v0_v is None: v0_v = "N/A"
        if v1_v is None: v1_v = "N/A"
        
        change = ""
        direction = ""
        if isinstance(v0_v, (int, float)) and isinstance(v1_v, (int, float)):
            diff = v1_v - v0_v
            if abs(diff) < 0.01:
                change = "0.00"
                direction = "➡️"
            else:
                change = f"{diff:+.2f}"
                # 回撤越小越好，其余越大越好
                if key in ("max_drawdown",):
                    direction = "✅" if diff > 0 else "❌"
                else:
                    direction = "✅" if diff > 0 else "❌"
        
        v0_s = f"{v0_v:.2f}" if isinstance(v0_v, float) else str(v0_v)
        v1_s = f"{v1_v:.2f}" if isinstance(v1_v, float) else str(v1_v)
        
        lines.append(f"| {label} | {v0_s} | {v1_s} | {change} | {direction} |")
    
    # 额外细节
    lines.extend([
        "",
        "## 分析结论",
        "",
    ])
    
    # 收益率变化
    tr0 = v0.get("total_return")
    tr1 = v1.get("total_return")
    if tr0 and tr1:
        diff = tr1 - tr0
        if diff > 2:
            lines.append(f"- **基本面数据补全效果显著**: 收益率从 {tr0:.2f}% 提升至 {tr1:.2f}% (+{diff:.2f}%)")
        elif diff > 0.5:
            lines.append(f"- **基本面数据有正向贡献**: 收益率从 {tr0:.2f}% 提升至 {tr1:.2f}% (+{diff:.2f}%)")
        elif diff > -0.5:
            lines.append(f"- **基本面数据影响中性**: 收益率从 {tr0:.2f}% 变为 {tr1:.2f}% (变化{diff:+.2f}%)")
        else:
            lines.append(f"- **基本面数据产生负向影响**: 收益率从 {tr0:.2f}% 降至 {tr1:.2f}% ({diff:.2f}%)")
            lines.append(f"  ⚠️ 需检查评分模型或PIT逻辑")
    
    # 回撤变化
    dd0 = v0.get("max_drawdown")
    dd1 = v1.get("max_drawdown")
    if dd0 and dd1:
        diff = dd1 - dd0
        if diff > 0:
            lines.append(f"- **最大回撤改善**: 从 {dd0:.2f}% 缩至 {dd1:.2f}% (+{diff:.2f}%)")
        else:
            lines.append(f"- **最大回撤恶化**: 从 {dd0:.2f}% 扩至 {dd1:.2f}% ({diff:.2f}%)")
    
    # 交易频率变化
    nt0 = v0.get("n_trades")
    nt1 = v1.get("n_trades")
    if nt0 and nt1:
        ratio = nt1 / nt0 * 100 if nt0 > 0 else 0
        if ratio < 80:
            lines.append(f"- **交易频率下降**: 从 {nt0:.0f} 笔降至 {nt1:.0f} 笔 ({ratio:.0f}%)")
            lines.append(f"  → 基本面过滤更严格, 信号质量提升")
        elif ratio > 120:
            lines.append(f"- **交易频率上升**: 从 {nt0:.0f} 笔升至 {nt1:.0f} 笔 ({ratio:.0f}%)")
            lines.append(f"  → 基本面评分释放了更多信号")
    
    lines.extend([
        "",
        "## 建议",
        "",
        "1. 如果v2.1效果明显优于v2.0 → 继续进入Step 4-6调优",
        "2. 如果效果接近 → 保持v2.1配置, 调优重点放在参数优化",
        "3. 如果效果退化 → 检查财务数据PIT逻辑, 确认评分正确性",
        "4. 无论结果如何, scores.parquet都是后续优化的基础",
        "",
        "---",
        "",
        "### 原始数据",
        "",
        f"- v2.0 报告: {v0.get('_source', 'N/A')}",
        f"- v2.1 报告: {v1.get('_source', 'N/A')}",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"📄 对比报告: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="v2.0 vs v2.1 对比")
    parser.add_argument("--baseline", default=str(WORKSPACE / "backtest_results" / "v2.0_full_summary.md"))
    parser.add_argument("--new", default=str(WORKSPACE / "backtest_results" / "v2.1_datafull_summary.md"))
    args = parser.parse_args()
    
    log("📊 v2.0 vs v2.1 对比")
    v0 = parse_summary(args.baseline)
    v1 = parse_summary(args.new)
    
    if not v0:
        log(f"⚠️ baseline 报告为空, 仅输出v2.1")
    if not v1:
        log(f"⚠️ v2.1 报告为空, 仅输出v2.0")
    
    v0["_source"] = args.baseline
    v1["_source"] = args.new
    
    output = OUTPUT_DIR / "v2.0_vs_v2.1.md"
    generate_report(v0, v1, str(output))


if __name__ == "__main__":
    main()
