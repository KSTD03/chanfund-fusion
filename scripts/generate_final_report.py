#!/usr/bin/env python3
"""
Step 8: 最终报告生成
====================
汇总 v2.0 → v2.1 → v2.2 绩效变化
输出: output/v2.2_final_report.md

用法:
    python3 scripts/generate_final_report.py
"""

import sys, os, re, json
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "output"
BACKTEST_RESULTS = WORKSPACE / "backtest_results"
COMPARISON_DIR = WORKSPACE / "comparison"
OPT_DIR = WORKSPACE / "optimization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def parse_summary(text: str) -> dict:
    """从报告文本中提取指标"""
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
        "n_signals": r"总信号[：:]\s*(\d+)",
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


def read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def main():
    log("=" * 60)
    log("Step 8: 最终报告生成")
    log("=" * 60)
    
    # 读取各版本报告
    v20_text = read_file(str(BACKTEST_RESULTS / "v2.0_full_summary.md"))
    v21_text = read_file(str(BACKTEST_RESULTS / "v2.1_datafull_summary.md"))
    v22_text = read_file(str(BACKTEST_RESULTS / "v2.2_optimal_summary.md"))
    
    # 如果独立的summary文件不存在，尝试从回测日志提取
    if not v20_text:
        v20_text = read_file(str(WORKSPACE / "v20_full_test.out"))
    if not v21_text:
        v21_text = "（v2.1 回测报告待生成）"
    if not v22_text:
        v22_text = "（v2.2 回测报告待生成）"
    
    v20 = parse_summary(v20_text)
    v21 = parse_summary(v21_text)
    v22 = parse_summary(v22_text)
    
    # 读取最优参数
    env_cfg = read_file(str(OPT_DIR / "best_env_multipliers.json"))
    factor_cfg = read_file(str(OPT_DIR / "best_factor_threshold.json"))
    tp_cfg = read_file(str(OPT_DIR / "best_tp_params.json"))
    
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v) if v else "N/A"
    
    lines = [
        "# ChanFund Fusion v2.2 — 最终优化报告",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 本报告是您唯一需要查看的最终交付物",
        "",
        "## 一、版本演进概览",
        "",
        "| 版本 | 说明 | 状态 |",
        "|------|------|------|",
        "| v1.4 | 基线版本 (2021-2026, 4442只) | ✅ 已知基线 |",
        "| v2.0 | 信号因子化+市场环境分类+共振得分 | ✅ 已完成 |",
        "| v2.1 | + 全量Tushare财务数据补全 | 📥 下载中 |",
        "| v2.2 | + 网格搜索最优参数 | 🔄 待运行 |",
        "",
        "## 二、绩效对比",
        "",
        "| 指标 | v2.0 | v2.1-datafull | v2.2-optimal | 相对于v1.4基线 |",
        "|------|:---:|:---:|:---:|:---:|",
    ]
    
    # 关键指标对比
    key_metrics = [
        ("total_return", "总收益率(%)"),
        ("ann_return", "年化收益率(%)"),
        ("sharpe", "夏普比率"),
        ("max_drawdown", "最大回撤(%)"),
        ("win_rate", "胜率(%)"),
        ("n_trades", "交易次数"),
        ("avg_profit", "平均盈利(%)"),
        ("avg_loss", "平均亏损(%)"),
        ("best_trade", "最佳单笔(%)"),
        ("worst_trade", "最差单笔(%)"),
        ("n_signals", "总信号数"),
        ("final_nav", "最终净值"),
    ]
    
    for key, label in key_metrics:
        v0_v = v20.get(key)
        v1_v = v21.get(key)
        v2_v = v22.get(key)
        lines.append(f"| {label} | {fmt(v0_v)} | {fmt(v1_v)} | {fmt(v2_v)} | — |")
    
    # 最优配置
    lines.extend([
        "",
        "## 三、最终推荐配置表",
        "",
        "| 参数分类 | 参数名 | 默认值 | 最优值 | 说明 |",
        "|----------|--------|:------:|:------:|------|",
    ])
    
    try:
        if env_cfg:
            env = json.loads(env_cfg)
            lines.append(f"| 环境乘数 | 牛市乘数 | 1.0 | {env.get('bull', 1.0)} | 趋势市仓位倍数 |")
            lines.append(f"| 环境乘数 | 震荡乘数 | 1.0 | {env.get('oscillate', 1.0)} | 震荡市仓位倍数 |")
            lines.append(f"| 环境乘数 | 熊市乘数 | 0.5 | {env.get('bear', 0.5)} | 下跌市仓位倍数 |")
    except: pass
    
    try:
        if factor_cfg:
            fac = json.loads(factor_cfg)
            lines.append(f"| 因子门槛 | buy_score_min | 50 | {fac.get('buy_score_min', 50)} | 最低买入得分 |")
            lines.append(f"| 因子门槛 | 默认评分 | 55 | {fac.get('default_score', 55)} | 无数据时默认分 |")
    except: pass
    
    try:
        if tp_cfg:
            tp = json.loads(tp_cfg)
            lines.append(f"| 止盈参数 | tp1 (第一止盈) | 6% | {tp.get('tp1_pct', 0.06):.0%} | 首次减仓30% |")
            lines.append(f"| 止盈参数 | tp2 (第二止盈) | 13% | {tp.get('tp2_pct', 0.13):.0%} | 二次减仓50% |")
            lines.append(f"| 止盈参数 | 峰值回落清仓 | 10% | {tp.get('tp_trail_retrace', 0.08):.0%} | 峰值回落比例 |")
            lines.append(f"| 止盈参数 | 保护期 | 0天 | {tp.get('tp_protect_days', 0)}天 | 开仓后不触发止盈 |")
    except: pass
    
    # 绩效分析
    lines.extend([
        "",
        "## 四、绩效变化曲线分析",
        "",
        "### 4.1 收益率演进",
    ])
    
    if v20.get("total_return") is not None:
        lines.append(f"- v2.0: {v20['total_return']:.2f}%")
    if v21.get("total_return") is not None:
        lines.append(f"- v2.1: {v21['total_return']:.2f}% (vs v2.0: {v21['total_return'] - v20.get('total_return', 0):+.2f}%)")
    if v22.get("total_return") is not None:
        lines.append(f"- v2.2: {v22['total_return']:.2f}% (vs v2.1: {v22['total_return'] - v21.get('total_return', 0):+.2f}%)")
    
    lines.extend([
        "",
        "### 4.2 回撤对比",
    ])
    
    if v20.get("max_drawdown") is not None:
        lines.append(f"- v2.0 最大回撤: {v20['max_drawdown']:.2f}%")
    if v21.get("max_drawdown") is not None:
        lines.append(f"- v2.1 最大回撤: {v21['max_drawdown']:.2f}%")
    if v22.get("max_drawdown") is not None:
        lines.append(f"- v2.2 最大回撤: {v22['max_drawdown']:.2f}%")
    
    # 差距分析
    lines.extend([
        "",
        "## 五、与v1.4基线的差距分析",
        "",
        "### v1.4已知表现",
        "- 收益率: ~29.02% (2021-2026, 4442只)",
        "- 夏普: ~3.44",
        "- 回撤: ~-13.71%",
        "- 胜率: ~58.6%",
        "",
        "### v2.x缺口",
    ])
    
    for v, name, v_label in [(v20, "2.0", "v2.0"), (v21, "2.1", "v2.1"), (v22, "2.2", "v2.2")]:
        if v.get("total_return") is not None and v.get("sharpe") is not None:
            gap_ret = v["total_return"] - 29.02
            gap_sharpe = v["sharpe"] - 3.44
            lines.append(f"- **{v_label}**: 收益率差距={gap_ret:+.2f}%, 夏普差距={gap_sharpe:+.2f}")
    
    lines.extend([
        "",
        "### 后续优化建议",
        "1. 升级PIT数据系统：确保评分的时间精确性",
        "2. 引入更丰富的因子：营收质量、现金流分析、行业对比",
        "3. 多周期信号融合：将日线信号与30min/周线信号联合验证",
        "4. 动态止盈调整：根据市场环境动态调整止盈阈值",
        "5. 机器学习排序：用XGBoost/LightGBM替代手工评分",
        "",
        "## 六、交付物清单",
        "",
        "| 文件 | 用途 | 状态 |",
        "|------|------|------|",
        "| output/v2.2_final_report.md | ✅ 本文件 - 唯一需要查看的总结报告 | 📄 当前文件 |",
        "| financial_data/scores.parquet | 补全后的基本面因子 | 📥 待完成 |",
        "| backtest_results/v2.1_datafull_summary.md | 数据补全效果 | 📥 待完成 |",
        "| backtest_results/v2.2_optimal_summary.md | 最优参数全量结果 | 🔄 待运行 |",
        "| chanfund_fusion/config_v2.2_optimal.yaml | 最终配置快照 | 📄 已创建 |",
        "| optimization/env_multiplier_results.csv | 环境乘数搜索记录 | 🔄 待运行 |",
        "| optimization/factor_threshold_results.csv | 因子门槛搜索记录 | 🔄 待运行 |",
        "| optimization/tp_param_results.csv | 止盈参数搜索记录 | 🔄 待运行 |",
        "",
        "---",
        "",
        "> **免责声明**: 本报告仅供研究学习参考，不构成任何投资建议。",
        "> 历史回测结果不代表未来收益。",
    ])
    
    with open(str(OUTPUT_DIR / "v2.2_final_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"📄 最终报告: {OUTPUT_DIR / 'v2.2_final_report.md'}")


if __name__ == "__main__":
    main()
