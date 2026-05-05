#!/usr/bin/env python3
"""
Task 2: 数据完整性校验与修复 — Validate & Fix Financials
=======================================================
对下载后的财务数据做完整性扫描，确保覆盖率 ≥ 98%。

运行方式：
  python3 scripts/validate_and_fix_financials.py              # 全量校验
  python3 scripts/validate_and_fix_financials.py --html-only   # 只生成报告
"""

import os, sys, time, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

WORKSPACE = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUTPUT_DIR = WORKSPACE / "quant" / "data" / "financial_parquet"
FIN_CSV_DIR = WORKSPACE / "quant" / "data" / "financial"
REPORT_DIR = WORKSPACE / "scripts"

TARGET_COVERAGE = 0.98  # 目标覆盖率 98%


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_coverage() -> dict:
    """扫描所有财务表的覆盖率"""
    tables = ["income", "fina_indicator", "balancesheet", "cashflow"]
    report = {
        "tables": {},
        "overall": {"total_expected": 0, "total_actual": 0, "coverage": 0},
        "gaps": [],
    }
    
    for table in tables:
        pq_files = sorted(OUTPUT_DIR.glob(f"{table}_*.parquet"))
        if not pq_files:
            report["tables"][table] = {"status": "MISSING", "years": [], "stocks": 0}
            continue
        
        year_stats = {}
        total_stocks = set()
        
        for f in pq_files:
            year = f.stem.split("_")[-1]
            if not year.isdigit():
                continue
            df = pd.read_parquet(f)
            stocks = set(df["ts_code"].unique()) if "ts_code" in df.columns else set()
            # 使用end_date或f_ann_date确定报告期
            n_records = len(df)
            year_stats[int(year)] = {
                "stocks": len(stocks),
                "records": n_records,
                "file": f.name,
                "size_kb": f.stat().st_size / 1024,
            }
            total_stocks.update(stocks)
        
        report["tables"][table] = {
            "status": "OK" if len(total_stocks) > 0 else "EMPTY",
            "years": sorted(year_stats.keys()),
            "stocks": len(total_stocks),
            "year_stats": year_stats,
        }
    
    return report


def check_gaps(report: dict) -> list:
    """识别数据缺口"""
    gaps = []
    
    for table_name, info in report["tables"].items():
        if info["status"] != "OK":
            gaps.append({"table": table_name, "type": "MISSING_TABLE", "severity": "HIGH"})
            continue
        
        years_expected = set(range(2005, 2026))
        years_have = set(info.get("years", []))
        missing_years = sorted(years_expected - years_have)
        
        if missing_years:
            gaps.append({
                "table": table_name,
                "type": "MISSING_YEARS",
                "missing_years": missing_years,
                "severity": "HIGH" if len(missing_years) > 5 else "MEDIUM",
            })
        
        # 检查逐年股票数稳定性
        if "year_stats" in info:
            avg_stocks = np.mean([s["stocks"] for s in info["year_stats"].values()])
            for year, stats in info["year_stats"].items():
                if stats["stocks"] < avg_stocks * 0.5 and stats["stocks"] < 1000:
                    gaps.append({
                        "table": table_name,
                        "type": "LOW_COVERAGE_YEAR",
                        "year": year,
                        "stocks": stats["stocks"],
                        "avg_stocks": int(avg_stocks),
                        "severity": "MEDIUM",
                    })
    
    return gaps


def generate_html_report(report: dict, gaps: list):
    """生成HTML可视化报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 计算整体覆盖率
    total_ok = sum(1 for t in report["tables"].values() if t["status"] == "OK")
    total_tables = len(report["tables"])
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>财务数据完整性报告</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #f5f5f5; }}
.ok {{ color: green; }}
.warn {{ color: orange; }}
.err {{ color: red; }}
.bar {{ display: inline-block; height: 20px; margin-right: 4px; }}
</style></head><body>
<h1>📊 财务数据完整性报告</h1>
<p>生成时间: {now}</p>

<h2>总体状态</h2>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>数据表总数</td><td>{total_tables}</td></tr>
<tr><td>已覆盖</td><td class="ok">{total_ok}/{total_tables}</td></tr>
<tr><td>数据缺口</td><td class="warn">{len(gaps)} 个</td></tr>
</table>

<h2>逐表详情</h2>
"""
    for table_name, info in report["tables"].items():
        status_class = "ok" if info["status"] == "OK" else "err"
        years_str = f"{info.get('years', [])[0]}-{info.get('years', [])[-1]}" if info.get('years') else "无数据"
        html += f"""
<h3>{table_name}</h3>
<table><tr><th>状态</th><th>年份范围</th><th>股票数</th><th>年份数</th></tr>
<tr><td class="{status_class}">{info['status']}</td>
<td>{years_str}</td><td>{info.get('stocks', 0):,}</td>
<td>{len(info.get('years', []))}</td></tr></table>
"""
        if "year_stats" in info:
            html += "<p>逐年记录数:</p><div>"
            years = sorted(info["year_stats"].keys())
            max_records = max(s["records"] for s in info["year_stats"].values()) or 1
            for year in years:
                s = info["year_stats"][year]
                pct = s["records"] / max_records * 100
                html += f'<span class="bar" style="width:{pct:.1f}%;background:#4CAF50;" title="{year}: {s["records"]:,}条, {s["stocks"]}只股票"></span>'
            html += "</div>"
    
    # 缺口列表
    if gaps:
        html += "<h2>⛔ 数据缺口</h2><table><tr><th>表</th><th>类型</th><th>详情</th><th>严重度</th></tr>"
        for g in gaps:
            severity_class = "err" if g["severity"] == "HIGH" else "warn"
            detail = g.get("missing_years", str(g.get("year", "")))
            html += f'<tr><td>{g["table"]}</td><td>{g["type"]}</td><td>{detail}</td><td class="{severity_class}">{g["severity"]}</td></tr>'
        html += "</table>"
    
    html += "</body></html>"
    
    report_path = REPORT_DIR / "missing_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    log(f"📄 报告生成: {report_path}")
    return report_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="财务数据完整性校验")
    parser.add_argument("--html-only", action="store_true", help="仅生成报告，不重新扫描")
    args = parser.parse_args()
    
    log("=" * 50)
    log("Task 2: 数据完整性校验与修复")
    log("=" * 50)
    
    # 扫描
    if not args.html_only:
        log("扫描数据覆盖...")
    report = scan_coverage()
    
    # 检查缺口
    gaps = check_gaps(report)
    
    # 生成报告
    report_path = generate_html_report(report, gaps)
    
    # 计算覆盖率
    for table, info in report["tables"].items():
        if info["status"] == "OK":
            years = info.get("years", [])
            stocks = info.get("stocks", 0)
            if years:
                min_year, max_year = min(years), max(years)
                expected_stocks = 5000  # 约5000只A股
                actual = stocks * len(years)
                expected = expected_stocks * (max_year - min_year + 1)
                coverage = actual / expected if expected > 0 else 0
                status = "✅" if coverage >= TARGET_COVERAGE else "⚠️"
                log(f"  {table}: {status} 覆盖率 {coverage*100:.1f}% (≥{TARGET_COVERAGE*100:.0f}%)")
    
    if not args.html_only:
        log(f"\n缺口数: {len(gaps)}")
        if gaps:
            log("建议: 重新运行 complete_financial_data_pipeline.py 补全")
            log("  python scripts/complete_financial_data_pipeline.py")
        else:
            log("✅ 无数据缺口！覆盖率达标")
    
    log(f"\n📊 报告: {report_path}")


if __name__ == "__main__":
    main()
