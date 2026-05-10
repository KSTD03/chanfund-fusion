#!/usr/bin/env python3
"""
下载完成监控 + 数据检验看门狗

在后台运行，每 5 分钟检测一次 cron 增量下载是否完成。
完成时自动执行数据检验并通知。
"""

import subprocess, time, json, sys
from pathlib import Path
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
FIN_PQ = WORKSPACE / "quant" / "data" / "financial_parquet"
SCORES = WORKSPACE / "financial_data"
LOG_FILE = SCORES / "watchdog.log"
DONE_FLAG = SCORES / "download_done.flag"
STOP_FLAG = SCORES / "watchdog_stop.flag"

def lg(m):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", buffering=1) as f:
        f.write(f"[{ts}] {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def get_all_codes():
    cache = WORKSPACE / "scripts" / "fin_temp" / "all_codes.json"
    if cache.exists():
        return json.load(open(cache))
    return []

def check_progress():
    """检查当前下载进度"""
    codes = get_all_codes()
    if not codes:
        return 0, 0, 0
    
    fina_done = set()
    income_done = set()
    
    for f in FIN_PQ.glob("fina_indicator_*.parquet"):
        code = f.stem.replace("fina_indicator_", "")
        fina_done.add(code)
    for f in FIN_PQ.glob("income_*.parquet"):
        code = f.stem.replace("income_", "")
        income_done.add(code)
    
    total = len(codes)
    return total, len(fina_done), len(income_done)

def is_download_complete():
    """判断是否完成（>=95%）"""
    total, fina, income = check_progress()
    if total == 0:
        return False
    return (fina >= total * 0.95) and (income >= total * 0.95)

def do_data_validation():
    """数据检验：全时间段、全A股"""
    lg("\n" + "="*60)
    lg("📊 执行数据检验...")
    lg("="*60)
    
    fina_file = FIN_PQ / "fina_indicator.parquet"
    income_file = FIN_PQ / "income.parquet"
    results = {}
    
    for name, fpath in [("fina_indicator", fina_file), ("income", income_file)]:
        if not fpath.exists():
            lg(f"  ❌ {name}: 文件不存在，尝试合并...")
            # 尝试合并
            files = sorted(FIN_PQ.glob(f"{name}_*.parquet"))
            if files:
                parts = [pd.read_parquet(f) for f in files]
                df = pd.concat(parts, ignore_index=True)
                df = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
                df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
                df.to_parquet(fpath, compression="snappy", index=False)
                lg(f"    ✅ 合并完成: {len(df)}行, {df['ts_code'].nunique()}只")
            else:
                lg(f"    ❌ 无数据文件可合并")
                results[name] = {"status": "NO_DATA"}
                continue
        
        df = pd.read_parquet(fpath)
        total_rows = len(df)
        unique_codes = df["ts_code"].nunique()
        date_min = df["end_date"].min()
        date_max = df["end_date"].max()
        
        expected_start = "2005-01-01"
        expected_end = "2026-04-30"
        
        time_ok = str(date_min)[:10] <= expected_start and str(date_max)[:10] >= expected_end[:10]
        
        # 按股票检查时间覆盖
        incomplete_stocks = 0
        stock_date_ranges = df.groupby("ts_code")["end_date"].agg(["min", "max"])
        for _, row in stock_date_ranges.iterrows():
            if str(row["min"])[:10] > expected_start or str(row["max"])[:10] < expected_end[:10]:
                incomplete_stocks += 1
        
        # 空值
        nulls = df.isnull().sum().to_dict()
        null_cols = {k: v for k, v in nulls.items() if v > 0}
        
        stock_pct = unique_codes / 5512 * 100
        
        lg(f"\n  📁 {name}:")
        lg(f"    - 总行数: {total_rows:,}")
        lg(f"    - 股票数: {unique_codes} ({stock_pct:.1f}% of A股)")
        lg(f"    - 时间范围: {str(date_min)[:10]} ~ {str(date_max)[:10]}")
        lg(f"    - 时间覆盖完整: {'✅' if time_ok else '❌ 不完整'}")
        lg(f"    - 时间覆盖不足的股票: {incomplete_stocks}")
        if null_cols:
            lg(f"    - 空值列: {null_cols}")
        else:
            lg(f"    - 空值: ✅ 无")
        
        results[name] = {
            "status": "OK",
            "rows": int(total_rows),
            "stocks": int(unique_codes),
            "date_range": [str(date_min)[:10], str(date_max)[:10]],
            "time_complete": time_ok,
            "incomplete_stocks": int(incomplete_stocks),
            "nulls": {k: int(v) for k, v in null_cols.items()} if null_cols else {},
            "stock_pct": round(stock_pct, 1)
        }
    
    # 检查 scores.parquet
    scores_file = SCORES / "scores.parquet"
    if scores_file.exists():
        sdf = pd.read_parquet(scores_file)
        lg(f"\n  📁 scores.parquet:")
        lg(f"    - 总行数: {len(sdf):,}")
        lg(f"    - 股票数: {sdf['ts_code'].nunique()}")
        lg(f"    - 评分均值: {sdf['fund_score'].mean():.1f}")
        lg(f"    - 评分中位数: {sdf['fund_score'].median():.1f}")
        results["scores"] = {
            "rows": int(len(sdf)),
            "stocks": int(sdf["ts_code"].nunique()),
            "mean_score": round(float(sdf["fund_score"].mean()), 1),
            "median_score": round(float(sdf["fund_score"].median()), 1)
        }
    
    # 综合结论
    all_ok = all(
        r.get("status") == "OK" for r in results.values() if isinstance(r, dict)
    )
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "overall": "PASS" if all_ok else "ISSUES"
    }
    json.dump(report, open(SCORES / "validation_report.json", "w"), indent=2, ensure_ascii=False)
    
    # 生成可读报告
    report_file = SCORES / "validation_report.md"
    with open(report_file, "w") as f:
        f.write("# 数据检验报告\n\n")
        f.write(f"**检验时间**: {report['timestamp']}\n")
        f.write(f"**结论**: {'✅ 通过' if all_ok else '⚠️ 存在异常'}\n\n")
        f.write("## 各表详情\n\n")
        for name, r in results.items():
            if not isinstance(r, dict):
                continue
            f.write(f"### {name}\n")
            f.write(f"- 状态: {'✅' if r.get('status') == 'OK' else '❌'}\n")
            f.write(f"- 行数: {r.get('rows', '?'):,}\n")
            f.write(f"- 股票数: {r.get('stocks', '?')} 只\n")
            f.write(f"- 时间范围: {r.get('date_range', ['?','?'])[0]} ~ {r.get('date_range', ['?','?'])[1]}\n")
            f.write(f"- 时间覆盖完整: {'✅' if r.get('time_complete') else '❌'}\n")
            if r.get("incomplete_stocks", 0) > 0:
                f.write(f"- 时间覆盖不足: {r['incomplete_stocks']} 只 ⚠️\n")
            if r.get("nulls"):
                f.write(f"- 空值字段: {r['nulls']}\n")
            f.write("\n")
    
    lg("\n" + "="*60)
    lg(f"📊 数据检验 {'✅ 通过' if all_ok else '⚠️ 存在异常'}")
    lg(f"报告已保存: financial_data/validation_report.json")
    lg(f"可读报告: financial_data/validation_report.md")
    lg("="*60)
    
    return report

def main():
    lg("="*60)
    lg("👀 下载完成监控看门狗启动")
    lg("   每 5 分钟检测一次进度")
    lg("   检测到完成 → 自动合并 + 数据检验")
    lg("="*60)
    
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    
    last_progress = ""
    stable_no_progress = 0
    
    while True:
        if STOP_FLAG.exists():
            lg("🛑 收到停止信号")
            STOP_FLAG.unlink()
            break
        
        if DONE_FLAG.exists():
            lg("✅ 下载完成标记已存在，跳过")
            break
        
        total, fina, inc = check_progress()
        pct = (fina + inc) / (total * 2) * 100 if total > 0 else 0
        
        progress_str = f"{fina}/{total} + {inc}/{total} ({pct:.1f}%)"
        
        if progress_str != last_progress:
            lg(f"📊 进度: fina={fina}/{total}, income={inc}/{total} ({pct:.1f}%)")
            last_progress = progress_str
            stable_no_progress = 0
        else:
            stable_no_progress += 1
            if stable_no_progress >= 12:  # 1 hour no progress
                lg(f"⚠️ 已 1 小时无进展 ({progress_str})，可能卡住了")
        
        if is_download_complete():
            lg("\n🎉" + "="*50)
            lg("🎉 下载完成！全量 A 股数据已就绪")
            lg("="*50)
            
            # 执行数据检验
            report = do_data_validation()
            
            # 写完成标记
            DONE_FLAG.write_text(f"Download completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            lg("✅ 完成标记已写入")
            break
        
        time.sleep(300)  # 5 分钟
    
    lg("👀 看门狗退出")

if __name__ == "__main__":
    main()
