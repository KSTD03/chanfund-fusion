#!/usr/bin/env python3
"""
Tushare 增量下载守护进程 — 连续运行模式

策略：
  - 每 15 分钟处理一批（100 只 × 2 张表）
  - 如果某批次失败数 > 5（不稳定），间歇增加 5 分钟
  - 连续 3 批稳定后，间歇恢复至 15 分钟基线
  - 全部完成后自动做数据检验
  - 后台运行（nohup / tmux）

用法：
    python3 scripts/download_daemon.py              # 正常启动
    python3 scripts/download_daemon.py --status     # 查看进度 + 守护状态
    python3 scripts/download_daemon.py --stop       # 发送停止标记
"""

import subprocess, time, json, sys, os, signal
from pathlib import Path
import pandas as pd

# ============ 配置 ============
WORKSPACE = Path(__file__).resolve().parent.parent
SCRIPT = WORKSPACE / "scripts" / "tushare_incremental_download.py"
PROGRESS = WORKSPACE / "financial_data" / "download_progress.json"
STATUS_FILE = WORKSPACE / "financial_data" / "daemon_status.json"
LOG_FILE = WORKSPACE / "financial_data" / "daemon_download.log"
SCORES_DIR = WORKSPACE / "financial_data"
FIN_PQ = WORKSPACE / "quant" / "data" / "financial_parquet"
FIN_CSV = WORKSPACE / "quant" / "data" / "financial"
DONE_FLAG = WORKSPACE / "financial_data" / "download_done.flag"

# 自适应间歇参数
INTERVAL_BASE = 15  # 基线间歇（分钟）
INTERVAL_MIN = 15   # 最小间歇
INTERVAL_MAX = 60   # 最大间歇
FAIL_THRESHOLD = 5  # 单批失败超过此值视为不稳定
STABLE_RUNS = 3     # 连续稳定批次数后重置间歇

# 停止文件（touch 此文件则优雅停止）
STOP_FLAG = WORKSPACE / "financial_data" / "daemon_stop.flag"

def lg(m):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", buffering=1) as f:
        f.write(f"[{ts}] {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load_progress():
    if not PROGRESS.exists():
        return {"completed": {"fina_indicator": [], "income": []},
                "failed": {"fina_indicator": [], "income": []},
                "tables": {}}
    return json.load(open(PROGRESS))

def get_total_stocks():
    """从进度文件获取股票总数"""
    p = load_progress()
    total = 5512  # 默认值
    for t in ("fina_indicator", "income"):
        if t in p.get("tables", {}):
            total = p["tables"][t].get("total", total)
    return total

def check_completion():
    """检查是否全部完成（>=95%）"""
    p = load_progress()
    total = get_total_stocks()
    fina_done = len(p["completed"].get("fina_indicator", []))
    income_done = len(p["completed"].get("income", []))
    return (fina_done >= total * 0.95) and (income_done >= total * 0.95)

def run_batch():
    """执行一批下载，返回成功/失败/错误信息"""
    lg("▶ 启动本批下载...")
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True, text=True, timeout=300)  # 5 min timeout
        out = result.stdout[-2000:] if result.stdout else ""
        err = result.stderr[-500:] if result.stderr else ""
        
        # 解析输出，估算失败数
        fail_count = 0
        success_count = 0
        for line in out.split("\n"):
            if "❌" in line:
                fail_count += 1
            if "OK" in line or "done" in line.lower():
                # 尝试提取数字
                pass
            if "ALL DONE" in line or "🎉" in line:
                lg("  📣 脚本报告全部完成!")
        
        # 重新从进度文件读精确数字
        p = load_progress()
        fina_fail = len(p["failed"].get("fina_indicator", []))
        inc_fail = len(p["failed"].get("income", []))
        total_fail = fina_fail + inc_fail
        
        if result.returncode != 0:
            lg(f"  ⚠️ 脚本退出码={result.returncode}, err={err[:100]}")
            return {"success": False, "fail_count": max(total_fail, 10), "output": out}
        
        return {"success": True, "fail_count": total_fail, "output": out}
    except subprocess.TimeoutExpired:
        lg("  ⚠️ 本批超时(>5min)，可能触发了限流")
        return {"success": False, "fail_count": 99, "output": "TIMEOUT"}
    except Exception as e:
        lg(f"  ❌ 本批异常: {e}")
        return {"success": False, "fail_count": 99, "output": str(e)}

def save_status(status_data):
    """保存守护进程状态"""
    json.dump(status_data, open(STATUS_FILE, "w"), indent=2, ensure_ascii=False)

def do_data_validation():
    """数据检验：全时间段、全A股"""
    lg("\n" + "="*60)
    lg("📊 开始数据检验...")
    lg("="*60)
    
    # 1. 检查 fina_indicator
    fina_file = FIN_PQ / "fina_indicator.parquet"
    income_file = FIN_PQ / "income.parquet"
    
    results = {}
    
    for name, fpath in [("fina_indicator", fina_file), ("income", income_file)]:
        if not fpath.exists():
            lg(f"  ❌ {name}: 文件不存在!")
            results[name] = {"status": "MISSING"}
            continue
        
        df = pd.read_parquet(fpath)
        total_rows = len(df)
        unique_codes = df["ts_code"].nunique()
        date_min = df["end_date"].min()
        date_max = df["end_date"].max()
        
        # 时间范围检验：2005-01-01 ~ 2026-04-30
        expected_start = "2005-01-01"
        expected_end = "2026-04-30"
        
        time_ok = str(date_min)[:10] <= expected_start and str(date_max)[:10] >= expected_end[:10]
        
        # 空值检验
        nulls = df.isnull().sum().to_dict()
        null_cols = {k: v for k, v in nulls.items() if v > 0}
        
        # 股票数量检验：A股约 5512 只
        stock_pct = unique_codes / 5512 * 100
        
        lg(f"\n  📁 {name}:")
        lg(f"    - 总行数: {total_rows:,}")
        lg(f"    - 股票数: {unique_codes} ({stock_pct:.1f}% of A股)")
        lg(f"    - 时间范围: {str(date_min)[:10]} ~ {str(date_max)[:10]}")
        lg(f"    - 时间覆盖完整: {'✅' if time_ok else '❌ 不完整'}")
        if null_cols:
            lg(f"    - 空值列: {null_cols}")
        else:
            lg(f"    - 空值: ✅ 无")
        
        results[name] = {
            "status": "OK",
            "rows": total_rows,
            "stocks": unique_codes,
            "date_range": (str(date_min)[:10], str(date_max)[:10]),
            "time_complete": time_ok,
            "nulls": null_cols,
            "stock_pct": stock_pct
        }
    
    # 2. 检查 scores.parquet
    scores_file = SCORES_DIR / "scores.parquet"
    if scores_file.exists():
        sdf = pd.read_parquet(scores_file)
        lg(f"\n  📁 scores.parquet:")
        lg(f"    - 总行数: {len(sdf):,}")
        lg(f"    - 股票数: {sdf['ts_code'].nunique()}")
        lg(f"    - 评分均值: {sdf['fund_score'].mean():.1f}")
        lg(f"    - 评分中位数: {sdf['fund_score'].median():.1f}")
        results["scores"] = {
            "rows": len(sdf),
            "stocks": sdf["ts_code"].nunique(),
            "mean_score": float(sdf["fund_score"].mean()),
            "median_score": float(sdf["fund_score"].median())
        }
    
    # 3. 保存检验报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "overall": "PASS" if all(r.get("status") == "OK" for r in results.values() if isinstance(r, dict)) else "ISSUES"
    }
    json.dump(report, open(SCORES_DIR / "validation_report.json", "w"), indent=2, ensure_ascii=False)
    
    lg("\n" + "="*60)
    lg(f"📊 数据检验 {'✅ 通过' if report['overall'] == 'PASS' else '⚠️ 存在问题'}")
    lg("报告已保存: financial_data/validation_report.json")
    lg("="*60)
    
    return report

def print_status():
    """打印当前进度"""
    p = load_progress()
    total = get_total_stocks()
    fina_done = len(p["completed"].get("fina_indicator", []))
    inc_done = len(p["completed"].get("income", []))
    fina_fail = len(p["failed"].get("fina_indicator", []))
    inc_fail = len(p["failed"].get("income", []))
    
    pct = (fina_done + inc_done) / (total * 2) * 100
    
    print(f"\n{'='*50}")
    print(f"   Tushare 增量下载进度")
    print(f"{'='*50}")
    print(f"  股票总数:     {total}")
    print(f"  fina_indicator: {fina_done}/{total} ({fina_done/total*100:.1f}%) 失败:{fina_fail}")
    print(f"  income:        {inc_done}/{total} ({inc_done/total*100:.1f}%) 失败:{inc_fail}")
    print(f"  总进度:       {pct:.1f}%")
    print(f"  最后运行:     {p['tables'].get('fina_indicator',{}).get('last_run','N/A')}")
    
    # 守护状态
    if STATUS_FILE.exists():
        s = json.load(open(STATUS_FILE))
        print(f"\n  守护进程状态:")
        print(f"    批次: {s.get('batch_num', '?')}")
        print(f"    间歇: {s.get('current_interval', '?')} 分钟")
        print(f"    稳定计数: {s.get('stable_count', '?')}")
        print(f"    运行中: {'是' if s.get('running') else '否'}")
        print(f"    最后批次: {s.get('last_batch_time', '?')}")
    
    print(f"{'='*50}\n")

def main():
    # --status 模式
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        print_status()
        return
    
    # --stop 模式
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        STOP_FLAG.touch()
        with open(STATUS_FILE, "w") as f:
            json.dump({"running": False, "stopped_by": "user", "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        print("🛑 停止标记已发送，下一轮将退出")
        return
    
    lg("="*60)
    lg("🚀 Tushare 增量下载守护进程启动")
    lg(f"   基线间歇: {INTERVAL_BASE} 分钟")
    lg(f"   批次大小: 100 只/批")
    lg(f"   不稳定阈值: 单批失败 > {FAIL_THRESHOLD}")
    lg("="*60)
    
    # 清除旧停止标记
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    
    current_interval = INTERVAL_BASE
    stable_count = 0
    batch_num = 0
    
    # 保存初始状态
    save_status({
        "running": True,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_num": 0,
        "current_interval": current_interval,
        "stable_count": 0
    })
    
    while True:
        # 检查停止标记
        if STOP_FLAG.exists():
            lg("🛑 收到停止信号，守护进程退出")
            STOP_FLAG.unlink()
            save_status({
                "running": False,
                "stop_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "batch_num": batch_num,
                "reason": "user_stop"
            })
            break
        
        # 检查是否全部完成
        if check_completion():
            lg("\n🎉" + "="*50)
            lg("🎉 所有股票数据下载完成！")
            lg("="*50 + "\n")
            
            # 执行数据检验
            report = do_data_validation()
            
            # 写完成标记
            DONE_FLAG.write_text(f"Download completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 输出最终状态
            print_status()
            
            # 保存最终状态
            save_status({
                "running": False,
                "completed": True,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "batch_num": batch_num,
                "validation": report.get("overall", "UNKNOWN")
            })
            
            lg("\n✅ 下载 + 数据检验全部完成！")
            lg(f"   检验报告: financial_data/validation_report.json")
            lg(f"   完成标记: financial_data/download_done.flag")
            break
        
        batch_num += 1
        batch_start = time.time()
        lg(f"\n{'─'*50}")
        lg(f"📦 第 {batch_num} 批 — 当前间歇 {current_interval} 分钟")
        
        # 先打印当前进度
        p = load_progress()
        total = get_total_stocks()
        fina_done = len(p["completed"].get("fina_indicator", []))
        inc_done = len(p["completed"].get("income", []))
        pct = (fina_done + inc_done) / (total * 2) * 100
        lg(f"📊 进度: fina={fina_done}/{total} income={inc_done}/{total} ({pct:.1f}%)")
        
        # 执行本批
        result = run_batch()
        batch_elapsed = time.time() - batch_start
        
        lg(f"⏱️ 本批耗时: {batch_elapsed:.0f}s")
        
        # 自适应间歇调整
        new_failures = result.get("fail_count", 0)
        if new_failures > FAIL_THRESHOLD or not result.get("success", True):
            # 不稳定，增加间歇
            stable_count = 0
            current_interval = min(current_interval + 5, INTERVAL_MAX)
            lg(f"⚠️ 检测到不稳定 (失败={new_failures})")
            lg(f"   → 间歇增加至 {current_interval} 分钟")
        else:
            stable_count += 1
            if stable_count >= STABLE_RUNS and current_interval > INTERVAL_BASE:
                current_interval = max(current_interval - 5, INTERVAL_BASE)
                lg(f"✅ 连续 {STABLE_RUNS} 批稳定，间歇恢复至 {current_interval} 分钟")
                stable_count = 0
            elif stable_count >= STABLE_RUNS:
                lg(f"✅ 运行稳定 (连续 {STABLE_RUNS} 批)")
                stable_count = 0
            else:
                lg(f"✅ 本批正常 (稳定计数: {stable_count}/{STABLE_RUNS})")
        
        # 更新状态
        save_status({
            "running": True,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_batch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_num": batch_num,
            "batch_elapsed_s": int(batch_elapsed),
            "current_interval": current_interval,
            "stable_count": stable_count,
            "last_fail_count": new_failures
        })
        
        # 打印下一批倒计时
        lg(f"⏳ 下一批在 {current_interval} 分钟后...")
        
        # 等待间歇
        # 分小段等待，每30秒检查一次停止标记
        wait_seconds = current_interval * 60
        for remaining in range(wait_seconds, 0, -30):
            if STOP_FLAG.exists():
                lg("🛑 等待期间收到停止信号")
                break
            time.sleep(min(30, remaining))
        
        # 如果等待期间被停止，退出循环
        if STOP_FLAG.exists():
            continue

if __name__ == "__main__":
    main()
