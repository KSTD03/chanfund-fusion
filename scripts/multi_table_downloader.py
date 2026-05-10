#!/usr/bin/env python3
"""
多表并行增量下载器 — 处理 6 张新表（资产负债表/现金流量表/分红/业绩预告/审计/主营）

策略：
  - 6 张表轮番下载，每轮每表 50 只（分批错开，避免 API 限流）
  - 表间间隔 60s，股票间隔 4s
  - 进度基于 parquet 文件检测（断点续传，与 crontab 不冲突）
  - 完成后自动合并 + 写完成标记

用法：
    python3 scripts/multi_table_downloader.py              # 正常运行
    python3 scripts/multi_table_downloader.py --status     # 查看进度
    python3 scripts/multi_table_downloader.py --stop       # 发送停止信号
"""

import subprocess, time, json, sys, os
from pathlib import Path
import pandas as pd

TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
WORKSPACE = Path(__file__).resolve().parent.parent
FIN_PQ = WORKSPACE / "quant" / "data" / "financial_parquet"
SCORES = WORKSPACE / "financial_data"
LOG_FILE = SCORES / "multitable_download.log"
STOP_FLAG = SCORES / "multitable_stop.flag"
DONE_FLAG = SCORES / "multitable_done.flag"
STATUS_FILE = SCORES / "multitable_status.json"

FIN_PQ.mkdir(parents=True, exist_ok=True)
SCORES.mkdir(parents=True, exist_ok=True)

# ============ 表定义 ============
TABLES = [
    {
        "name": "balancesheet",
        "fields": "ts_code,ann_date,end_date,total_assets,total_liab,money_cap,accounts_receiv,inventories,fix_assets,intan_assets,total_hldr_eqy_exc_min_int,st_borr,lt_borr,accounts_pay,deferred_inc",
        "batch": 50,  # 资产负债表数据量大，每批少一点
        "pause": 4.0,
    },
    {
        "name": "cashflow",
        "fields": "ts_code,ann_date,end_date,net_profit,c_fr_sale_sg,n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,free_cashflow,c_paid_goods_s,c_paid_to_for_empl,c_paid_for_taxes",
        "batch": 50,
        "pause": 4.0,
    },
    {
        "name": "dividend",
        "fields": "ts_code,end_date,ann_date,div_proc,cash_div,cash_div_tax,record_date,ex_date",
        "batch": 100,
        "pause": 3.0,
    },
    {
        "name": "forecast",
        "fields": "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,last_parent_net",
        "batch": 100,
        "pause": 3.0,
    },
    {
        "name": "fina_audit",
        "fields": "ts_code,ann_date,end_date,audit_result,audit_fees",
        "batch": 100,
        "pause": 3.0,
    },
    {
        "name": "fina_mainbz",
        "fields": "ts_code,end_date,bz_item,bz_sales,bz_profit,bz_cost",
        "batch": 100,
        "pause": 3.0,
    },
]

def lg(m):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", buffering=1) as f:
        f.write(f"[{ts}] {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def get_all_codes():
    cache = WORKSPACE / "scripts" / "fin_temp" / "all_codes.json"
    if cache.exists():
        return json.load(open(cache))
    # 从 API 获取
    p = json.dumps({"api_name":"stock_basic","token":TOKEN,
                     "params":{"exchange":"","list_status":"L"},
                     "fields":"ts_code"})
    r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","10",
        "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
        capture_output=True, text=True, timeout=15)
    d = json.loads(r.stdout)
    if d.get("code") == 0 and d.get("data") and d["data"].get("items"):
        codes = [i[0] for i in d["data"]["items"]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(codes, open(cache, "w"))
        return codes
    return []

def count_progress(table_name):
    """统计某表的已下载股票数"""
    done = set()
    prefix = f"{table_name}_"
    for f in FIN_PQ.glob(f"{prefix}*.parquet"):
        code = f.stem[len(prefix):]
        done.add(code)
    return done

def download_batch(table_name, fields, batch_size, pause):
    """下载一批股票，返回成功数和失败数"""
    codes = get_all_codes()
    if not codes:
        return 0, 0
    done = count_progress(table_name)
    pending = [c for c in codes if c not in done]
    
    if not pending:
        return 0, 0  # 全部完成
    
    batch = pending[:batch_size]
    success, failed = 0, 0
    
    for code in batch:
        p = json.dumps({"api_name": table_name, "token": TOKEN,
                         "params": {"ts_code": code, "start_date": "20050101", "end_date": "20260430"},
                         "fields": fields})
        try:
            r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","15",
                "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
                capture_output=True, text=True, timeout=20)
            d = json.loads(r.stdout)
            if d.get("code") == 0 and d.get("data") and d["data"].get("items"):
                items = d["data"]["items"]
                col_names = fields.split(",") if fields else []
                df = pd.DataFrame(items, columns=col_names)
                outfile = FIN_PQ / f"{table_name}_{code}.parquet"
                df.to_parquet(outfile, compression="snappy", index=False)
                success += 1
            else:
                # 空数据也标记为完成（该股无此表数据）
                success += 1
            time.sleep(pause)
        except Exception as e:
            failed += 1
            lg(f"    ⚠️ {table_name}/{code}: {str(e)[:60]}")
            time.sleep(pause)
    
    return success, failed

def merge_table(table_name):
    """合并单股 parquet 为完整文件"""
    files = sorted(FIN_PQ.glob(f"{table_name}_*.parquet"))
    if not files:
        lg(f"  ⚠️ {table_name}: 无文件可合并")
        return False
    
    lg(f"  合并 {table_name}: {len(files)} 个文件...")
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:
            lg(f"    ⚠️ 读取 {f.name} 失败: {e}")
    
    if parts:
        df = pd.concat(parts, ignore_index=True)
        dedup_cols = [c for c in ["ts_code", "end_date"] if c in df.columns]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
            df = df.sort_values(dedup_cols).reset_index(drop=True)
        out = FIN_PQ / f"{table_name}.parquet"
        df.to_parquet(out, compression="snappy", index=False)
        lg(f"  ✅ {table_name}: {len(df)}行, {df['ts_code'].nunique()}只")
        return True
    return False

def print_status():
    """打印所有表的状态"""
    codes = get_all_codes()
    total = len(codes)
    
    print(f"\n{'='*55}")
    print(f"   多表并行下载进度")
    print(f"{'='*55}")
    print(f"   股票总数: {total}")
    print(f"{'='*55}")
    
    all_tables = [
        ("fina_indicator", "财务指标"),
        ("income", "利润表"),
        ("balancesheet", "资产负债表"),
        ("cashflow", "现金流量表"),
        ("dividend", "分红送股"),
        ("forecast", "业绩预告"),
        ("fina_audit", "审计意见"),
        ("fina_mainbz", "主营构成"),
    ]
    
    for table_name, label in all_tables:
        done = count_progress(table_name)
        merged = "✅" if (FIN_PQ / f"{table_name}.parquet").exists() else " "
        pct = len(done) / total * 100 if total > 0 else 0
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  {bar} {label:8s} | {len(done):>5}/{total} ({pct:5.1f}%) {' 📦' if merged == '✅' else ''}")
    
    # 守护状态
    if STATUS_FILE.exists():
        s = json.load(open(STATUS_FILE))
        print(f"{'='*55}")
        print(f"   运行: {'是' if s.get('running') else '否'}")
        print(f"   轮次: {s.get('cycle', 0)}")
        print(f"   当前表: {s.get('current_table', 'N/A')}")
        print(f"   最后运行: {s.get('last_activity', 'N/A')}")
    
    print(f"{'='*55}\n")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        print_status()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        STOP_FLAG.touch()
        print("🛑 停止信号已发送")
        return
    
    lg("="*60)
    lg("🚀 多表并行增量下载器启动")
    lg(f"   共 {len(TABLES)} 张新表，轮番下载")
    lg("="*60)
    
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    if DONE_FLAG.exists():
        DONE_FLAG.unlink()
    
    cycle = 0
    total_stocks = len(get_all_codes())
    
    # 状态文件
    save_status = {
        "running": True,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle": 0,
    }
    json.dump(save_status, open(STATUS_FILE, "w"))
    
    while True:
        if STOP_FLAG.exists():
            lg("🛑 收到停止信号")
            STOP_FLAG.unlink()
            break
        
        cycle += 1
        cycle_start = time.time()
        lg(f"\n{'─'*55}")
        lg(f"🔄 第 {cycle} 轮 — 处理 {len(TABLES)} 张表")
        
        all_tables_done = True
        
        for ti, tbl in enumerate(TABLES):
            if STOP_FLAG.exists():
                break
            
            name = tbl["name"]
            done = count_progress(name)
            pct = len(done) / total_stocks * 100 if total_stocks > 0 else 0
            
            if len(done) >= total_stocks * 0.98:
                lg(f"  ✅ {name}: 已完成 ({len(done)}/{total_stocks})")
                continue
            
            all_tables_done = False
            
            # 表间错开：每张表延迟不同时间（分钟级偏移）
            stagger_delay = ti * 60  # 60s offset per table
            if stagger_delay > 0:
                lg(f"  ⏳ {name}: 等待 {stagger_delay}s 错峰... ({len(done)}/{total_stocks}, {pct:.1f}%)")
                for _ in range(stagger_delay // 5):
                    if STOP_FLAG.exists():
                        break
                    time.sleep(5)
                if STOP_FLAG.exists():
                    break
            
            lg(f"  ▶ {name}: {len(done)}/{total_stocks} ({pct:.1f}%)")
            s, f = download_batch(name, tbl["fields"], tbl["batch"], tbl["pause"])
            if s > 0 or f > 0:
                done = count_progress(name)
                pct = len(done) / total_stocks * 100
                lg(f"    ✅ 完成: {s} OK, {f} fail → 总进度 {len(done)}/{total_stocks} ({pct:.1f}%)")
            
            # 表间间隔（主间隔，错峰后已经有序）
            if ti < len(TABLES) - 1:
                lg(f"    ⏸ 表间休息 30s...")
                time.sleep(30)
        
        # 更新状态
        save_status = {
            "running": True,
            "cycle": cycle,
            "last_activity": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_table": "all",
        }
        json.dump(save_status, open(STATUS_FILE, "w"))
        
        if all_tables_done:
            lg("\n🎉" + "="*50)
            lg("🎉 所有 6 张新表下载完成！")
            lg("="*50)
            
            # 合并所有表
            lg("\n📦 合并文件...")
            for tbl in TABLES:
                merge_table(tbl["name"])
            
            DONE_FLAG.write_text(f"All tables completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            save_status["running"] = False
            save_status["completed"] = True
            save_status["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            json.dump(save_status, open(STATUS_FILE, "w"))
            
            lg("✅ 全部完成！")
            
            # 打印最终状态
            print_status()
            break
        
        # 轮间等待：整轮完成后等待再开始下一轮
        cycle_elapsed = time.time() - cycle_start
        wait_time = max(60, 600 - cycle_elapsed)  # 至少 60s，最多补到 10 分钟
        lg(f"\n⏳ 轮次耗时: {cycle_elapsed:.0f}s，等待 {wait_time:.0f}s 后下一轮")
        
        for _ in range(int(wait_time // 10)):
            if STOP_FLAG.exists():
                break
            time.sleep(10)
    
    lg("多表下载器退出")

if __name__ == "__main__":
    main()
