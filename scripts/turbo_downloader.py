#!/usr/bin/env python3
"""
🚀 Tushare 极速统一下载器

设计目标：最快、最稳定地完成全量 A 股 8 张财务表下载
策略：
  - 统一调度，不重复、不空闲
  - 按剩余量排序：几乎完成的先搞定
  - 小表大口吃（200/批），大表小口吃（50/批）
  - 每 3 小时保存一次进度快照（可 cron 查询）
  - 完成后自动合并 + 数据检验

用法：
    python3 scripts/turbo_downloader.py                  # 启动（自动后台守护）
    python3 scripts/turbo_downloader.py --status         # 查看进度
    python3 scripts/turbo_downloader.py --stop           # 优雅停止
"""

import subprocess, time, json, sys, os, random
from pathlib import Path
import pandas as pd

# ============ 配置 ============
TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
WORKSPACE = Path(__file__).resolve().parent.parent
FIN_PQ = WORKSPACE / "quant" / "data" / "financial_parquet"
SCORES = WORKSPACE / "financial_data"
LOG = SCORES / "turbo_download.log"
PROGRESS_FILE = SCORES / "turbo_progress.json"
SNAPSHOT_DIR = SCORES / "snapshots"
STOP_FLAG = SCORES / "turbo_stop.flag"
DONE_FLAG = SCORES / "turbo_done.flag"

FIN_PQ.mkdir(parents=True, exist_ok=True)
SCORES.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ============ 表定义（按数据大小排列） ============
# (name, fields, batch_size, api_pause, description)
# pause 大幅缩减: 免费版 Tushare 支持 ~200次/min, 0.3s间隔完全合规
# 失败时有重试+自适应流控保证稳定性
TABLES = [
    ("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,bps", 200, 0.3, "财务指标"),
    ("income", "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps", 150, 0.3, "利润表"),
    ("dividend", "ts_code,end_date,ann_date,div_proc,cash_div,cash_div_tax,record_date,ex_date", 250, 0.3, "分红送股"),
    ("forecast", "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,last_parent_net", 250, 0.3, "业绩预告"),
    ("fina_audit", "ts_code,ann_date,end_date,audit_result,audit_fees", 300, 0.3, "审计意见"),
    ("fina_mainbz", "ts_code,end_date,bz_item,bz_sales,bz_profit,bz_cost", 150, 0.3, "主营构成"),
    ("cashflow", "ts_code,ann_date,end_date,net_profit,c_fr_sale_sg,n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,free_cashflow", 80, 0.3, "现金流量表"),
    ("balancesheet", "ts_code,ann_date,end_date,total_assets,total_liab,money_cap,accounts_receiv,inventories,fix_assets,total_hldr_eqy_exc_min_int,st_borr,lt_borr,deferred_inc", 60, 0.3, "资产负债表"),
]

def lg(m):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {m}"
    with open(LOG, "a", buffering=1) as f:
        f.write(line + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def get_all_codes():
    """获取全量 A 股代码"""
    cache = WORKSPACE / "scripts/fin_temp/all_codes.json"
    if cache.exists():
        return json.load(open(cache))
    
    p = json.dumps({"api_name":"stock_basic","token":TOKEN,
                     "params":{"exchange":"","list_status":"L"},
                     "fields":"ts_code"})
    r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","15",
        "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
        capture_output=True, text=True, timeout=20)
    d = json.loads(r.stdout)
    codes = []
    if d.get("code") == 0 and d.get("data") and d["data"].get("items"):
        codes = [i[0] for i in d["data"]["items"]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(codes, open(cache, "w"))
    return codes

def count_done(table_name):
    """基于文件检测某表已下载股票数"""
    done = set()
    prefix = f"{table_name}_"
    for f in FIN_PQ.glob(f"{prefix}*.parquet"):
        code = f.stem[len(prefix):]
        # 过滤掉合并后的文件
        if '_' not in code and code and len(code) > 5:
            done.add(code)
    return done

def tushare_curl(api_name, fields, **params):
    """单次 curl 调用，返回 DataFrame
    
    - 正常数据 → 返回 DataFrame（可能为空行）
    - API 返回空但未报错 → 返回空 DataFrame（该股无此表数据）
    - 网络/API 异常 → 返回 None（需重试）
    """
    p = json.dumps({"api_name": api_name, "token": TOKEN,
                     "params": params, "fields": fields})
    try:
        r = subprocess.run(["curl","-s","--connect-timeout","8","--max-time","20",
            "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
            capture_output=True, text=True, timeout=25)
        d = json.loads(r.stdout)
        if d.get("code") == 0 and d.get("data"):
            # code=0 且 data 存在 → API 正常响应
            items = d["data"].get("items", [])
            col_names = fields.split(",") if fields else []
            if items:
                return pd.DataFrame(items, columns=col_names)
            else:
                # 空数据但合法（该股无此表数据）→ 返回空 DataFrame
                return pd.DataFrame(columns=col_names)
        # API 报错（code != 0 等）→ 网络/业务异常
        return None
    except:
        return None

def download_one(table_name, fields, codes_batch, pause):
    """下载一批股票到单股 parquet，含重试+自适应流控
    
    返回 (成功数, 失败数)
    注：API 返回空数据（该股无此表）也计为成功
    """
    s, f = 0, 0
    consecutive_fails = 0
    current_pause = pause
    
    for i, code in enumerate(codes_batch):
        # ===== 尝试下载（含1次重试） =====
        df = None
        for attempt in range(2):  # 首次 + 1次重试
            df = tushare_curl(table_name, fields,
                              ts_code=code, start_date="20050101", end_date="20260430")
            if df is not None:
                break  # 成功则跳出重试
            if attempt == 0:
                time.sleep(1.0 + random.uniform(0, 0.5))  # 重试前短退避
        
        # ===== 处理结果 =====
        if df is not None:
            # df 可能是空 DataFrame（该股无此表数据）→ 仍保存空白 parquet 标记完成
            outfile = FIN_PQ / f"{table_name}_{code}.parquet"
            df.to_parquet(outfile, compression="snappy", index=False)
            s += 1
            consecutive_fails = 0
        else:
            f += 1
            consecutive_fails += 1
        
        # ===== 自适应流控 =====
        if consecutive_fails >= 5:
            actual_pause = 1.0  # 连续失败5次 → 降速
        elif consecutive_fails >= 3:
            actual_pause = 0.5
        else:
            actual_pause = current_pause
        
        # 加入微量随机抖动，防同步突增
        time.sleep(actual_pause + random.uniform(0, 0.1))
        
        # 每50条报一次进度（大batch时有用）
        if (i + 1) % 50 == 0 and len(codes_batch) > 50:
            lg(f"    ⏩ batch内: {i+1}/{len(codes_batch)} ({s}OK/{f}fail)")
    
    return s, f

def merge_table(table_name):
    """合并单股 parquet 为完整文件"""
    files = sorted(FIN_PQ.glob(f"{table_name}_*.parquet"))
    if not files:
        return None
    
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except:
            pass
    
    if parts:
        df = pd.concat(parts, ignore_index=True)
        dedup = [c for c in ["ts_code", "end_date"] if c in df.columns]
        if dedup:
            df = df.drop_duplicates(subset=dedup, keep="last")
            df = df.sort_values(dedup).reset_index(drop=True)
        out = FIN_PQ / f"{table_name}.parquet"
        df.to_parquet(out, compression="snappy", index=False)
        return df
    return None

def build_progress():
    """构建完整进度报告"""
    codes = get_all_codes()
    total = len(codes) if codes else 5512
    
    entries = []
    for name, _, _, _, desc in TABLES:
        done = count_done(name)
        pct = len(done) / total * 100
        merged = (FIN_PQ / f"{name}.parquet").exists()
        remaining = total - len(done)
        entries.append({
            "name": name, "desc": desc,
            "done": len(done), "total": total,
            "pct": round(pct, 1),
            "remaining": remaining,
            "merged": merged,
        })
    
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tables": entries}

def print_progress(entries=None, header="进度报告"):
    """打印进度表格"""
    if entries is None:
        entries = build_progress()["tables"]
    
    lines = [f"\n{'='*60}", f"  📊 {header}", f"{'='*60}"]
    for e in entries:
        d, t = e["done"], e["total"]
        pct = e["pct"]
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        emoji = "✅" if pct >= 99 else ("👍" if pct >= 80 else ("⏳" if pct > 0 else "❌"))
        rem = e["remaining"]
        lines.append(f"  {bar} {emoji} {e['desc']:8s} | {d:>5}/{t} ({pct:5.1f}%) 剩余{rem:>4}")
    
    # 总计
    total_done = sum(e["done"] for e in entries)
    total_all = sum(e["total"] for e in entries)
    total_pct = total_done / total_all * 100
    bar_len = 20
    filled = int(total_pct / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"{'─'*60}")
    lines.append(f"  {bar} 📊 总计: {total_done:>8,}/{total_all:,} ({total_pct:.1f}%)")
    lines.append(f"{'='*60}\n")
    
    return "\n".join(lines)

def save_snapshot(progress):
    """保存 3 小时快照"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"progress_{ts}.json"
    json.dump(progress, open(path, "w"), indent=2)

def do_validation():
    """数据检验"""
    lg("\n📊 开始数据检验...")
    
    for name, _, _, _, desc in TABLES:
        fpath = FIN_PQ / f"{name}.parquet"
        if not fpath.exists():
            lg(f"  ⚠️ {name} ({desc}): 未合并，跳过")
            continue
        
        df = pd.read_parquet(fpath)
        codes = df["ts_code"].nunique()
        rows = len(df)
        dmin = str(df["end_date"].min())[:10]
        dmax = str(df["end_date"].max())[:10]
        
        expected = "2005-01-01" <= dmin and "2026-04-30" >= dmax
        stock_pct = codes / 5512 * 100
        nulls = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
        
        lg(f"  📁 {desc:8s} ({name}):")
        lg(f"      行数={rows:>8,}  股票={codes}({stock_pct:.1f}%)  时间={dmin}~{dmax}  {'✅' if expected else '❌'}")
        if nulls:
            lg(f"      空值: {nulls}")
    
    report = build_progress()
    report["validation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(report, open(SCORES / "turbo_validation_report.json", "w"), indent=2)
    lg("✅ 数据检验完成")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        p = build_progress()
        print(print_progress(p["tables"]))
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        STOP_FLAG.touch()
        print("🛑 停止信号已发送")
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        do_validation()
        return
    
    lg("=" * 60)
    lg("🚀 Tushare 极速统一下载器 启动")
    lg(f"   共 {len(TABLES)} 张表, 自适应优先调度")
    lg(f"   目标: 最快+最稳定完成全量A股数据")
    lg("=" * 60)
    
    if STOP_FLAG.exists(): STOP_FLAG.unlink()
    if DONE_FLAG.exists(): DONE_FLAG.unlink()
    
    codes = get_all_codes()
    total = len(codes) if codes else 5512
    lg(f"   股票总数: {total}")
    
    cycle = 0
    last_snapshot_time = 0
    SNAPSHOT_INTERVAL = 3600  # 1 小时快照（更细粒度恢复点）
    
    while True:
        if STOP_FLAG.exists():
            lg("🛑 收到停止信号")
            STOP_FLAG.unlink()
            break
        
        cycle += 1
        cycle_start = time.time()
        
        # ===== 检测所有表进度 =====
        progress = build_progress()
        entries = progress["tables"]
        
        # 按剩余量排序：剩余越少越优先
        active = [e for e in entries if e["remaining"] > 0]
        completed = [e for e in entries if e["remaining"] == 0]
        
        if not active:
            lg("\n🎉" + "="*50)
            lg("🎉 全部 {len(TABLES)} 张表下载完成！")
            lg("="*50)
            
            # 合并所有表
            lg("\n📦 合并文件...")
            for name, _, _, _, desc in TABLES:
                df = merge_table(name)
                if df is not None:
                    lg(f"  ✅ {desc}: {len(df)}行, {df['ts_code'].nunique()}只")
                else:
                    lg(f"  ⚠️ {desc}: 合并失败或无文件")
            
            # 数据检验
            do_validation()
            
            DONE_FLAG.write_text(f"All done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            lg("✅ 全部完成！")
            
            # 最终进度
            lg(print_progress(header="最终进度"))
            break
        
        # ===== 打印本轮进度 =====
        lg(f"\n{'─'*55}")
        lg(f"🔄 第{cycle}轮 — {len(active)}张表待下载, {len(completed)}张已完成")
        
        for e in active:
            lg(f"  {e['desc']:8s}: {e['done']}/{e['total']} ({e['pct']:.1f}%) 剩余{e['remaining']}")
        
        # ===== 3小时快照 =====
        now = time.time()
        if now - last_snapshot_time >= SNAPSHOT_INTERVAL:
            save_snapshot(progress)
            last_snapshot_time = now
            lg(f"📸 3小时快照已保存 -> snapshots/progress_{time.strftime('%Y%m%d_%H%M%S')}.json")
        
        # ===== 按优先级处理各表 =====
        # 按剩余量升序排列（almost done first）
        active_sorted = sorted(active, key=lambda e: e["remaining"])
        
        for ei, e in enumerate(active_sorted):
            if STOP_FLAG.exists():
                break
            
            name = e["name"]
            remaining = e["remaining"]
            
            # 查表配置
            tbl_cfg = None
            for t in TABLES:
                if t[0] == name:
                    tbl_cfg = t
                    break
            if not tbl_cfg:
                continue
            
            _, fields, batch_size, pause, desc = tbl_cfg
            
            # 自适应batch：剩余少就一次搞定
            actual_batch = min(batch_size, remaining)
            
            # 取待下载股票
            done = count_done(name)
            pending = [c for c in codes if c not in done]
            if not pending:
                continue
            
            batch_codes = pending[:actual_batch]
            
            lg(f"\n  ▶ {desc} ({name}): {len(batch_codes)}只 [剩余{remaining}]")
            
            batch_t0 = time.time()
            s, f = download_one(name, fields, batch_codes, pause)
            batch_t = time.time() - batch_t0
            
            # 更新进度
            done_new = count_done(name)
            pct_new = len(done_new) / total * 100
            lg(f"    ✅ {s} OK, {f} fail → {len(done_new)}/{total} ({pct_new:.1f}%) [{batch_t:.0f}s]")
            
            # 表间短暂间隔（仅防突增）
            if ei < len(active_sorted) - 1:
                time.sleep(3)
        
        # ===== 单轮耗时 & 紧凑下一轮 =====
        cycle_t = time.time() - cycle_start
        
        # 检查是否还有待下载
        still_active = build_progress()["tables"]
        still_pending = [e for e in still_active if e["remaining"] > 0]
        
        if not still_pending:
            continue  # 会触发上方的完成检测
        
        # 紧凑轮间：不空等，只给5s防止请求堆积
        wait = max(0, 8 - cycle_t)
        if wait > 0:
            lg(f"\n⏸ 本轮{cycle_t:.0f}s, {wait:.0f}s后下一轮...")
            time.sleep(wait)
        else:
            lg(f"\n⚡ 立即下一轮（本轮{cycle_t:.0f}s）")
    
    lg("极速下载器退出")

if __name__ == "__main__":
    main()
