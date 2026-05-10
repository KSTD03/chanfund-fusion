#!/usr/bin/env python3
"""
Tushare 分时增量下载 — 每小时执行一次，每次 100 只股票 × 2 张表
48 小时覆盖 5,512 只全部 A 股

断点续传：已成功的不重复下载
失败跳过：记录到 failed_list，下次重试
完成自动收尾：与 baostock 合并重建 scores.parquet

用法：
    python3 scripts/tushare_incremental_download.py              # 独立执行
    python3 scripts/tushare_incremental_download.py --force      # 强制重试失败列表
    python3 scripts/tushare_incremental_download.py --status     # 查看进度
"""

import subprocess, time, json, sys, os
from pathlib import Path
import pandas as pd

# ============ 配置 ============
TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
WORKSPACE = Path(__file__).resolve().parent.parent
FIN_PQ = WORKSPACE / "quant" / "data" / "financial_parquet"
FIN_CSV = WORKSPACE / "quant" / "data" / "financial"
SCORES = WORKSPACE / "financial_data"
PROGRESS = SCORES / "download_progress.json"
LOG_FILE = SCORES / "hourly_download.log"
BATCH_SIZE = 100  # 每次处理 100 只
PAUSE = 4.0       # 请求间隔（保持 ≤ 20/min 避免触发代理限流）

FIN_PQ.mkdir(parents=True, exist_ok=True); SCORES.mkdir(parents=True, exist_ok=True)

def lg(m):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", buffering=1) as f:
        f.write(f"[{ts}] {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def tushare_api(api_name, fields="", **params):
    """单次 Tushare API 调用（curl + 超时保护）"""
    payload = json.dumps({"api_name": api_name, "token": TOKEN, "params": params, "fields": fields})
    try:
        r = subprocess.run(
            ["curl", "-s", "--connect-timeout", "5", "--max-time", "15",
             "-X", "POST", URL, "-H", "Content-Type: application/json", "-d", payload],
            capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            raise Exception(f"curl exit={r.returncode}")
        d = json.loads(r.stdout)
        if d.get("code") != 0:
            msg = d.get("msg", f"code={d['code']}")
            if "rate limited" in msg.lower() or "limit" in msg.lower():
                print(f"[RL] rate limited, wait 60s...", flush=True)
                time.sleep(30)
            raise Exception(msg)
        if not d.get("data") or not d["data"].get("items"):
            return []
        return d["data"]["items"]
    except:
        raise

def get_stock_list(force_refresh=False):
    """获取全量股票代码列表"""
    cache = WORKSPACE / "scripts" / "fin_temp" / "all_codes.json"
    if cache.exists() and not force_refresh:
        return json.load(open(cache))
    items = tushare_api("stock_basic", fields="ts_code", exchange="", list_status="L")
    codes = [i[0] for i in items]
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(codes, open(cache, "w"))
    return codes

def load_progress():
    """加载下载进度"""
    if not PROGRESS.exists():
        return {"completed": {}, "failed": {}, "tables": {}}
    return json.load(open(PROGRESS))

def save_progress(p):
    """保存下载进度"""
    json.dump(p, open(PROGRESS, "w"), indent=2)

def download_table(table, fields, codes, progress, table_key):
    """增量下载一张表"""
    completed = set(progress["completed"].get(table_key, []))
    failed = set(progress["failed"].get(table_key, []))
    all_codes_set = set(codes)
    
    # 找待下载的股票
    pending = sorted(all_codes_set - completed)
    batch = pending[:BATCH_SIZE]
    
    if not batch:
        lg(f"  {table_key}: 全部完成 (0 pending)")
        return 0
    
    # 首次重试失败列表
    retry_failed = failed & set(batch)
    if retry_failed:
        lg(f"  {table_key}: 重试 {len(retry_failed)} 只失败股票")
    
    success, fail_count = 0, 0
    for code in batch:
        try:
            items = tushare_api(table, fields, ts_code=code,
                                start_date="20050101", end_date="20260430")
            time.sleep(PAUSE)
            if items:
                df = pd.DataFrame(items, columns=fields.split(",") if fields else None)
                outfile = FIN_PQ / f"{table_key}_{code}.parquet"
                df.to_parquet(outfile, compression="snappy", index=False)
                success += 1
                completed.add(code)
                if code in failed:
                    failed.discard(code)
            else:
                # 空数据也算成功（该股无此表数据）
                completed.add(code)
                if code in failed:
                    failed.discard(code)
        except Exception as e:
            fail_count += 1
            failed.add(code)
            lg(f"    ❌ {code}: {str(e)[:60]}")
            time.sleep(5)
    
    # 更新进度
    progress["completed"][table_key] = list(completed)
    progress["failed"][table_key] = list(failed)
    progress["tables"][table_key] = {
        "total": len(codes),
        "done": len(completed),
        "failed": len(failed),
        "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_progress(progress)
    
    lg(f"  {table_key}: {success} OK, {fail_count} fail ({len(completed)}/{len(codes)} done)")
    return success

def merge_table_files(table_key):
    """合并该表的所有单股 parquet 为完整文件"""
    files = sorted(FIN_PQ.glob(f"{table_key}_*.parquet"))
    if not files:
        return None
    
    lg(f"  合并 {table_key}: {len(files)} 个文件...")
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except:
            pass
    
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
        df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
        out = FIN_PQ / f"{table_key}.parquet"
        df.to_parquet(out, compression="snappy", index=False)
        lg(f"  ✅ {table_key}: {len(df)}行, {df['ts_code'].nunique()}只")
        return df
    return None

def rebuild_scores():
    """> 与 baostock 合并重建 scores.parquet"""
    lg("\n📊 重建 scores.parquet...")
    
    # 1. 加载 baostock CSV 评分
    csv_rows = []
    for fpath in sorted(FIN_CSV.glob("financial_*.csv")):
        df = pd.read_csv(fpath)
        date_key = fpath.stem.replace("financial_", "")
        if len(date_key) == 8:
            date_key = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"
        cols = {c.strip(): c for c in df.columns}
        for _, r in df.iterrows():
            code = str(r.get(cols.get("股票代码", "code"), "")).strip()
            if not code or code == "nan": continue
            code_str = str(int(float(code))) if code.replace(".","").isdigit() else code
            if not (code_str.startswith("sh.") or code_str.startswith("sz.")):
                code_str = f"sh.{code_str}" if code_str.startswith(("6","9")) else f"sz.{code_str}"
            roe = float(r.get(cols.get("净资产收益率","roe"),0) or 0)/100
            gm = float(r.get(cols.get("销售毛利率","gross_margin"),0) or 0)/100
            rg = float(r.get(cols.get("营业总收入-同比增长","revenue_growth"),0) or 0)/100
            eps = float(r.get(cols.get("每股收益","eps"),0) or 0)
            s = 0
            if roe >= 0.20: s += 30
            elif roe >= 0.10: s += 20
            elif roe >= 0.05: s += 10
            if gm >= 0.40: s += 25
            elif gm >= 0.25: s += 15
            elif gm >= 0.15: s += 10
            if rg >= 0.20: s += 25
            elif rg >= 0.10: s += 20
            elif rg >= 0: s += 15
            elif rg >= -0.20: s += 5
            else: s -= 10
            if eps > 0: s += 20
            elif eps > -0.5: s += 5
            else: s -= 5
            csv_rows.append({"ts_code": code_str, "end_date": date_key, "fund_score": max(0, min(100, s))})
    
    csv_df = pd.DataFrame(csv_rows) if csv_rows else pd.DataFrame()
    lg(f"  baostock: {len(csv_df)}行, {csv_df['ts_code'].nunique() if not csv_df.empty else 0}只")
    
    # 2. 加载 Tushare fina_indicator 评分
    fina_file = FIN_PQ / "fina_indicator.parquet"
    fina_df = pd.read_parquet(fina_file) if fina_file.exists() else pd.DataFrame()
    
    tushare_rows = []
    if not fina_df.empty:
        for _, r in fina_df.iterrows():
            code = r["ts_code"]
            ed = str(r["end_date"])[:10]
            roe = float(r["roe"] if pd.notna(r.get("roe")) else 0) / 100
            gm = float(r["gross_margin"] if pd.notna(r.get("gross_margin")) else 0) / 100
            eps = float(r["eps"] if pd.notna(r.get("eps")) else 0)
            s = 0
            if roe >= 0.20: s += 30
            elif roe >= 0.10: s += 20
            elif roe >= 0.05: s += 10
            if gm >= 0.40: s += 25
            elif gm >= 0.25: s += 15
            elif gm >= 0.15: s += 10
            if eps > 0: s += 20
            elif eps > -0.5: s += 5
            else: s -= 5
            tushare_rows.append({"ts_code": code, "end_date": ed, "fund_score": max(0, min(100, s))})
    
    tushare_df = pd.DataFrame(tushare_rows) if tushare_rows else pd.DataFrame()
    lg(f"  Tushare: {len(tushare_df)}行, {tushare_df['ts_code'].nunique() if not tushare_df.empty else 0}只")
    
    # 3. 合并：baostock为主，Tushare补充
    if not csv_df.empty and not tushare_df.empty:
        combined = pd.concat([csv_df, tushare_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "end_date"], keep="first")
        combined = combined.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    elif not csv_df.empty:
        combined = csv_df
    elif not tushare_df.empty:
        combined = tushare_df
    else:
        lg("  ⚠️ 无数据源")
        return False
    
    combined.to_parquet(SCORES / "scores.parquet", compression="snappy", index=False)
    lg(f"  ✅ scores.parquet: {len(combined)}行, {combined['ts_code'].nunique()}只, mean={combined['fund_score'].mean():.0f}")
    return True

def check_completion(progress, codes):
    """检查是否全部完成（>=95%即视为完成）"""
    total = len(codes)
    fina_done = len(progress["completed"].get("fina_indicator", []))
    income_done = len(progress["completed"].get("income", []))
    
    all_done = (fina_done >= total * 0.95) and (income_done >= total * 0.95)
    if all_done:
        lg("\n🎉 全部股票已下载完成！")
        # 合并文件
        merge_table_files("fina_indicator")
        merge_table_files("income")
        # 重建评分
        rebuild_scores()
        # 写完成标记
        with open(SCORES / "pipeline.log", "w") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALL DONE\n")
        lg("✅ 完成标记已写入")
    else:
        pct = (fina_done + income_done) / (total * 2) * 100
        lg(f"📊 总进度: {pct:.1f}% (fina={fina_done}/{total}, income={income_done}/{total})")

def main():
    time.sleep(5)  # 启动延迟，避免触发限流
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        # 状态查询
        p = load_progress()
        codes = get_stock_list()
        total = len(codes)
        fina = len(p["completed"].get("fina_indicator", []))
        inc = len(p["completed"].get("income", []))
        print(f"股票总数: {total}")
        print(f"fina_indicator: {fina}/{total} ({fina/total*100:.1f}%)")
        print(f"income:        {inc}/{total} ({inc/total*100:.1f}%)")
        print(f"失败: fina={len(p['failed'].get('fina_indicator',[]))}, "
              f"income={len(p['failed'].get('income',[]))}")
        return
    
    start = time.time()
    codes = get_stock_list()
    progress = load_progress()
    
    lg(f"🔰 分时增量下载 — batch={BATCH_SIZE}, 共{len(codes)}只股票")
    
    # 下载 fina_indicator
    n1 = download_table("fina_indicator",
        "ts_code,ann_date,end_date,eps,roe,gross_margin,bps",
        codes, progress, "fina_indicator")
    
    # 下载 income
    n2 = download_table("income",
        "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps",
        codes, progress, "income")
    
    elapsed = time.time() - start
    lg(f"⏱️ 本轮: 处理 {n1+n2} 只请求, 耗时 {elapsed/60:.1f} 分钟")
    
    # 检查是否全部完成
    check_completion(progress, codes)
    
    # 如果有效据则合并
    if n1 > 0:
        fina_file = FIN_PQ / "fina_indicator.parquet"
        if not fina_file.exists():
            merge_table_files("fina_indicator")

if __name__ == "__main__":
    main()
