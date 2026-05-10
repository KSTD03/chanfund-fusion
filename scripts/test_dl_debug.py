#!/usr/bin/env python3
"""超稳健版 - 每步打日志"""
import sys, os, time, json, subprocess, traceback
from pathlib import Path

LOG = open("scripts/fin_debug.log", "w", buffering=1)
def lg(m): 
    ts = time.strftime("%H:%M:%S")
    LOG.write(f"[{ts}] {m}\n"); LOG.flush()
    print(f"[{ts}] {m}", flush=True)

lg("=== 开始 ===")

TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"

def curl(n, f="", **kw):
    p = json.dumps({"api_name":n,"token":TOKEN,"params":kw,"fields":f})
    for a in range(3):
        try:
            r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","15",
                "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
                capture_output=True, text=True, timeout=25)
            d = json.loads(r.stdout)
            if d["code"]!=0: raise Exception(d.get("msg",""))
            if not d.get("data") or not d["data"].get("items"): return []
            return d["data"]["items"]
        except Exception as e:
            if a<2: time.sleep(5*(a+1))
            else: raise

lg("获取股票列表...")
its = curl("stock_basic", fields="ts_code", exchange="", list_status="L")
codes = [i[0] for i in its]
lg(f"共{len(codes)}只: {codes[0]}..{codes[-1]}")

import pandas as pd
FIN = Path("quant/data/financial_parquet")
FIN.mkdir(parents=True, exist_ok=True)

lg("开始下载 fina_indicator...")
t0 = time.time(); all_dfs = []; bi = 0
PAUSE = 2.0

for i, c in enumerate(codes):
    if i >= 10: break  # 只测试10只
    lg(f"[{i}] {c}...")
    try:
        t1 = time.time()
        its = curl("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,ocf_per_share,bps",
                   ts_code=c, start_date="20050101", end_date="20260430")
        el = time.time()-t1
        cols = ["ts_code","ann_date","end_date","eps","roe","gross_margin","ocf_per_share","bps"]
        df = pd.DataFrame(its, columns=cols)
        all_dfs.append(df)
        lg(f"  ✅ {len(df)}行 ({el:.1f}s)")
        time.sleep(PAUSE)
    except Exception as e:
        lg(f"  ❌ {type(e).__name__}: {str(e)}")
        time.sleep(10)
        continue
    
    if (i+1)%100 == 0:
        el = time.time()-t0
        lg(f"  [{i+1}/{len(codes)}] ({el:.0f}s)")

if all_dfs:
    pd.concat(all_dfs, ignore_index=True).to_parquet(FIN/"test_10.parquet", compression="snappy")
    lg("✅ 保存 test_10.parquet")

lg(f"完成! {(time.time()-t0)/60:.1f}min")
LOG.close()
print("DONE", flush=True)
