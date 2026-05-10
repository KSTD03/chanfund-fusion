#!/usr/bin/env python3
"""下载 income 表"""
import subprocess, time, json
from pathlib import Path
import pandas as pd

TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
FIN = Path("/home/quant/.openclaw/workspace/quant/data/financial_parquet")
TEMP = Path("/home/quant/.openclaw/workspace/scripts/fin_temp")
LOG = open(FIN.parent.parent / "scripts" / "fin_income.log", "w", buffering=1)
def lg(m):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {m}", flush=True)
    LOG.write(f"[{ts}] {m}\n")
    LOG.flush()

def curl(n, f="", **kw):
    p = json.dumps({"api_name":n,"token":TOKEN,"params":kw,"fields":f})
    r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","10",
        "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
        capture_output=True, text=True, timeout=15)
    d = json.loads(r.stdout)
    if d["code"] != 0: return []
    if not d.get("data") or not d["data"].get("items"): return []
    return d["data"]["items"]

lg("Get stock list...")
its = curl("stock_basic", fields="ts_code", exchange="", list_status="L")
codes = [i[0] for i in its]
lg(f"{len(codes)} stocks")

sf = TEMP / "income_done.json"
done = set(json.load(open(sf))) if sf.exists() else set()
lg(f"Download income: {len(done)}/{len(codes)} done")

all_dfs, bi, t0 = [], 0, time.time()
for i, c in enumerate(codes):
    if c in done: continue
    try:
        its = curl("income", "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps",
                   ts_code=c, start_date="20050101", end_date="20260430")
        time.sleep(2.0)
        if its:
            all_dfs.append(pd.DataFrame(its, columns=["ts_code","ann_date","end_date","total_revenue","revenue","basic_eps"]))
            done.add(c)
    except:
        done.add(c)
        time.sleep(2)
    if (i+1) % 200 == 0 or (i+1) == len(codes):
        if all_dfs:
            o = pd.concat(all_dfs, ignore_index=True)
            o.to_parquet(FIN / f"income_{bi}.parquet", compression="snappy")
            all_dfs = []; bi += 1
        json.dump(list(done), open(sf, "w"))
        pct = len(done) / len(codes) * 100
        elapsed = time.time() - t0
        rate = (i+1) / elapsed if elapsed > 0 else 0
        rem_m = (len(codes) - i - 1) / rate / 60 if rate > 0 else 0
        lg(f"  [{i+1}/{len(codes)}] {len(done)} done ({pct:.1f}%) rate={rate:.0f}/s rem={rem_m:.0f}min")

if all_dfs:
    pd.concat(all_dfs, ignore_index=True).to_parquet(FIN / f"income_{bi}.parquet", compression="snappy")

files = sorted(FIN.glob("income_*.parquet"))
if files:
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df.drop_duplicates(subset=["ts_code","end_date"], keep="last")
    df = df.sort_values(["ts_code","end_date"]).reset_index(drop=True)
    df.to_parquet(FIN / "income.parquet", compression="snappy")
    for f in files: f.unlink()
    lg(f"Merged: {len(df)} rows, {df['ts_code'].nunique()} stocks")

if sf.exists(): sf.unlink()
lg("DONE!")
LOG.close()
