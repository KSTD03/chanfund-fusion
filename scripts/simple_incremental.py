#!/usr/bin/env python3
"""极简增量下载 — 每次跑100只, 断点续传"""
import subprocess, time, json
from pathlib import Path

TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
FIN = Path("quant/data/financial_parquet")
FIN.mkdir(exist_ok=True)

def log(m):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {m}", flush=True)

def curl(n, f="", **kw):
    p = json.dumps({"api_name":n,"token":TOKEN,"params":kw,"fields":f})
    r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","10",
        "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
        capture_output=True, text=True, timeout=15)
    d = json.loads(r.stdout)
    if d["code"] != 0: raise Exception(d.get("msg",""))
    return d["data"]["items"] if d.get("data") and d["data"].get("items") else []

log("启动...")

# Get all codes (cache in file)
cache = Path("scripts/fin_temp/all_codes.json")
if cache.exists():
    codes = json.load(open(cache))
else:
    cache.parent.mkdir(exist_ok=True)
    codes = [i[0] for i in curl("stock_basic", fields="ts_code", exchange="", list_status="L")]
    json.dump(codes, open(cache, "w"))
log(f"{len(codes)} stocks")

# Load progress from per-stock files
done = set()
for f in FIN.glob("fina_indicator_*.parquet"):
    code = f.name.replace("fina_indicator_", "").replace(".parquet", "")
    done.add(code)
log(f"已有: {len(done)}/{len(codes)}")

# Get next 100 pending
pending = [c for c in codes if c not in done][:100]
if not pending:
    log("全部完成!")
    exit(0)

log(f"下载: {pending[0]}..{pending[-1]} ({len(pending)}只)")

t0 = time.time()
for i, code in enumerate(pending):
    try:
        items = curl("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,bps",
                     ts_code=code, start_date="20050101", end_date="20260430")
        if items:
            import pandas as pd
            df = pd.DataFrame(items, columns=["ts_code","ann_date","end_date","eps","roe","gross_margin","bps"])
            df.to_parquet(FIN / f"fina_indicator_{code}.parquet", compression="snappy", index=False)
        done.add(code)
        time.sleep(4.0)
    except Exception as e:
        done.add(code)  # skip failed
        log(f"  ⚠️ {code}: {str(e)[:60]}")
        time.sleep(4.0)
    if (i+1) % 50 == 0:
        log(f"  {i+1}/{len(pending)} ({time.time()-t0:.0f}s)")

# Same for income
pending_inc = [c for c in codes if c not in {f.name.replace("income_","").replace(".parquet","") for f in FIN.glob("income_*.parquet")}][:100]
if pending_inc:
    log(f"下载income: {pending_inc[0]}..{pending_inc[-1]} ({len(pending_inc)}只)")
    for i, code in enumerate(pending_inc):
        try:
            items = curl("income", "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps",
                         ts_code=code, start_date="20050101", end_date="20260430")
            if items:
                import pandas as pd
                df = pd.DataFrame(items, columns=["ts_code","ann_date","end_date","total_revenue","revenue","basic_eps"])
                df.to_parquet(FIN / f"income_{code}.parquet", compression="snappy", index=False)
            time.sleep(4.0)
        except:
            time.sleep(4.0)
        if (i+1) % 50 == 0:
            log(f"  income {i+1}/{len(pending_inc)}")

log(f"完成! ({time.time()-t0:.0f}s)")
