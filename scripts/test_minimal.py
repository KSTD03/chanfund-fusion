#!/usr/bin/env python3
"""极简版 - 只下载前3只就退出"""
import subprocess, time, json, sys
from pathlib import Path

LOG = open("scripts/fin_debug2.log", "w", buffering=1)
def lg(m): LOG.write(f"[{time.strftime('%H:%M:%S')}] {m}\n"); LOG.flush(); print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
PAUSE = 2.0

lg("START")

def curl(n, f="", **kw):
    p = json.dumps({"api_name":n,"token":TOKEN,"params":kw,"fields":f})
    lg(f"  curl payload len={len(p)}")
    r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","15",
        "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
        capture_output=True, text=True, timeout=25)
    lg(f"  curl done, stdout_len={len(r.stdout)}, stderr_len={len(r.stderr)}")
    d = json.loads(r.stdout)
    lg(f"  json parse OK, code={d['code']}")
    if not d.get("data") or not d["data"].get("items"): return []
    return d["data"]["items"]

lg("Getting codes...")
its = curl("stock_basic", fields="ts_code", exchange="", list_status="L")
codes = [i[0] for i in its]
lg(f"{len(codes)} codes: {codes[0]}..{codes[-1]}")

lg("Getting fina_indicator for first 3...")
for i, c in enumerate(codes[:3]):
    lg(f"[{i}] {c} starting...")
    t1 = time.time()
    try:
        its = curl("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,bps",
                   ts_code=c, start_date="20050101", end_date="20260430")
        lg(f"  -> {len(its)} rows ({time.time()-t1:.1f}s)")
    except Exception as e:
        lg(f"  -> ERROR: {type(e).__name__}: {e} ({time.time()-t1:.1f}s)")
    time.sleep(PAUSE)

lg("DONE")
LOG.close()
