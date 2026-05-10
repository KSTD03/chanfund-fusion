#!/usr/bin/env python3
"""财务下载 — 原始HTTP版，30s超时保护"""
import socket, time, sys, json, requests
socket.setdefaulttimeout(10)
from pathlib import Path
import pandas as pd

URL = "http://47.109.59.144:8989/dataapi"
TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
FIN = Path("quant/data/financial_parquet")
SCORES = Path("financial_data")
FIN.mkdir(parents=True, exist_ok=True); SCORES.mkdir(parents=True, exist_ok=True)

def api(api_name, fields="", **params):
    r = requests.post(URL, json={"api_name": api_name, "token": TOKEN,
        "params": params, "fields": fields}, timeout=(5, 20))
    d = r.json()
    if d["code"] != 0: raise Exception(d.get("msg","API error"))
    return pd.DataFrame(d["data"]["items"], columns=d["data"]["fields"]) if d.get("data") else pd.DataFrame()

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# Clean old batch files
for f in FIN.glob("batch_*.parquet"): f.unlink()
SLEEP, BATCH = 2.0, 50

codes = api("stock_basic", fields="ts_code", exchange="", list_status="L")["ts_code"].tolist()[:500]
log(f"500只: {codes[0]}..{codes[-1]}")

def dl(table, fields):
    log(f"📥 {table}...")
    t0, state_file = time.time(), FIN / f"state_{table}.json"
    start = 0
    if state_file.exists():
        start = json.load(open(state_file))["idx"]
        log(f"  续传: {start}/{len(codes)}")
    for i in range(start, len(codes), BATCH):
        b_codes = codes[i:i+BATCH]
        b_dfs = []
        for j, code in enumerate(b_codes):
            try:
                t1 = time.time()
                df = api(table, fields, ts_code=code, start_date="20050101", end_date="20260430")
                elapsed = time.time() - t1
                if elapsed > 20: time.sleep(10); continue
                time.sleep(SLEEP)
                if not df.empty: b_dfs.append(df)
            except Exception as e:
                log(f"  ⚠️ [{i+j}] {code}: {type(e).__name__}: {str(e)[:50]}")
                time.sleep(10)
        if b_dfs:
            pd.concat(b_dfs, ignore_index=True).to_parquet(
                FIN / f"batch_{table}_{i//BATCH:04d}.parquet", compression="snappy", index=False)
            log(f"  ✅ [{i+BATCH}/{len(codes)}] ({time.time()-t0:.0f}s)")
        json.dump({"idx": i + BATCH}, open(state_file, "w"))
    if state_file.exists(): state_file.unlink()
    log(f"  ✅ ({time.time()-t0:.0f}s)")

dl("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,ocf_per_share,bps")
fina = pd.concat([pd.read_parquet(f) for f in sorted(FIN.glob("batch_fina_indicator_*.parquet"))]) if list(FIN.glob("batch_fina_indicator_*.parquet")) else pd.DataFrame()
fina = fina.drop_duplicates(subset=["ts_code","end_date"], keep="last")
log(f"📊 fina: {len(fina)}行 {fina['ts_code'].nunique()}只" if not fina.empty else "⚠️ fina: 空")

dl("income", "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps")
income = pd.concat([pd.read_parquet(f) for f in sorted(FIN.glob("batch_income_*.parquet"))]) if list(FIN.glob("batch_income_*.parquet")) else pd.DataFrame()
income = income.drop_duplicates(subset=["ts_code","end_date"], keep="last")
log(f"📊 income: {len(income)}行 {income['ts_code'].nunique()}只" if not income.empty else "⚠️ income: 空")

if not fina.empty:
    log("\n📊 评分...")
    growth = {}
    for code, grp in income.groupby("ts_code"):
        grp = grp.sort_values("end_date"); prev = {}; revs = {}
        for _, r in grp.iterrows():
            ed = str(r["end_date"])[:10]
            rev = r.get("total_revenue") or r.get("revenue", 0)
            if pd.isna(rev) or rev == 0: continue
            y, m = int(ed[:4]), ed[5:7]
            pk = (y-1, m)
            revs[ed] = (rev - prev[pk]) / prev[pk] if pk in prev and prev[pk] > 0 else 0.0
            prev[(y,m)] = rev
        if revs: growth[code] = revs

    rows = []
    for _, r in fina.iterrows():
        code, ed = r["ts_code"], str(r["end_date"])[:10]
        roe = float(r.get("roe",0) or 0)/100; gm = float(r.get("gross_margin",0) or 0)/100
        eps = float(r.get("eps",0) or 0); rg = growth.get(code,{}).get(ed,0.0)
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
        rows.append({"ts_code": code, "end_date": ed, "fund_score": max(0, min(100, s))})

    scores = pd.DataFrame(rows)
    scores.to_parquet(SCORES / "scores.parquet", compression="snappy", index=False)
    log(f"✅ scores.parquet: {len(scores)}行 {scores['ts_code'].nunique()}只")
    for f in FIN.glob("batch_*.parquet"): f.unlink()
log("🎉 完成!")
