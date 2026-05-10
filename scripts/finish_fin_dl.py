#!/usr/bin/env python3
"""Finish financial data download - resume from done state"""
import subprocess, time, json, os
from pathlib import Path
import pandas as pd

TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
FIN = Path("quant/data/financial_parquet"); FIN.mkdir(exist_ok=True)
TEMP = Path("scripts/fin_temp"); TEMP.mkdir(exist_ok=True)
LOG_FILE = Path("scripts/fin_finish.log")

def lg(m):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {m}", flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {m}\n")

def curl(n, f="", **kw):
    p = json.dumps({"api_name":n,"token":TOKEN,"params":kw,"fields":f})
    r = subprocess.run(["curl","-s","--connect-timeout","5","--max-time","10",
        "-X","POST",URL,"-H","Content-Type: application/json","-d",p],
        capture_output=True, text=True, timeout=15)
    d = json.loads(r.stdout)
    if d["code"] != 0: return []
    if not d.get("data") or not d["data"].get("items"): return []
    return d["data"]["items"]

def do_score(fina, income):
    """Compute fund_score per row"""
    growth = {}
    for code, grp in income.groupby("ts_code"):
        grp = grp.sort_values("end_date")
        prev = {}
        revs = {}
        for _, r in grp.iterrows():
            ed = str(r["end_date"])[:10]
            rev = r.get("total_revenue") or r.get("revenue", 0)
            if pd.isna(rev) or rev == 0:
                continue
            y = int(ed[:4])
            m = ed[5:7]
            pk = (y - 1, m)
            if pk in prev and prev[pk] > 0:
                revs[ed] = (rev - prev[pk]) / prev[pk]
            else:
                revs[ed] = 0.0
            prev[(y, m)] = rev
        if revs:
            growth[code] = revs
    rows = []
    for _, r in fina.iterrows():
        code = r["ts_code"]
        ed = str(r["end_date"])[:10]
        roe = float(r["roe"] if pd.notna(r.get("roe")) else 0) / 100
        gm = float(r["gross_margin"] if pd.notna(r.get("gross_margin")) else 0) / 100
        eps = float(r["eps"] if pd.notna(r.get("eps")) else 0)
        rg = growth.get(code, {}).get(ed, 0.0)
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
        s = max(0, min(100, s))
        rows.append({"ts_code": code, "end_date": ed, "fund_score": s})
    return pd.DataFrame(rows)

lg("Get stock list...")
its = curl("stock_basic", fields="ts_code", exchange="", list_status="L")
codes = [i[0] for i in its]
lg(f"{len(codes)} stocks")

def dl(table, fields):
    sf = TEMP / f"{table}_done.json"
    done = set(json.load(open(sf))) if sf.exists() else set()
    lg(f"Download {table}: {len(done)}/{len(codes)} done")
    all_dfs = []
    bi = 0
    t0 = time.time()
    for i, c in enumerate(codes):
        if c in done:
            continue
        try:
            its = curl(table, fields, ts_code=c, start_date="20050101", end_date="20260430")
            time.sleep(2.0)
            if its:
                df = pd.DataFrame(its, columns=fields.split(","))
                all_dfs.append(df)
                done.add(c)
        except Exception as e:
            done.add(c)
            time.sleep(2)
        if (i + 1) % 100 == 0 or (i + 1) == len(codes):
            if all_dfs:
                part = pd.concat(all_dfs, ignore_index=True)
                part.to_parquet(FIN / f"{table}_{bi}.parquet", compression="snappy")
                all_dfs = []
                bi += 1
            json.dump(list(done), open(sf, "w"))
            pct = len(done) / len(codes) * 100
            lg(f"  [{i+1}/{len(codes)}] {len(done)} done ({pct:.1f}%)")
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_parquet(FIN / f"{table}_{bi}.parquet", compression="snappy")
    if sf.exists():
        sf.unlink()
    files = sorted(FIN.glob(f"{table}_*.parquet"))
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files])
        df = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
        df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
        df.to_parquet(FIN / f"{table}.parquet", compression="snappy")
        for f in files:
            f.unlink()
        lg(f"Merged {table}: {len(df)} rows, {df['ts_code'].nunique()} stocks")

dl("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,bps")
lg("fina_indicator DONE, starting income...")
dl("income", "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps")

fina = pd.read_parquet(FIN / "fina_indicator.parquet") if (FIN / "fina_indicator.parquet").exists() else pd.DataFrame()
income = pd.read_parquet(FIN / "income.parquet") if (FIN / "income.parquet").exists() else pd.DataFrame()
if not fina.empty:
    lg("Computing scores...")
    scores = do_score(fina, income)
    scores.to_parquet("financial_data/scores.parquet", compression="snappy")
    lg(f"scores.parquet: {len(scores)} rows, {scores['ts_code'].nunique()} stocks, mean={scores['fund_score'].mean():.0f}")
lg("ALL DONE!")
