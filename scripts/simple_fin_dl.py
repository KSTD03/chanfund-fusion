#!/usr/bin/env python3
"""极简财务数据下载器 — 500只快速验证"""
import time, sys, json
from pathlib import Path
import pandas as pd
from tushare.pro import client as _C
_C.DataApi._DataApi__http_url = "http://47.109.59.144:8989/dataapi"
import tushare as ts
ts.set_token("A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY")
pro = ts.pro_api()

FIN = Path("quant/data/financial_parquet")
SCORES = Path("financial_data")
FIN.mkdir(parents=True, exist_ok=True)
SCORES.mkdir(parents=True, exist_ok=True)
SLEEP = 1.5

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("获取股票列表...")
dfb = pro.stock_basic(exchange="", list_status="L", fields="ts_code")
codes = dfb["ts_code"].tolist()[:500]
log(f"500只: {codes[0]} ... {codes[-1]}")

def dl(table, fields):
    log(f"下载 {table}...")
    all_dfs, t0 = [], time.time()
    for i, code in enumerate(codes):
        try:
            df = pro.query(table, ts_code=code, start_date="20050101", end_date="20260430", fields=fields)
            time.sleep(SLEEP)
            if df is not None and len(df) > 0: all_dfs.append(df)
        except Exception as e:
            log(f"  [{i}] {code}: {e}")
            time.sleep(5)
        if (i+1) % 100 == 0:
            log(f"  [{i+1}/500] ({(i+1)/500*100:.0f}%)")
            batch = pd.concat(all_dfs, ignore_index=True)
            batch.to_parquet(FIN / f"{table}_{i//100}.parquet", compression="snappy", index=False)
            log(f"    -> {len(batch)} 行")
            all_dfs = []
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_parquet(FIN / f"{table}_final.parquet", compression="snappy")

dl("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,ocf_per_share,bps")
log(f"fina_indicator 完成! {(time.time()-t0)/60:.1f}min")

dl("income", "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps")
log(f"income 完成! {(time.time()-t0)/60:.1f}min")

log("生成评分...")
fin = sorted(FIN.glob("fina_indicator*.parquet"))
inc = sorted(FIN.glob("income*.parquet"))
fina = pd.concat([pd.read_parquet(f) for f in fin]).drop_duplicates(subset=["ts_code","end_date"], keep="last") if fin else pd.DataFrame()
income = pd.concat([pd.read_parquet(f) for f in inc]).drop_duplicates(subset=["ts_code","end_date"], keep="last") if inc else pd.DataFrame()
log(f"  fina: {len(fina)}行 {fina['ts_code'].nunique()}只" if not fina.empty else "  fina: 空")
log(f"  income: {len(income)}行 {income['ts_code'].nunique()}只" if not income.empty else "  income: 空")

growth = {}
for code, grp in income.groupby("ts_code"):
    grp = grp.sort_values("end_date"); prev = {}; revs = {}
    for _, r in grp.iterrows():
        ed, rev = r["end_date"], r.get("total_revenue") or r.get("revenue", 0)
        if pd.isna(rev) or rev == 0: continue
        pk = (int(ed[:4])-1, ed[5:7])
        revs[ed] = (rev - prev[pk]) / prev[pk] if pk in prev and prev[pk] > 0 else 0.0
        prev[(int(ed[:4]), ed[5:7])] = rev
    if revs: growth[code] = revs

def score(roe, gm, rg, eps):
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
    return max(0, min(100, s))

rows = []
for _, r in fina.iterrows():
    ed, code = str(r["end_date"])[:10], r["ts_code"]
    rows.append({"ts_code": code, "end_date": ed, "fund_score": score(
        float(r.get("roe",0) or 0)/100, float(r.get("gross_margin",0) or 0)/100,
        growth.get(code,{}).get(ed,0.0), float(r.get("eps",0) or 0)
    )})

pd.DataFrame(rows).to_parquet(SCORES / "scores.parquet", compression="snappy", index=False)
log(f"✅ scores.parquet: {len(rows)}行")
for f in FIN.glob("fina_indicator_*.parquet"): f.unlink()
for f in FIN.glob("income_*.parquet"): f.unlink()
log("✅ 完成!")
