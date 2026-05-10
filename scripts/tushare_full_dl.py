#!/usr/bin/env python3
"""Tushare全量财务数据下载 — 原始HTTP + 重试 + 续传"""
import time, sys, json, requests
from pathlib import Path
import pandas as pd
import numpy as np

URL = "http://47.109.59.144:8989/dataapi"
TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"

FIN = Path("quant/data/financial_parquet")
TEMP = Path("scripts/fin_temp")
FIN.mkdir(parents=True, exist_ok=True); TEMP.mkdir(parents=True, exist_ok=True)

REST_SEC = 2.0

def api(api_name, fields="", **params):
    retries = 3
    for attempt in range(retries):
        try:
            r = requests.post(URL, json={
                "api_name": api_name, "token": TOKEN,
                "params": params, "fields": fields
            }, timeout=(10, 30))
            d = r.json()
            if d["code"] != 0:
                raise Exception(d.get("msg", f"code={d['code']}"))
            if not d.get("data") or not d["data"].get("items"):
                return pd.DataFrame()
            return pd.DataFrame(d["data"]["items"], columns=d["data"]["fields"])
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise

def log(m):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {m}", flush=True)

codes = api("stock_basic", fields="ts_code", exchange="", list_status="L")
codes = codes["ts_code"].tolist()
log(f"📋 {len(codes)}只上市股票")

# 还下载已退市的
try:
    codes_d = api("stock_basic", fields="ts_code", exchange="", list_status="D")
    codes_d = codes_d["ts_code"].tolist()
    codes = list(set(codes) | set(codes_d))
    log(f"  含 {len(codes_d)}只退市, 共 {len(codes)}只")
except:
    pass

def dl(table, fields):
    log(f"\n📥 {table}...")
    state_file = TEMP / f"{table}_done.json"
    done = set()
    if state_file.exists():
        done = set(json.load(open(state_file)))
        log(f"  已有 {len(done)}/{len(codes)}, 续传")
    
    t0 = time.time()
    all_dfs, batch_idx = [], 0
    for i, code in enumerate(codes):
        if code in done:
            continue
        try:
            df = api(table, fields, ts_code=code, start_date="20050101", end_date="20260430")
            time.sleep(REST_SEC)
            if not df.empty:
                all_dfs.append(df)
                done.add(code)
        except Exception as e:
            log(f"  ⚠️ [{i}] {code}: {str(e)[:80]}")
            time.sleep(10)
            continue
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            rem = (len(codes) - i - 1) / rate if rate > 0 else 0
            log(f"  [{i+1}/{len(codes)}] ({rate:.0f}只/s, 剩余{rem/60:.0f}min)")
            if all_dfs:
                pd.concat(all_dfs, ignore_index=True).to_parquet(
                    FIN / f"{table}_{batch_idx}.parquet", compression="snappy", index=False)
                all_dfs = []; batch_idx += 1
            json.dump(list(done), open(state_file, "w"))
    
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_parquet(
            FIN / f"{table}_{batch_idx}.parquet", compression="snappy", index=False)
    if state_file.exists(): state_file.unlink()
    log(f"✅ {table} 完成 ({time.time()-t0:.0f}s)")

def merge_(table):
    files = sorted(FIN.glob(f"{table}_*.parquet"))
    if not files: return pd.DataFrame()
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    df.to_parquet(FIN / f"{table}.parquet", compression="snappy", index=False)
    for f in files: f.unlink()
    log(f"📊 {table}: {len(df)}行 {df['ts_code'].nunique()}只 {df['end_date'].min()}~{df['end_date'].max()}")
    return df

dl("fina_indicator", "ts_code,ann_date,end_date,eps,roe,gross_margin,ocf_per_share,bps")
fina = merge_("fina_indicator")
dl("income", "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps")
income = merge_("income")

if not fina.empty:
    log("\n📊 生成评分...")
    growth = {}
    for code, grp in income.groupby("ts_code"):
        grp = grp.sort_values("end_date"); prev = {}; revs = {}
        for _, r in grp.iterrows():
            ed = str(r["end_date"])[:10]; rev = r.get("total_revenue") or r.get("revenue", 0)
            if pd.isna(rev) or rev == 0: continue
            y, m = int(ed[:4]), ed[5:7]; pk = (y-1, m)
            revs[ed] = (rev - prev[pk]) / prev[pk] if pk in prev and prev[pk] > 0 else 0.0
            prev[(y,m)] = rev
        if revs: growth[code] = revs

    def sc(roe, gm, rg, eps):
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

    rows = [{"ts_code": r["ts_code"], "end_date": str(r["end_date"])[:10],
        "fund_score": sc(float(r.get("roe",0)or 0)/100, float(r.get("gross_margin",0)or 0)/100,
            growth.get(r["ts_code"],{}).get(str(r["end_date"])[:10],0.0), float(r.get("eps",0)or 0))
    } for _, r in fina.iterrows()]

    scores = pd.DataFrame(rows)
    scores.to_parquet("financial_data/scores.parquet", compression="snappy", index=False)
    log(f"✅ scores.parquet: {len(scores)}行 {scores['ts_code'].nunique()}只")
    log(f"   mean={scores['fund_score'].mean():.1f} min={scores['fund_score'].min():.0f} max={scores['fund_score'].max():.0f}")

log("\n🎉 全部完成!")
