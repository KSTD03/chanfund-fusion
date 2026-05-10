#!/usr/bin/env python3
"""Tushare全量下载 — 慢速稳定版"""
import time, sys, json, requests
from pathlib import Path
import pandas as pd

URL = "http://47.109.59.144:8989/dataapi"
TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
FIN = Path("quant/data/financial_parquet"); FIN.mkdir(parents=True, exist_ok=True)
TEMP = Path("scripts/fin_temp"); TEMP.mkdir(parents=True, exist_ok=True)

def api(n, f="", **kw):
    for a in range(3):
        try:
            r = requests.post(URL, json={"api_name":n,"token":TOKEN,"params":kw,"fields":f},
                              timeout=(10,30), headers={"Connection":"close"})
            d=r.json()
            if d["code"]!=0: raise Exception(d.get("msg",""))
            if not d.get("data") or not d["data"].get("items"): return pd.DataFrame()
            return pd.DataFrame(d["data"]["items"], columns=d["data"]["fields"])
        except Exception as e:
            if a<2: time.sleep(5*(a+1))
            else: raise

sleep_time = float(sys.argv[1]) if len(sys.argv)>1 else 3.0

log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
log(f"休眠={sleep_time}s/只")

# 只下载上市股票（5512只），不退市
codes = api("stock_basic", fields="ts_code", exchange="", list_status="L")["ts_code"].tolist()
log(f"📋 {len(codes)}只")

def dl(t,f):
    sf=TEMP/f"{t}_done.json"
    done=set(json.load(open(sf))) if sf.exists() else set()
    log(f"📥 {t}: {len(done)}/{len(codes)}已有")
    all_dfs,bi,t0=[],0,time.time()
    for i,c in enumerate(codes):
        if c in done: continue
        try:
            df=api(t,f,ts_code=c,start_date="20050101",end_date="20260430")
            time.sleep(sleep_time)
            if not df.empty: all_dfs.append(df); done.add(c)
        except Exception as e:
            log(f"⚠️[{i}]{c}:{str(e)[:60]}")
            time.sleep(15)
        if (i+1)%100==0:
            e=time.time()-t0; r=(i+1)/e
            log(f"  [{i+1}/{len(codes)}] {r:.1f}只/s 剩余{(len(codes)-i-1)/r/60:.0f}min")
            if all_dfs:
                pd.concat(all_dfs,ignore_index=True).to_parquet(FIN/f"{t}_{bi}.parquet", compression="snappy")
                all_dfs,bi=[],bi+1
            json.dump(list(done),open(sf,"w"))
    if all_dfs: pd.concat(all_dfs,ignore_index=True).to_parquet(FIN/f"{t}_{bi}.parquet", compression="snappy")
    if sf.exists(): sf.unlink()
    files=sorted(FIN.glob(f"{t}_*.parquet"))
    if files:
        df=pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(subset=["ts_code","end_date"],keep="last").sort_values(["ts_code","end_date"]).reset_index(drop=True)
        df.to_parquet(FIN/f"{t}.parquet",compression="snappy"); [f.unlink() for f in files]
        log(f"✅ {t}: {len(df)}行 {df['ts_code'].nunique()}只")

dl("fina_indicator","ts_code,ann_date,end_date,eps,roe,gross_margin,ocf_per_share,bps")
dl("income","ts_code,ann_date,end_date,total_revenue,revenue,basic_eps")

# Score
fina=pd.read_parquet(FIN/"fina_indicator.parquet") if (FIN/"fina_indicator.parquet").exists() else pd.DataFrame()
income=pd.read_parquet(FIN/"income.parquet") if (FIN/"income.parquet").exists() else pd.DataFrame()
if not fina.empty:
    log("📊评分...")
    g={}
    for c,gr in income.groupby("ts_code"):
        gr=gr.sort_values("end_date"); p={}; rs={}
        for _,r in gr.iterrows():
            ed=str(r["end_date"])[:10]; rev=r.get("total_revenue") or r.get("revenue",0)
            if pd.isna(rev) or rev==0: continue
            y,m=int(ed[:4]),ed[5:7]; pk=(y-1,m)
            rs[ed]=(rev-p[pk])/p[pk] if pk in p and p[pk]>0 else 0.0
            p[(y,m)]=rev
        if rs: g[c]=rs
    def sc(roe,gm,rg,eps):
        s=0
        if roe>=0.20: s+=30
        elif roe>=0.10: s+=20
        elif roe>=0.05: s+=10
        if gm>=0.40: s+=25
        elif gm>=0.25: s+=15
        elif gm>=0.15: s+=10
        if rg>=0.20: s+=25
        elif rg>=0.10: s+=20
        elif rg>=0: s+=15
        elif rg>=-0.20: s+=5
        else: s-=10
        if eps>0: s+=20
        elif eps>-0.5: s+=5
        else: s-=5
        return max(0,min(100,s))
    rows=[{"ts_code":r["ts_code"],"end_date":str(r["end_date"])[:10],
        "fund_score":sc(float(r.get("roe",0)or 0)/100,float(r.get("gross_margin",0)or 0)/100,
            g.get(r["ts_code"],{}).get(str(r["end_date"])[:10],0),float(r.get("eps",0)or 0))
    } for _,r in fina.iterrows()]
    pd.DataFrame(rows).to_parquet("financial_data/scores.parquet",compression="snappy")
    log(f"✅ scores.parquet: {len(rows)}行")

log("🎉完成!")
