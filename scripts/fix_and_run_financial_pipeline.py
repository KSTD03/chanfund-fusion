#!/usr/bin/env python3
"""
Step 1: 财务数据补全管线 (SDK版)
=================================
从Tushare下载 fina_indicator + income 全量数据 (2005-至今)
使用Python SDK + 代理配置，输出Parquet + scores.parquet

用法:
    python3 scripts/fix_and_run_financial_pipeline.py              # 全量下载
    python3 scripts/fix_and_run_financial_pipeline.py --quick      # 快速验证(500只)
    python3 scripts/fix_and_run_financial_pipeline.py --resume     # 断点续传
"""

import os, sys, time, json, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ============================================================
# Tushare 代理配置
# ============================================================
import tushare as ts
from tushare.pro import client as _ts_client
_ts_client.DataApi._DataApi__http_url = "http://47.109.59.144:8989/dataapi"
TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
ts.set_token(TOKEN)
PRO = ts.pro_api()

# ============================================================
# 路径
# ============================================================
WS = Path(__file__).resolve().parent.parent
FIN_PQ = WS / "quant" / "data" / "financial_parquet"
SCORES_DIR = WS / "financial_data"
TEMP = WS / "scripts" / "fin_temp"
os.makedirs(FIN_PQ, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(TEMP, exist_ok=True)

SLEEP = 1.5  # ~40次/min，保守

t0 = time.time()


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
def warn(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ {m}", flush=True)


def get_stocks() -> list:
    """获取全A股代码列表"""
    cache = TEMP / "codes.json"
    if cache.exists():
        return json.load(open(cache))
    log("📋 拉取股票列表...")
    df = pd.concat([
        PRO.stock_basic(exchange="", list_status="L", fields="ts_code"),
        PRO.stock_basic(exchange="", list_status="D", fields="ts_code"),
    ]).drop_duplicates("ts_code")
    codes = df["ts_code"].tolist()
    json.dump(codes, open(cache, "w"))
    log(f"  共 {len(codes)} 只")
    return codes


def download_table(table: str, codes: list, fields: str, resume: bool = False):
    """批量下载一张财务表"""
    total = len(codes)
    save_every = 100
    done_file = TEMP / f"{table}_done.txt"
    idx_file = TEMP / f"{table}_idx.txt"

    done = set()
    if resume and done_file.exists():
        done = set(open(done_file).read().strip().split("\n"))
        done.discard("")
        log(f"🔄 {table} 续传: 已有 {len(done)}/{total} 只")

    start = 0
    if resume and idx_file.exists():
        start = int(open(idx_file).read().strip())

    all_dfs = []
    for i, code in enumerate(codes):
        if i < start:
            continue
        if code in done:
            continue

        try:
            df = PRO.query(table, ts_code=code, start_date="20050101",
                          end_date="20260430", fields=fields)
            time.sleep(SLEEP)
        except Exception as e:
            warn(f"  [{i}/{total}] {code}: {e}")
            time.sleep(5)
            continue

        if df is not None and len(df) > 0:
            all_dfs.append(df)
            done.add(code)

        if (i + 1) % save_every == 0:
            log(f"  {table} [{i+1}/{total}] ({(i+1)/total*100:.0f}%)")
            open(idx_file, "w").write(str(i + 1))
            open(done_file, "w").write("\n".join(sorted(done)))
            save_batch(table, all_dfs, idx=i // save_every)
            all_dfs = []

        # 耗时估算
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            log(f"  📊 速率: {rate:.1f}只/秒, 剩余: {remaining/60:.0f}分钟")

    # 保存余量
    if all_dfs:
        save_batch(table, all_dfs, idx=999)

    # 清理
    for f in [done_file, idx_file]:
        if f.exists(): f.unlink()

    log(f"✅ {table} 下载完成: {total} 只, 耗时 {(time.time()-t0)/60:.1f}分钟")


def save_batch(table: str, dfs: list, idx: int):
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    f = FIN_PQ / f"{table}_batch_{idx:04d}.parquet"
    df.to_parquet(f, compression="snappy", index=False)


def concat_table(table: str) -> pd.DataFrame:
    """合并所有碎片"""
    parts = []
    for f in sorted(FIN_PQ.glob(f"{table}_batch_*.parquet")):
        parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("ann_date", na_position="last") \
           .drop_duplicates(subset=["ts_code", "end_date"], keep="last") \
           .sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    df.to_parquet(FIN_PQ / f"{table}.parquet", compression="snappy", index=False)
    for f in FIN_PQ.glob(f"{table}_batch_*.parquet"):
        f.unlink()
    log(f"✅ {table} 合并: {len(df)} 行, {df['ts_code'].nunique()} 只, "
        f"{df['end_date'].min()}~{df['end_date'].max()}")
    return df


def calc_growth(income_df: pd.DataFrame) -> dict:
    """YoY营收增长率 {code: {end_date: rate}}"""
    if income_df.empty:
        return {}
    df = income_df.copy()
    result = {}
    for code, grp in df.groupby("ts_code"):
        grp = grp.sort_values("end_date")
        prev = {}
        revs = {}
        for _, row in grp.iterrows():
            ed = row["end_date"]
            rev = row.get("total_revenue") or row.get("revenue", 0)
            if pd.isna(rev) or rev == 0:
                continue
            m = ed[5:7]
            y = int(ed[:4])
            k = (y, m)
            pk = (y - 1, m)
            if pk in prev and prev[pk] > 0:
                revs[ed] = (rev - prev[pk]) / prev[pk]
            else:
                revs[ed] = 0.0
            prev[k] = rev
        if revs:
            result[code] = revs
    return result


def gen_scores(fina: pd.DataFrame, income: pd.DataFrame):
    """生成 scores.parquet"""
    log("📊 生成评分...")
    if fina.empty:
        log("⚠️ fina_indicator 无数据")
        return

    growth = calc_growth(income)

    rows = []
    for _, r in fina.iterrows():
        code = r["ts_code"]
        ed = str(r["end_date"])[:10]
        ann = str(r.get("ann_date", ed))[:10]
        roe = float(r.get("roe", 0) or 0) / 100
        gm = float(r.get("gross_margin", 0) or 0) / 100
        eps = float(r.get("eps", 0) or 0)
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

        rows.append({
            "ts_code": code, "end_date": ed, "ann_date": ann,
            "roe": roe, "gross_margin": gm, "eps": eps,
            "revenue_growth": rg, "fund_score": max(0, min(100, s)),
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ts_code", "end_date"])
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    df.to_parquet(SCORES_DIR / "scores.parquet", compression="snappy", index=False)
    log(f"✅ scores.parquet: {len(df)} 行, {df['ts_code'].nunique()} 只")
    log(f"   评分分布: mean={df['fund_score'].mean():.1f}, "
        f"min={df['fund_score'].min():.0f}, max={df['fund_score'].max():.0f}")
    return df


def quick():
    log("=" * 55)
    log("⚡ 快速验证 (500只)")
    log("=" * 55)
    codes = get_stocks()[:500]
    log(f"📋 500 只")

    download_table("fina_indicator", codes, "ts_code,ann_date,end_date,eps,roe,gross_margin,ocf_per_share,bps")
    fina = concat_table("fina_indicator")

    download_table("income", codes, "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps")
    income = concat_table("income")

    gen_scores(fina, income)
    log(f"✅ 完成! {(time.time()-t0)/60:.1f}min")


def full(resume=False):
    log("=" * 55)
    log("🔰 全量财务数据补全")
    log("=" * 55)
    codes = get_stocks()
    log(f"📋 {len(codes)} 只")

    est = len(codes) * SLEEP * 2 / 60 / 60  # 2 tables
    log(f"⏱️ 预估: {est:.1f} 小时")

    download_table("fina_indicator", codes,
                   "ts_code,ann_date,end_date,eps,roe,gross_margin,ocf_per_share,bps",
                   resume=resume)
    fina = concat_table("fina_indicator")

    download_table("income", codes,
                   "ts_code,ann_date,end_date,total_revenue,revenue,basic_eps",
                   resume=resume)
    income = concat_table("income")

    gen_scores(fina, income)

    cov_f = len(set(fina["ts_code"])) / len(codes) * 100 if not fina.empty else 0
    cov_i = len(set(income["ts_code"])) / len(codes) * 100 if not income.empty else 0
    log(f"\n覆盖率: fina={cov_f:.1f}%  income={cov_i:.1f}%")
    log(f"✅ 完成! {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.quick: quick()
    else: full(resume=a.resume)
