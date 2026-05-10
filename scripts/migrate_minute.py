#!/usr/bin/env python3
"""
⏱ 分钟线数据迁移（逐文件流式写入，低内存）

将 data/min5_csv/ 迁移到 data/minute/freq=5min/year=YYYY/month=MM/
"""

import os, sys, json, time, csv
from pathlib import Path
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
DATA = WORKSPACE / "quant" / "data"
SRC = DATA / "min5_csv"
DST = DATA / "minute"

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

COLS_KEEP = ["stock_code", "trade_date", "datetime", "open", "high", "low", "close", "volume", "amount"]

def migrate_minute():
    log("⏱ 分钟线迁移（流式写入）")
    log(f"  源: {SRC}")
    log(f"  目标: {DST}")
    DST.mkdir(parents=True, exist_ok=True)
    
    files = sorted(SRC.glob("*.csv"))
    if not files:
        log("  ⚠️ 无数据"); return
    log(f"  源文件: {len(files)} 个")
    
    # 扫描：按 (year, month) 分组文件
    monthly_files = {}
    skipped = 0
    log("  扫描...")
    for fpath in files:
        try:
            with open(fpath) as fh:
                next(fh)
                row1 = next(fh).strip()
            # time列格式: 20250102093500000
            parts = row1.split(",")
            ts = parts[1].strip('" ')
            yr, mo = int(ts[:4]), int(ts[4:6])
            monthly_files.setdefault((yr, mo), []).append(fpath)
        except:
            skipped += 1
            continue
    
    if not monthly_files:
        log("  ❌ 无有效文件"); return
    
    log(f"  {len(monthly_files)} 个月份, " +
        f"{min(monthly_files.keys())} ~ {max(monthly_files.keys())}" +
        (f" ({skipped}跳过)" if skipped else ""))
    
    # 逐月流式写入：每个CSV读完后直接写parquet（增量）
    total_written = 0
    for (yr, mo) in sorted(monthly_files.keys()):
        part_dir = DST / "freq=5min" / f"year={yr}" / f"month={mo:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out = part_dir / "data.parquet"
        
        chunks = []
        n_files = 0
        for fpath in monthly_files[(yr, mo)]:
            code = fpath.stem
            try:
                df = pd.read_csv(fpath, usecols=[0,1,2,3,4,5,6,7,8])
                if df.empty:
                    continue
                df = df.rename(columns={"code": "stock_code"})
                # 解析datetime
                df["datetime"] = pd.to_datetime(df["time"].astype(str).str[:14], format="%Y%m%d%H%M%S", errors="coerce")
                df = df.dropna(subset=["datetime"])
                df["trade_date"] = df["datetime"].dt.strftime("%Y-%m-%d")
                df = df[COLS_KEEP]
                chunks.append(df)
                n_files += 1
            except:
                continue
        
        if not chunks:
            continue
        
        combined = pd.concat(chunks, ignore_index=True)
        combined = combined.sort_values(["stock_code", "datetime"]).reset_index(drop=True)
        combined.to_parquet(out, compression="snappy", index=False)
        total_written += len(combined)
        log(f"  ✅ {yr}-{mo:02d}: {len(combined):>8,} 行 ({n_files} 个文件)")
    
    # 元数据
    meta = {
        "columns": COLS_KEEP,
        "partitions": ["freq", "year", "month"],
        "source": "min5_csv",
        "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": total_written,
    }
    json.dump(meta, open(DST / "_meta.json", "w"), ensure_ascii=False, indent=2)
    log(f"\n✅ 分钟线迁移完成，总计 {total_written:,} 行")

if __name__ == "__main__":
    migrate_minute()
