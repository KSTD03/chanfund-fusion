#!/usr/bin/env python3
"""
📐 数据目录规范化迁移脚本

目标结构：
  quant/data/
  ├── daily/year=YYYY/    ← 日线 parquet，按年分区
  ├── minute/freq=5min/year=YYYY/month=MM/  ← 分钟线
  ├── financial/report_type=XXX/year=YYYY/   ← 财务数据
  ├── fundamental/year=YYYY/     ← 每日估值指标
  ├── macro/index=XXX/year=YYYY/  ← 宏观/指数数据
  └── meta/                      ← 配置、断点续传等

用法：
  python3 scripts/migrate_data_structure.py         # 执行迁移
  python3 scripts/migrate_data_structure.py --dry   # 只预览不写
  python3 scripts/migrate_data_structure.py --cleanup  # 迁移完成后清理旧目录
"""

import os, sys, json, shutil, time
from pathlib import Path
import pandas as pd
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
DATA = WORKSPACE / "quant" / "data"
DRY_RUN = "--dry" in sys.argv
CLEANUP = "--cleanup" in sys.argv

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def ensure_dir(p):
    if not p.exists():
        if DRY_RUN:
            log(f"  [DRY] mkdir {p}")
        else:
            p.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 📈 日线：data/daily/year=YYYY/
#    数据源：data/daily_parquet/ (每股票一个文件, 6列: date,open,high,low,close,volume)
# ============================================================
def migrate_daily():
    log("\n" + "="*60)
    log("1/5 📈 迁移日线 → daily/")
    
    src = DATA / "daily_parquet"
    dst_base = DATA / "daily"
    files = sorted(src.glob("*.parquet"))
    log(f"  源文件: {len(files)} 个")
    
    if not files:
        log("  ⚠️ 无数据，跳过")
        return
    
    # 按年收集数据
    yearly = {}
    total_rows = 0
    for fpath in files:
        code = fpath.stem  # e.g. sh.600000
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            log(f"  ⚠️ 读取失败 {fpath.name}: {e}")
            continue
        if df.empty:
            continue
        df = df.copy()
        df["stock_code"] = code
        # 标准化列名
        df = df.rename(columns={"date": "trade_date"})
        # 确保 trade_date 是 date 类型
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["year"] = df["trade_date"].dt.year
        
        total_rows += len(df)
        for yr, grp in df.groupby("year"):
            # 删除 year 列（分区键不存）
            grp = grp.drop(columns=["year"])
            yearly.setdefault(int(yr), []).append(grp)
    
    log(f"  总计 {total_rows:,} 行, 覆盖 {len(yearly)} 个年份")
    
    for yr in sorted(yearly.keys()):
        part_dir = dst_base / f"year={yr}"
        ensure_dir(part_dir)
        combined = pd.concat(yearly[yr], ignore_index=True)
        combined = combined.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        out = part_dir / "data.parquet"
        
        if DRY_RUN:
            log(f"  [DRY] {yr}: {len(combined):>8,} 行 → {out}")
        else:
            combined.to_parquet(out, compression="snappy", index=False)
            log(f"  ✅ {yr}: {len(combined):>8,} 行 → {out}")
    
    # 写分区元数据
    if not DRY_RUN:
        meta = {
            "columns": ["stock_code", "trade_date", "open", "high", "low", "close", "volume"],
            "partitions": ["year"],
            "source": "daily_parquet (10jqka)",
            "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_rows": int(total_rows),
        }
        json.dump(meta, open(dst_base / "_meta.json", "w"), ensure_ascii=False, indent=2)
        log(f"  📝 元数据已写")


# ============================================================
# 2. 📊 分钟线：data/minute/freq=5min/year=YYYY/month=MM/
#    数据源：data/min5_csv/ (每股票一个文件)
# ============================================================
def migrate_minute():
    log("\n" + "="*60)
    log("2/5 📊 迁移分钟线 → minute/")
    
    src = DATA / "min5_csv"
    dst_base = DATA / "minute"
    files = sorted(src.glob("*.csv"))
    log(f"  源文件: {len(files)} 个")
    
    if not files:
        log("  ⚠️ 无数据，跳过")
        return
    
    # 按年-月-股票汇总
    monthly = {}
    total_rows = 0
    skipped = 0
    
    for fpath in files:
        code = fpath.stem  # e.g. sh.600000
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            skipped += 1
            continue
        if df.empty:
            skipped += 1
            continue
        
        # 标准化
        df = df.rename(columns={
            "date": "trade_date",
            "time": "datetime_str",
            "code": "stock_code",
        })
        if "stock_code" not in df.columns:
            df["stock_code"] = code
        
        # 解析时间
        # datetime_str 格式: 20250102093500000
        df["datetime"] = pd.to_datetime(df["datetime_str"].astype(str).str[:14], format="%Y%m%d%H%M%S", errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        
        total_rows += len(df)
        
        for (yr, mo), grp in df.groupby(["year", "month"]):
            grp = grp.drop(columns=["year", "month"])
            key = (int(yr), int(mo))
            monthly.setdefault(key, []).append(grp)
    
    log(f"  总计 {total_rows:,} 行, 覆盖 {len(monthly)} 个月份" +
        (f" ({skipped} 跳过)" if skipped else ""))
    
    for (yr, mo) in sorted(monthly.keys()):
        part_dir = dst_base / "freq=5min" / f"year={yr}" / f"month={mo:02d}"
        ensure_dir(part_dir)
        combined = pd.concat(monthly[(yr, mo)], ignore_index=True)
        combined = combined.sort_values(["stock_code", "datetime"]).reset_index(drop=True)
        
        # 选标准列
        keep_cols = [c for c in ["stock_code", "trade_date", "datetime", "open", "high", "low", "close", "volume", "amount"] if c in combined.columns]
        combined = combined[keep_cols]
        out = part_dir / "data.parquet"
        
        if DRY_RUN:
            log(f"  [DRY] {yr}-{mo:02d}: {len(combined):>8,} 行 → {out}")
        else:
            combined.to_parquet(out, compression="snappy", index=False)
    
    if not DRY_RUN:
        meta = {
            "columns": ["stock_code", "trade_date", "datetime", "open", "high", "low", "close", "volume", "amount"],
            "partitions": ["freq", "year", "month"],
            "source": "min5_csv",
            "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_rows": int(total_rows),
        }
        json.dump(meta, open(dst_base / "_meta.json", "w"), ensure_ascii=False, indent=2)
        log(f"  📝 元数据已写")


# ============================================================
# 3. 🏦 财务数据：data/financial/report_type=XXX/year=YYYY/
#    数据源：data/financial_parquet/ 合并后的 parquet 文件
# ============================================================
def migrate_financial():
    log("\n" + "="*60)
    log("3/5 🏦 迁移财务数据 → financial/")
    
    src = DATA / "financial_parquet"
    dst_base = DATA / "financial"
    
    # 合并的 parquet 文件
    merged = sorted(src.glob("*.parquet"))
    merged = [f for f in merged if "_" not in f.stem]  # 排除单个股票的
    
    log(f"  源文件: {len(merged)} 张合并表")
    
    for fpath in merged:
        report_type = fpath.stem
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            log(f"  ⚠️ 读取失败 {fpath.name}: {e}")
            continue
        
        if df.empty:
            continue
        
        # 确保 end_date 存在且可分区
        if "end_date" not in df.columns:
            log(f"  ⚠️ {report_type}: 无 end_date 列，跳过分区")
            continue
        
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["year"] = df["end_date"].dt.year
        
        total_rows = len(df)
        n_years = df["year"].nunique()
        log(f"  📁 {report_type}: {total_rows:,} 行, {n_years} 个年份")
        
        for yr, grp in df.groupby("year"):
            part_dir = dst_base / f"report_type={report_type}" / f"year={int(yr)}"
            ensure_dir(part_dir)
            grp = grp.drop(columns=["year"])
            grp = grp.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
            out = part_dir / "data.parquet"
            
            if DRY_RUN:
                log(f"    [DRY] {yr}: {len(grp):>6,} 行 → {out}")
            else:
                grp.to_parquet(out, compression="snappy", index=False)
                log(f"    ✅ {yr}: {len(grp):>6,} 行 → {out}")
    
    # 也迁移旧 financial/*.csv 快照
    old_fin = DATA / "financial"
    old_csvs = sorted(old_fin.glob("financial_*.csv"))
    if old_csvs:
        log(f"\n  迁移旧财务快照 ({len(old_csvs)} 个):")
        for fpath in old_csvs:
            try:
                df = pd.read_csv(fpath, encoding="utf-8-sig")
            except:
                continue
            if df.empty:
                continue
            # 转 parquet 放到 financial/legacy/
            part_dir = dst_base / "legacy"
            ensure_dir(part_dir)
            out = part_dir / f"{fpath.stem}.parquet"
            if not DRY_RUN:
                df.to_parquet(out, compression="snappy", index=False)
                log(f"    ✅ {fpath.name} ({len(df)} 行)")
            else:
                log(f"    [DRY] {fpath.name} → {out}")
    
    if not DRY_RUN:
        meta = {
            "partitions": ["report_type", "year"],
            "source": "financial_parquet (tushare turbo_downloader)",
            "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(meta, open(dst_base / "_meta.json", "w"), ensure_ascii=False, indent=2)
        log(f"  📝 元数据已写")


# ============================================================
# 4. 📋 基本面指标：data/fundamental/year=YYYY/
#    数据源：data/daily_csv/ 中的 PE/PB/PS/PCF + 日线基础
#    说明：daily_parquet 只有 OHLCV，rich 指标在 daily_csv 中
# ============================================================
def migrate_fundamental():
    log("\n" + "="*60)
    log("4/5 📋 迁移基本面指标 → fundamental/")
    
    src = DATA / "daily_csv"
    dst_base = DATA / "fundamental"
    files = sorted(src.glob("*.csv"))
    log(f"  源文件: {len(files)} 个")
    
    if not files:
        log("  ⚠️ 无数据，跳过")
        return
    
    yearly = {}
    total_rows = 0
    
    for fpath in files:
        code = fpath.stem
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        if df.empty:
            continue
        
        # 提取估值指标 + 复权因子 + 基础行情
        val_cols = ["date", "code", "open", "high", "low", "close", "volume", "amount",
                     "adjustflag", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST"]
        avail = [c for c in val_cols if c in df.columns]
        df = df[avail].copy()
        df["stock_code"] = code
        df["trade_date"] = pd.to_datetime(df["date"])
        df["year"] = df["trade_date"].dt.year
        
        total_rows += len(df)
        for yr, grp in df.groupby("year"):
            grp = grp.drop(columns=["year"])
            yearly.setdefault(int(yr), []).append(grp)
    
    log(f"  总计 {total_rows:,} 行, 覆盖 {len(yearly)} 个年份")
    
    for yr in sorted(yearly.keys()):
        part_dir = dst_base / f"year={yr}"
        ensure_dir(part_dir)
        combined = pd.concat(yearly[yr], ignore_index=True)
        combined = combined.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        out = part_dir / "data.parquet"
        
        if DRY_RUN:
            log(f"  [DRY] {yr}: {len(combined):>8,} 行 → {out}")
        else:
            combined.to_parquet(out, compression="snappy", index=False)
            log(f"  ✅ {yr}: {len(combined):>8,} 行 → {out}")
    
    # 迁移行业分类 → fundamental/
    ind_src = DATA / "stock_basic" / "industry_classification.csv"
    if ind_src.exists():
        if not DRY_RUN:
            ensure_dir(dst_base / "static")
            import shutil
            shutil.copy(ind_src, dst_base / "static" / "industry_classification.csv")
            log(f"  ✅ 行业分类已迁移")
    
    if not DRY_RUN:
        meta = {
            "columns": ["stock_code", "trade_date", "open", "high", "low", "close", "volume",
                        "amount", "adjustflag", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST"],
            "partitions": ["year"],
            "source": "daily_csv (10jqka)",
            "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_rows": int(total_rows),
        }
        json.dump(meta, open(dst_base / "_meta.json", "w"), ensure_ascii=False, indent=2)
        log(f"  📝 元数据已写")


# ============================================================
# 5. 📉 指数/宏观：data/macro/index=XXX/year=YYYY/
#    数据源：data/csv/ 中的指数文件 (sh.000001 = 上证指数)
#           + data/stock_basic/hs300_daily.csv
# ============================================================
def migrate_macro():
    log("\n" + "="*60)
    log("5/5 📉 迁移指数/宏观 → macro/")
    
    src = DATA / "csv"
    dst_base = DATA / "macro"
    
    # 识别指数文件：读取CSV头，7列(仅OHLCV+amount)才是真正的指数
    # 个股有13+列(含PE/PB/PS/PCF等估值指标)
    index_files = []
    stock_files = []
    for f in src.glob("*.csv"):
        try:
            with open(f) as fh:
                header = next(csv.reader(fh))
            if len(header) <= 8:  # 指数只有 date,open,high,low,close,volume,amount
                index_files.append(f)
            else:
                stock_files.append(f)
        except:
            stock_files.append(f)
    
    log(f"  指数文件: {len(index_files)} 个, 个股文件: {len(stock_files)} 个(保留原路径)")
    
    # 迁移指数
    for fpath in index_files:
        try:
            df = pd.read_csv(fpath)
        except:
            continue
        if df.empty:
            continue
        
        index_name = fpath.stem
        df["trade_date"] = pd.to_datetime(df["date"])
        df["year"] = df["trade_date"].dt.year
        
        for yr, grp in df.groupby("year"):
            part_dir = dst_base / f"index={index_name}" / f"year={int(yr)}"
            ensure_dir(part_dir)
            grp = grp.drop(columns=["year"])
            grp = grp.sort_values("trade_date").reset_index(drop=True)
            out = part_dir / "data.parquet"
            
            if DRY_RUN:
                log(f"  [DRY] {index_name}/{yr}: {len(grp):>5,} 行 → {out}")
            else:
                grp.to_parquet(out, compression="snappy", index=False)
    
    # hs300_daily (成分股日调整记录)
    hs300_path = DATA / "stock_basic" / "hs300_daily.csv"
    if hs300_path.exists():
        try:
            df = pd.read_csv(hs300_path)
            if "year" not in df.columns and "date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["date"])
                df["year"] = df["trade_date"].dt.year
                # 按年分区
                for yr, grp in df.groupby("year"):
                    part_dir = dst_base / "index=hs300" / f"year={int(yr)}"
                    ensure_dir(part_dir)
                    grp = grp.drop(columns=["year"])
                    grp = grp.sort_values("trade_date").reset_index(drop=True)
                    out = part_dir / "data.parquet"
                    if not DRY_RUN:
                        grp.to_parquet(out, compression="snappy", index=False)
                        log(f"  ✅ hs300/{yr}: {len(grp):>5,} 行")
                    else:
                        log(f"  [DRY] hs300/{yr}: {len(grp):>5,} 行")
        except Exception as e:
            log(f"  ⚠️ hs300_daily 迁移失败: {e}")
    
    if not DRY_RUN:
        meta = {
            "partitions": ["index", "year"],
            "source": "csv/ + stock_basic/",
            "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(meta, open(dst_base / "_meta.json", "w"), ensure_ascii=False, indent=2)
        log(f"  📝 元数据已写")


# ============================================================
# 6. 🗂 元数据：data/meta/
#    股票列表、断点续传记录等
# ============================================================
def migrate_meta():
    log("\n" + "="*60)
    log("6/6 🗂 迁移元数据 → meta/")
    
    meta_dir = DATA / "meta"
    ensure_dir(meta_dir)
    
    # eligible_stocks.json
    src = DATA / "eligible_stocks.json"
    if src.exists():
        dst = meta_dir / "eligible_stocks.json"
        if DRY_RUN:
            log(f"  [DRY] {src.name} → {dst}")
        else:
            import shutil
            shutil.copy(src, dst)
            log(f"  ✅ eligible_stocks.json")
    
    # stock_codes.txt
    src = DATA / "stock_codes.txt"
    if src.exists():
        dst = meta_dir / "stock_codes.txt"
        if DRY_RUN:
            log(f"  [DRY] {src.name} → {dst}")
        else:
            shutil.copy(src, dst)
            log(f"  ✅ stock_codes.txt")
    
    # quality_report.json
    src = DATA / "quality_report.json"
    if src.exists():
        dst = meta_dir / "quality_report.json"
        if DRY_RUN:
            log(f"  [DRY] {src.name} → {dst}")
        else:
            shutil.copy(src, dst)
            log(f"  ✅ quality_report.json")
    
    # active_pool.json
    src = DATA / "financial" / "active_pool.json"
    if src.exists():
        dst = meta_dir / "active_pool.json"
        if DRY_RUN:
            log(f"  [DRY] {src.name} → {dst}")
        else:
            shutil.copy(src, dst)
            log(f"  ✅ active_pool.json")


# ============================================================
# 清理旧目录（仅 --cleanup 时）
# ============================================================
def do_cleanup():
    log("\n" + "="*60)
    log("🧹 清理旧目录...")
    
    old_dirs = ["daily_csv", "daily_parquet", "min5_csv", "stock_basic"]
    old_files = ["eligible_stocks.json", "stock_codes.txt", "quality_report.json"]
    
    for name in old_dirs:
        p = DATA / name
        if p.exists() and p.is_dir():
            # 先备份为 .old
            bak = DATA / f"{name}.old"
            if DRY_RUN:
                log(f"  [DRY] mv {name} → {name}.old")
            else:
                if bak.exists():
                    shutil.rmtree(bak)
                p.rename(bak)
                log(f"  ✅ {name} → {name}.old")
    
    for name in old_files:
        p = DATA / name
        if p.exists():
            if DRY_RUN:
                log(f"  [DRY] rm {name}")
            else:
                p.unlink()
                log(f"  ✅ 删除 {name}")
    
    # financial_parquet 太大，压缩归档
    fp = DATA / "financial_parquet"
    if fp.exists():
        bak = DATA / "financial_parquet.old"
        if DRY_RUN:
            log(f"  [DRY] mv financial_parquet/ → financial_parquet.old/")
        else:
            if bak.exists():
                shutil.rmtree(bak)
            fp.rename(bak)
            log(f"  ✅ financial_parquet/ → financial_parquet.old/")
    
    # csv/ 里的个股文件也可清理 (指数已迁移)
    csv_dir = DATA / "csv"
    stock_files_removed = 0
    if csv_dir.exists():
        for f in csv_dir.glob("*.csv"):
            code = f.stem.split(".")[-1]
            if len(code) == 6 and not (code.startswith("000") or code.startswith("399") or code.startswith("9")):
                if not DRY_RUN:
                    f.unlink()
                stock_files_removed += 1
        log(f"  ✅ 清理 {stock_files_removed} 个股CSV (指数未动)" if not DRY_RUN else f"  [DRY] 清理 {stock_files_removed} 个股CSV")


# ============================================================
# 主流程
# ============================================================
def main():
    log("📐 数据目录规范化迁移")
    log(f"  工作目录: {WORKSPACE}")
    log(f"  模式: {'🔍 DRY RUN (只预览)' if DRY_RUN else '🚀 正式迁移'}")
    
    if DRY_RUN:
        log("  提示: 去掉 --dry 执行实际迁移")
    
    t0 = time.time()
    
    migrate_daily()
    # migrate_minute()  # 分钟数据量大，单独脚本处理 scripts/migrate_minute.py
    migrate_financial()
    migrate_fundamental()
    migrate_macro()
    migrate_meta()
    
    if CLEANUP:
        do_cleanup()
    
    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"✅ 迁移{'预览' if DRY_RUN else '完成'}，耗时 {elapsed:.0f}s")
    
    # 最终目录结构
    log(f"\n📂 目标结构:")
    for root, dirs, files in os.walk(DATA):
        level = root.replace(str(DATA), "").count(os.sep)
        indent = "  " * level
        if level <= 4:
            log(f"{indent}{os.path.basename(root)}/")
            if level <= 3:
                for f in sorted(files)[:3]:
                    fpath = Path(root) / f
                    size = fpath.stat().st_size
                    log(f"{indent}  {f} ({size/1024:.0f}KB)")


if __name__ == "__main__":
    main()
