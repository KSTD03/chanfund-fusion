"""将K线CSV数据批量转换为Parquet格式（优化2）"""
import os, sys, time
from pathlib import Path
import pandas as pd

DATA_DIR = Path("/home/quant/.openclaw/workspace/quant/data/daily_csv")
OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/quant/data/daily_parquet")
PARQUET_COLS = ["date", "open", "high", "low", "close", "volume"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_files = sorted(DATA_DIR.glob("*.csv"))
print(f"📂 找到 {len(csv_files)} 个CSV文件, 开始转换...")

t0 = time.time()
converted = 0
for fpath in csv_files:
    try:
        df = pd.read_csv(fpath, dtype={"code": str})
        # 保留标准列
        cols_map = {c.strip().lower(): c for c in df.columns}
        keep = {k: cols_map[k] for k in PARQUET_COLS if k in cols_map}
        out = df[list(keep.values())].copy()
        out.columns = [c.strip().lower() for c in out.columns]
        code = fpath.stem  # 文件名不含扩展名
        out.to_parquet(OUTPUT_DIR / f"{code}.parquet", compression="snappy", index=False)
        converted += 1
        if converted % 500 == 0:
            pct = converted / len(csv_files) * 100
            print(f"  进度 {converted}/{len(csv_files)} ({pct:.0f}%)")
    except Exception as e:
        # 有些文件可能是空文件或格式特殊，跳过
        print(f"  ⚠️ 跳过 {fpath.name}: {e}")
        continue

elapsed = time.time() - t0
print(f"\n✅ 转换完成: {converted}/{len(csv_files)} 文件, 耗时 {elapsed:.0f}s")
print(f"📁 输出目录: {OUTPUT_DIR}")

# 对比大小
csv_size = sum(f.stat().st_size for f in csv_files)
parquet_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
print(f"  原始CSV: {csv_size/1024**3:.2f} GB")
print(f"  Parquet: {parquet_size/1024**3:.2f} GB")
print(f"  压缩比: {parquet_size/csv_size*100:.1f}%")
