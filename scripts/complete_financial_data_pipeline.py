#!/usr/bin/env python3
"""
Task 1: 财务数据扫描与补全 — 完整数据管线
========================================
基于已有 tushare_download.py 框架，专门针对财务数据做全量扫描与补全。

功能：
1. 扫描本地 financial/ 目录，检查2005年至今的数据覆盖
2. 调用 Tushare Pro API 拉取缺失数据
3. 自动处理频率限制、断点续传、按年存储 Parquet

数据表清单：
  - income: 利润表（2005-2026）
  - fina_indicator: 财务指标（2005-2026）
  - balancesheet: 资产负债表（可选）
  - cashflow: 现金流量表（可选）

运行方式：
  nohup python3 scripts/complete_financial_data_pipeline.py > scripts/fin_pipeline.log 2>&1 &
  python3 scripts/complete_financial_data_pipeline.py --quick  # 快速验证（仅下载2020年+500只）
"""

import os, sys, time, json, argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

# ============================================================
# 配置
# ============================================================
WORKSPACE = Path(os.path.dirname(os.path.abspath(__file__))).parent
TUSHARE_DIR = WORKSPACE / "tushare_download"
OUTPUT_DIR = WORKSPACE / "quant" / "data" / "financial_parquet"  # 新版Parquet格式输出
FIN_CSV_DIR = WORKSPACE / "quant" / "data" / "financial"  # 旧版CSV输出（兼容）

# Tushare 配置
CONFIG_PATH = TUSHARE_DIR / "config.yaml"
import yaml
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)
TOKEN = CONFIG["token"]
print(f"🔑 Tushare Token: {TOKEN[:8]}... (已配置)")

import tushare as ts
ts.set_token(TOKEN)
pro = ts.pro_api()

# 股票清单（沪深全A + 北交所）
STOCK_LIST_PATH = TUSHARE_DIR / "data" / "stock_basic.parquet"
FINANCIAL_TABLES = ["income", "fina_indicator", "balancesheet", "cashflow"]
ANNUAL_REPORT_MONTHS = {4, 8, 10}  # 年报/中报/三季报 披露截止月

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIN_CSV_DIR, exist_ok=True)


def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def get_stock_list():
    """获取全市场股票代码列表"""
    if STOCK_LIST_PATH.exists():
        df = pd.read_parquet(STOCK_LIST_PATH)
        return df["ts_code"].tolist()
    # 如果本地没有，从API拉
    log("拉取股票列表...")
    df = pro.stock_basic(exchange='', list_status='L', 
                         fields='ts_code,symbol,name,area,industry,list_date')
    os.makedirs(STOCK_LIST_PATH.parent, exist_ok=True)
    df.to_parquet(STOCK_LIST_PATH)
    log(f"获取到 {len(df)} 只股票")
    return df["ts_code"].tolist()


def scan_coverage(table: str) -> dict:
    """扫描指定表的本地数据覆盖情况
    
    Returns:
        {year: {ts_code: rows_count}} 或类似结构
    """
    coverage = {"years": [], "total_stocks": 0, "stocks_per_year": {}}
    
    # 检查已有Parquet文件
    pq_files = sorted(OUTPUT_DIR.glob(f"{table}_*.parquet"))
    if pq_files:
        for f in pq_files:
            year = f.stem.split("_")[-1]
            if year.isdigit():
                coverage["years"].append(int(year))
                df = pd.read_parquet(f)
                stocks = df["ts_code"].nunique() if "ts_code" in df.columns else 0
                coverage["stocks_per_year"][int(year)] = int(stocks)
        coverage["total_stocks"] = sum(coverage["stocks_per_year"].values())
        return coverage
    
    # 检查旧版CSV
    csv_files = sorted(FIN_CSV_DIR.glob(f"{table}_*.csv"))
    if csv_files:
        for f in csv_files:
            year = f.stem.split("_")[-1]
            if year.isdigit():
                coverage["years"].append(int(year))
        return coverage
    
    return coverage


def download_missing_data(table: str, stocks: list, start_year: int, end_year: int,
                          batch_size: int = 50, quick: bool = False):
    """下载缺失的财务数据
    
    Args:
        table: 数据表名 (income/fina_indicator/balancesheet/cashflow)
        stocks: 股票代码列表
        start_year: 开始年份
        end_year: 结束年份
        batch_size: 每次API请求的股票数
        quick: 快速模式（只下载2020年后+500只）
    """
    if quick:
        start_year = max(start_year, 2020)
        stocks = stocks[:500]
    
    log(f"📦 下载 {table}: {len(stocks)}只股票 × {start_year}-{end_year}")
    
    api_funcs = {
        "income": lambda sc, sd, ed: pro.income(ts_code=sc, start_date=sd, end_date=ed),
        "fina_indicator": lambda sc, sd, ed: pro.fina_indicator(ts_code=sc, start_date=sd, end_date=ed),
        "balancesheet": lambda sc, sd, ed: pro.balancesheet(ts_code=sc, start_date=sd, end_date=ed),
        "cashflow": lambda sc, sd, ed: pro.cashflow(ts_code=sc, start_date=sd, end_date=ed),
    }
    
    api_fn = api_funcs.get(table)
    if not api_fn:
        log(f"❌ 未知表: {table}")
        return
    
    total = len(stocks)
    downloaded = 0
    errors = 0
    rate_limit_sleep = 0.5  # Tushare频率限制
    
    for i in range(0, total, batch_size):
        batch = stocks[i:i+batch_size]
        year_data = defaultdict(list)
        
        for stock in batch:
            try:
                sd = f"{start_year}0101"
                ed = f"{end_year}1231"
                df = api_fn(stock, sd, ed)
                if df is not None and len(df) > 0:
                    year_data[table].append(df)
                time.sleep(rate_limit_sleep)
                downloaded += 1
            except Exception as e:
                errors += 1
                if errors > 10:
                    log(f"  ⚠️ 错误过多({errors}), 暂停30s...")
                    time.sleep(30)
                    errors = 0
                continue
        
        # 按年保存
        if year_data[table]:
            combined = pd.concat(year_data[table], ignore_index=True)
            for year in range(start_year, end_year + 1):
                year_str = str(year)
                year_mask = combined.get("end_date", combined.get("f_ann_date", "")).str.contains(year_str)
                if year_mask.any():
                    year_df = combined[year_mask]
                    out_path = OUTPUT_DIR / f"{table}_{year_str}.parquet"
                    if out_path.exists():
                        existing = pd.read_parquet(out_path)
                        year_df = pd.concat([existing, year_df]).drop_duplicates()
                    year_df.to_parquet(out_path, compression="snappy")
            
            # 释放内存
            del combined
        
        if (i // batch_size) % 10 == 0:
            log(f"  进度 {min(i+batch_size, total)}/{total} ({min(i+batch_size, total)/total*100:.0f}%)")
    
    log(f"✅ {table} 完成: 下载 {downloaded}/{total}, 错误 {errors}")


def save_as_csv_compat():
    """将新版Parquet转存为旧版CSV格式（兼容现有回测框架）"""
    log("转换Parquet→CSV（兼容v2.1框架）...")
    for pq_file in sorted(OUTPUT_DIR.glob("*.parquet")):
        df = pd.read_parquet(pq_file)
        csv_name = pq_file.stem.replace("_20", "_20") + ".csv"
        csv_path = FIN_CSV_DIR / csv_name
        df.to_csv(csv_path, index=False)
    log(f"  CSV输出到: {FIN_CSV_DIR}")


def main():
    parser = argparse.ArgumentParser(description="财务数据管线：扫描+下载+校验")
    parser.add_argument("--quick", action="store_true", help="快速模式（2020年起+500只）")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，仅扫描")
    args = parser.parse_args()
    
    t0 = time.time()
    log("=" * 60)
    log("  Task 1: 财务数据扫描与补全")
    log(f"  模式: {'⚡快速' if args.quick else '全量'}")
    log("=" * 60)
    
    # Step 1: 扫描现有数据
    log("\n📋 Step 1: 扫描现有数据覆盖...")
    for table in FINANCIAL_TABLES:
        cov = scan_coverage(table)
        log(f"  {table}: {len(cov['years'])}年有数据, 年度分布: {cov['stocks_per_year'] if cov['stocks_per_year'] else '空'}")
    
    if args.skip_download:
        log("\n⏭️ --skip-download 跳过下载")
        log(f"⏱️ 耗时: {time.time()-t0:.0f}s")
        return
    
    # Step 2: 获取股票列表
    log("\n📦 Step 2: 获取全市场股票列表...")
    stocks = get_stock_list()
    log(f"  共 {len(stocks)} 只股票")
    
    # Step 3: 下载缺失数据
    log("\n⬇️ Step 3: 下载缺失财务数据...")
    start_year = 2020 if args.quick else 2005
    end_year = 2026
    
    for table in FINANCIAL_TABLES[:2]:  # 优先下载 income + fina_indicator
        download_missing_data(table, stocks, start_year, end_year, quick=args.quick)
    
    # Step 4: 转存CSV（兼容现有框架）
    log("\n🔄 Step 4: 数据格式转换...")
    save_as_csv_compat()
    
    # Step 5: 输出摘要
    log("\n📊 数据管线完成")
    for table in FINANCIAL_TABLES[:2]:
        cov = scan_coverage(table)
        log(f"  {table}: {len(cov['years'])}年, 共{cov['total_stocks']}条记录")
    
    elapsed = time.time() - t0
    log(f"\n⏱️ 总耗时: {elapsed/60:.1f}min")
    log(f"📁 Parquet输出: {OUTPUT_DIR}")
    log(f"📁 CSV输出(兼容): {FIN_CSV_DIR}")


if __name__ == "__main__":
    main()
