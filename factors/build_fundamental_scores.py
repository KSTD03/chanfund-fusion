#!/usr/bin/env python3
"""
重建基本面因子评分 scores.parquet

读取 financial_parquet.old/ 中的合并财务数据，
为每只股票每季度计算基本面得分 (0-100)，
输出到 financial_data/scores.parquet。

评分维度:
  1. ROE          (0-25) 盈利能力
  2. 毛利率       (0-15) 竞争壁垒
  3. 资产负债率   (0-15) 财务安全
  4. FCF/净利润比 (0-15) 现金流质量
  5. PE 百分位    (0-10) 估值
  6. PB 百分位    (0-10) 估值
  7. 应计利润     (0-5)  盈余质量
  8. 营收增长率   (0-5)  成长

用法:
  python3 chanfund_fusion/factors/build_fundamental_scores.py --rebuild
"""

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

WORKSPACE = Path(__file__).resolve().parent.parent.parent
FIN_OLD = WORKSPACE / "quant" / "data" / "financial_parquet.old"
OUTPUT = WORKSPACE / "financial_data" / "scores.parquet"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load_parquet(name):
    f = FIN_OLD / name
    if f.exists():
        df = pd.read_parquet(f)
        log(f"  📥 {name}: {len(df):,} 行, {df['ts_code'].nunique()} 只")
        return df
    log(f"  ⚠️ {name} 未找到")
    return None

def build_scores():
    t0 = time.time()
    log("🔨 重建基本面因子评分")
    log(f"  数据源: {FIN_OLD}")
    log(f"  输出: {OUTPUT}")

    # 1. 加载数据
    ind = load_parquet("fina_indicator.parquet")   # roe, gross_margin, bps, eps
    inc = load_parquet("income.parquet")            # total_revenue, basic_eps
    bs  = load_parquet("balancesheet.parquet")      # total_assets, total_liab, total_hldr_eqy
    cf  = load_parquet("cashflow.parquet")          # net_profit, free_cashflow, n_cashflow_act
    fct = load_parquet("forecast.parquet")           # net_profit_min/max

    # 2. 标准化日期
    for df in [ind, inc, bs, cf]:
        if df is not None:
            df["end_date"] = pd.to_datetime(df["end_date"])
    
    # 3. 合并到统一表
    log("\n  🔗 合并财务表...")
    merged = ind.copy()
    
    # merge income
    inc_keep = inc[["ts_code", "end_date", "total_revenue", "basic_eps"]].copy()
    merged = merged.merge(inc_keep, on=["ts_code", "end_date"], how="left")
    
    # merge balancesheet
    bs_keep = bs[["ts_code", "end_date", "total_assets", "total_liab", 
                   "total_hldr_eqy_exc_min_int", "money_cap", "accounts_receiv",
                   "inventories", "st_borr", "lt_borr"]].copy()
    merged = merged.merge(bs_keep, on=["ts_code", "end_date"], how="left")
    
    # merge cashflow
    cf_keep = cf[["ts_code", "end_date", "net_profit", "free_cashflow",
                   "n_cashflow_act", "c_fr_sale_sg"]].copy()
    merged = merged.merge(cf_keep, on=["ts_code", "end_date"], how="left")
    
    # 4. 计算衍生指标
    log("  🧮 计算衍生指标...")
    merged["gross_margin"] = merged["gross_margin"].fillna(0)
    merged["roe"] = merged["roe"].fillna(0)
    
    # 资产负债率
    merged["debt_ratio"] = merged["total_liab"].fillna(0) / merged["total_assets"].replace(0, np.nan).fillna(1)
    merged["debt_ratio"] = merged["debt_ratio"].clip(0, 1)
    merged["debt_safety"] = 1 - merged["debt_ratio"]  # 越高越安全
    
    # FCF/净利润比
    merged["fcf_ratio"] = merged["free_cashflow"].fillna(0) / merged["net_profit"].replace(0, np.nan).abs().fillna(1)
    merged["fcf_ratio"] = merged["fcf_ratio"].clip(-2, 5)
    
    # 应计利润 (Accrual = 经营现金流 - 净利润)/总资产
    merged["accrual"] = (merged["n_cashflow_act"].fillna(0) - merged["net_profit"].fillna(0)) / merged["total_assets"].replace(0, 1)
    
    # 营收增长率 (YoY, 同一只股票)
    merged = merged.sort_values(["ts_code", "end_date"])
    merged["revenue_prev"] = merged.groupby("ts_code")["total_revenue"].shift(4)  # 4季度前
    merged["revenue_growth"] = (merged["total_revenue"] - merged["revenue_prev"]) / merged["revenue_prev"].replace(0, np.nan)
    merged["revenue_growth"] = merged["revenue_growth"].clip(-1, 5).fillna(0)
    
    # 5. 截面排名打分 (每季度每个指标在同季度全A股中排名)
    log("  📊 截面排名打分...")
    
    def rank_score(series, ascending=True, weight=1.0, cap=100):
        """截面百分位打分，ascending=True表示越大越好"""
        if series.dtype not in [np.float64, np.int64, np.float32]:
            series = series.astype(float)
        result = series.rank(pct=True) if ascending else (1 - series.rank(pct=True))
        return (result * weight * cap).clip(0, weight * cap)
    
    score_components = {}
    
    for yr in range(2005, 2027):
        for q in range(1, 5):
            mask = (merged["end_date"].dt.year == yr) & (merged["end_date"].dt.quarter == q)
            idx = merged[mask].index
            if len(idx) == 0:
                continue
            
            # ROE (0-25)
            merged.loc[idx, "score_roe"] = rank_score(merged.loc[idx, "roe"], ascending=True, weight=0.25)
            # 毛利率 (0-15)
            merged.loc[idx, "score_gross"] = rank_score(merged.loc[idx, "gross_margin"], ascending=True, weight=0.15)
            # 负债安全 (0-15)
            merged.loc[idx, "score_debt"] = rank_score(merged.loc[idx, "debt_safety"], ascending=True, weight=0.15)
            # FCF质量 (0-15)
            merged.loc[idx, "score_fcf"] = rank_score(merged.loc[idx, "fcf_ratio"], ascending=True, weight=0.15)
            # 应计利润 (0-5, 越小越好)
            merged.loc[idx, "score_accrual"] = rank_score(merged.loc[idx, "accrual"], ascending=False, weight=0.05)
            # 营收增长 (0-5)
            merged.loc[idx, "score_growth"] = rank_score(merged.loc[idx, "revenue_growth"], ascending=True, weight=0.05)
            # PE估值 (0-10)
            merged.loc[idx, "score_pe"] = rank_score(merged.loc[idx, "basic_eps"].fillna(0), ascending=True, weight=0.10)
            # PB估值 (0-10)
            merged.loc[idx, "score_pb"] = rank_score(merged.loc[idx, "bps"].fillna(0), ascending=True, weight=0.10)
    
    # 6. 综合得分
    score_cols = ["score_roe", "score_gross", "score_debt", "score_fcf",
                  "score_accrual", "score_growth", "score_pe", "score_pb"]
    avail_cols = [c for c in score_cols if c in merged.columns]
    merged["fund_score"] = merged[avail_cols].sum(axis=1).clip(0, 100).round(0).astype(int)
    
    # 7. 输出
    log(f"\n  💾 输出评分结果...")
    result = merged[["ts_code", "end_date", "fund_score"]].dropna(subset=["fund_score"]).copy()
    result["end_date"] = result["end_date"].dt.strftime("%Y-%m-%d")
    result = result.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    
    result.to_parquet(OUTPUT, compression="snappy", index=False)
    
    # 统计
    elapsed = time.time() - t0
    log(f"\n{'='*50}")
    log(f"✅ 评分重建完成！{elapsed:.0f}s")
    log(f"  总行数: {len(result):,}")
    log(f"  股票数: {result['ts_code'].nunique():,}")
    log(f"  日期范围: {result['end_date'].min()} ~ {result['end_date'].max()}")
    log(f"  得分分布:")
    log(f"    mean={result['fund_score'].mean():.1f}  median={result['fund_score'].median():.0f}")
    log(f"    min={result['fund_score'].min()}  max={result['fund_score'].max()}")
    log(f"    Q1={result['fund_score'].quantile(0.25):.0f}  Q3={result['fund_score'].quantile(0.75):.0f}")
    log(f"  输出: {OUTPUT}")
    log(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="重建评分")
    args = parser.parse_args()
    if args.rebuild:
        build_scores()
    else:
        print("请使用 --rebuild 参数")
