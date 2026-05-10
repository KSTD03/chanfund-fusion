"""
统一数据加载器

职责：
1. 从 Qlib Parquet 加载行情数据
2. 从 scores.parquet 加载基本面评分
3. 格式转换（Tushare ↔ Qlib 股票代码）
4. 指数数据加载（用于环境分类器）
"""

from __future__ import annotations

import time
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from quant_framework.config.config import BacktestConfig

logger = logging.getLogger(__name__)

# == 股票代码转换映射 ==
_TUSHARE_TO_QLIB = {"SH": "sh", "SZ": "sz", "BJ": "bj"}
_QLIB_TO_TUSHARE = {"sh": "SH", "sz": "SZ", "bj": "BJ"}


def to_ts_code(qlib_code: str) -> str:
    """将 Qlib 代码格式 (sh.600519) 转为 Tushare 格式 (600519.SH)"""
    parts = qlib_code.split(".")
    if len(parts) != 2:
        return qlib_code
    market = _QLIB_TO_TUSHARE.get(parts[0], parts[0].upper())
    return f"{parts[1]}.{market}"


def to_qlib_code(ts_code: str) -> Optional[str]:
    """将 Tushare 代码格式 (600519.SH) 转为 Qlib 格式 (sh.600519)"""
    parts = ts_code.split(".")
    if len(parts) != 2:
        return None
    code, market = parts
    qlib_market = _TUSHARE_TO_QLIB.get(market.upper())
    if qlib_market:
        return f"{qlib_market}.{code}"
    return None


class DataLoader:
    """统一数据加载器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.workspace = Path(config.parquet_path).parent.parent.parent
        if not self.workspace.exists():
            self.workspace = Path("/home/quant/.openclaw/workspace")
    
    def load_all(self) -> dict:
        """加载全部数据"""
        parquet_path = Path(self.config.parquet_path)
        if not parquet_path.exists():
            # 尝试默认路径
            parquet_path = self.workspace / "quant" / "data" / "daily_parquet" / "_all.parquet"
        
        if not parquet_path.exists():
            logger.error(f"❌ 数据文件不存在: {parquet_path}")
            return {}
        
        logger.info(f"📂 加载行情数据: {parquet_path}")
        t0 = time.time()
        
        df = pd.read_parquet(parquet_path)
        df.columns = [c.lower() for c in df.columns]
        
        logger.info(f"   ✅ {len(df)} 行, {time.time()-t0:.0f}s")
        
        return {"df": df, "source": str(parquet_path)}
    
    def load_market_data(self, data: dict) -> Tuple[List[date], dict]:
        """从 dataframe 提取行情数据
        
        Returns:
            (all_dates, daily_data) 
            - all_dates: 排序后的日期列表
            - daily_data: {date_str: {symbol: kbar}}
        """
        df = data.get("df")
        if df is None or df.empty:
            return [], {}
        
        # 标准化列名
        col_map = {
            "date": "date", "code": "symbol",
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume", 
            "amount": "amount", "vwap": "vwap",
        }
        
        # 重命名
        df = df.rename(columns={c: c.lower() for c in df.columns})
        
        # 确保需要的列存在
        required = ["date", "symbol", "close"]
        for col in required:
            if col not in df.columns:
                logger.error(f"❌ 缺少必填列: {col}")
                return [], {}
        
        # 排序
        df = df.sort_values(["date", "symbol"])
        
        # 转成 {date: {symbol: kbar}}
        daily_data = defaultdict(dict)
        all_dates = sorted(df["date"].unique())
        
        for _, row in df.iterrows():
            d = str(row["date"])
            daily_data[d][row["symbol"]] = {
                "open": row.get("open", 0),
                "high": row.get("high", 0),
                "low": row.get("low", 0),
                "close": row["close"],
                "volume": row.get("volume", 0),
                "amount": row.get("amount", 0),
                "vwap": row.get("vwap", 0),
            }
        
        return all_dates, dict(daily_data)
    
    def load_scores(self) -> Dict:
        """加载基本面评分数据
        
        Returns:
            {ts_code: {end_date: fund_score}}
        """
        if not self.config.fund_score_source:
            return {}
        
        scores_path = Path(self.config.scores_path)
        if not scores_path.exists():
            scores_path = self.workspace / "financial_data" / "scores.parquet"
        
        if not scores_path.exists():
            logger.warning(f"⚠️ scores.parquet 不存在: {scores_path}")
            return {}
        
        t0 = time.time()
        logger.info(f"📥 加载 scores.parquet: {scores_path}")
        
        df = pd.read_parquet(scores_path)
        logger.info(f"   ✅ {len(df)} 行, {df['ts_code'].nunique()} 只股票, {time.time()-t0:.0f}s")
        
        scores_cache = {}
        for _, row in df.iterrows():
            ts_code = row["ts_code"]
            end_date = str(row["end_date"])
            score = float(row["fund_score"])
            scores_cache.setdefault(ts_code, {})[end_date] = score
        
        return scores_cache
