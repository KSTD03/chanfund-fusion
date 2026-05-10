"""
配置模式验证 — config_schema.py
========================
验证 config.yaml 的完整性和参数合法性。
"""

from __future__ import annotations

import os
from typing import List, Optional
import yaml


_SCHEMA_HINTS = {
    "tech.bi_min_kbar": "int >= 3, 笔的最小K线数量",
    "tech.bi_min_kbar_dynamic": "bool, 允许根据ATR动态调整",
    "tech.macd.fast": "int, MACD快线参数",
    "tech.macd.slow": "int, MACD慢线参数",
    "tech.macd.signal": "int, MACD信号线参数",
    "tech.trial.position_pct": "float 0-1, 试探仓相对标准仓位的比例",
    "tech.trial.expire_kbars": "int, 试探仓最大持有K线数",
    "fund.fin_mix_days": "int, 财报平滑过渡天数",
    "fund.fin_mix_ratio": "float 0-1, 新财报初始权重",
    "fusion.buy_score_min": "int 0-100, 买入最低融合分",
    "fusion.daily_risk_budget_pct": "float 0-1, 日风险预算(总权益%)",
    "position.single_stock_max_pct": "float 0-1, 单只最大仓位",
    "position.sector_max_pct": "float 0-1, 单行业最大暴露",
    "position.slippage": "float, 滑点率",
    "position.risk_per_trade_pct": "float 0-1, 单笔风险比例",
    "backtest.initial_cash": "float > 0, 初始资金",
    "backtest.commission": "float, 佣金率",
    "backtest.tax": "float, 印花税率",
}


def load_config(path: str = None) -> dict:
    """加载配置文件

    Args:
        path: 配置文件路径，None表示使用默认路径

    Returns:
        config: 配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置验证失败
    """
    if path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "config.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """验证配置合法性

    Args:
        config: 配置字典

    Raises:
        ValueError: 配置不合法
    """
    errors: List[str] = []

    # 检查必要顶级字段
    required = ["tech", "fund", "fusion", "position"]
    for key in required:
        if key not in config:
            errors.append(f"Missing required section: '{key}'")

    # tech 验证
    tech = config.get("tech", {})
    if "bi_min_kbar" in tech and tech["bi_min_kbar"] < 3:
        errors.append("tech.bi_min_kbar must >= 3")

    # macd 验证
    macd = tech.get("macd", {})
    if macd.get("fast", 12) >= macd.get("slow", 26):
        errors.append("tech.macd.fast must < tech.macd.slow")

    # fusion 验证
    fusion = config.get("fusion", {})
    buy_min = fusion.get("buy_score_min", 60)
    strong_buy = fusion.get("strong_buy_score", 85)
    if buy_min >= strong_buy:
        errors.append("fusion.buy_score_min must < fusion.strong_buy_score")

    # position 验证
    pos = config.get("position", {})
    single = pos.get("single_stock_max_pct", 0.08)
    sector = pos.get("sector_max_pct", 0.25)
    if single > sector:
        errors.append("single_stock_max_pct must <= sector_max_pct")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """合并覆盖配置（支持 experiment_overrides.yaml 模式）

    Args:
        base_config: 基础配置
        overrides: 覆盖配置（只写需要改动的参数）

    Returns:
        merged: 合并后的配置
    """
    merged = {}
    for key, value in base_config.items():
        if key in overrides and isinstance(value, dict) and isinstance(overrides[key], dict):
            merged[key] = {**value, **overrides[key]}
        elif key in overrides:
            merged[key] = overrides[key]
        else:
            merged[key] = value
    return merged


def display_config_hints():
    """打印配置项提示"""
    print("=== Config Schema Hints ===")
    for key, hint in sorted(_SCHEMA_HINTS.items()):
        print(f"  {key}: {hint}")
