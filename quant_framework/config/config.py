"""
统一配置管理

职责：
1. 加载 YAML/JSON 配置文件
2. 提供默认值
3. 版本间配置差异的规范管理
4. 配置校验

用法：
    config = BacktestConfig.load("config_v2.4.yaml")
    # 或直接构造：
    config = BacktestConfig(
        name="v2.4",
        scores_path="financial_data/scores.parquet",
        env_classifier="ma",
        fund_score_min=40,
        fund_score_mode="post_trade",  # or "pre_trade"
    )
"""

from __future__ import annotations
import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class BacktestConfig:
    """统一回测配置"""
    
    # === 基本信息 ===
    name: str = "default"
    description: str = ""
    
    # === 数据 ===
    parquet_path: str = "quant/data/daily_parquet/_all.parquet"
    scores_path: str = ""
    warmup_days: int = 2431
    backtest_days: int = 1531
    
    # === 策略参数 ===
    initial_capital: float = 1_000_000.0
    max_positions: int = 5
    position_pct: float = 0.15  # 单只股票仓位百分比
    
    # === 环境分类器 ===
    env_classifier: str = "ma"   # "ma" or "er" or "none"
    env_params: dict = field(default_factory=lambda: {
        "vol_percentile": 0.85,
        "vol_min": 0.30,
        "ma_fast": 20,
        "ma_slow": 60,
    })
    env_multipliers: dict = field(default_factory=lambda: {
        "BULL": 1.0,
        "OSCILLATE": 0.8,
        "BEAR": 0.5,
    })
    
    # === 基本面评分 ===
    fund_score_source: str = "none"  # "parquet", "default", or "none"
    fund_score_default: float = 50.0
    fund_score_mode: str = "post_trade"  # "pre_trade" (拒绝) or "post_trade" (减仓)
    fund_score_min: float = 40.0  # pre_trade: 拒绝阈值; post_trade: 减仓阈值
    fund_score_pos_reduce: float = 0.5  # post_trade: 减仓比例
    
    # === 共振计算 ===
    resonance_enabled: bool = False
    tech_weight: float = 0.6
    fund_weight: float = 0.4
    
    # === 止损止盈 ===
    stop_loss_pct: float = 0.05
    take_profit_1: float = 0.08
    take_profit_2: float = 0.15
    trailing_stop_pct: float = 0.03
    
    # === 信号 ===
    min_resonance: float = 0.3
    
    # === 输出 ===
    output_dir: str = "output/"
    
    # === 日志 ===
    log_level: str = "INFO"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: str):
        d = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def load(cls, path: str) -> "BacktestConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
    
    @classmethod
    def load_v2_4_default(cls) -> "BacktestConfig":
        """v2.4 默认配置"""
        return cls(
            name="v2.4",
            description="v2.4: 真实scores，MA环境分类器，post-trade减仓",
            scores_path="financial_data/scores.parquet",
            fund_score_source="parquet",
            fund_score_mode="post_trade",
            fund_score_min=40,
            env_classifier="ma",
            resonance_enabled=False,
        )
    
    @classmethod
    def load_v2_1_default(cls) -> "BacktestConfig":
        """v2.1 配置（作为参看）"""
        return cls(
            name="v2.1_ref",
            description="v2.1参考: pre-trade过滤 + 共振加权",
            scores_path="financial_data/scores.parquet",
            fund_score_source="parquet",
            fund_score_mode="pre_trade",
            fund_score_min=30,
            env_classifier="none",
            resonance_enabled=True,
            tech_weight=0.6,
            fund_weight=0.4,
        )
