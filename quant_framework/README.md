# quant_framework — 统一量化回测框架

> 创建日期：2026-05-10
> 目标：替代 `run_chanfund_v*.py` 和 `scripts/task*.py` 中的重复代码

## 架构

```
quant_framework/
├── core/
│   └── engine.py       # 统一回测引擎（单一 run() 入口）
├── data/
│   └── loader.py       # 统一数据加载（Qlib Parquet + scores.parquet）
├── signals/
│   └── chan.py         # 缠论信号检测器（placeholder）
├── environment/
│   └── classifier.py   # 环境分类器（MA/ER/None）
├── risk/
│   └── manager.py      # 风险管理器（止损/止盈/移动止损）
├── output/
│   └── reporter.py     # 结果报告器（CSV/JSON/Markdown）
├── config/
│   └── config.py       # 配置管理（dataclass + YAML）
├── README.md           # 本文件
└── requirements.txt    # 依赖
```

## 使用方式

```python
from quant_framework.core.engine import run_backtest
from quant_framework.config.config import BacktestConfig

# v2.4 配置
config = BacktestConfig.load_v2_4_default()
result = run_backtest(config)

# 自定义配置
config = BacktestConfig(
    name="my_test",
    fund_score_source="parquet",
    fund_score_mode="pre_trade",  # pre_trade 过滤（类似v2.1）
    fund_score_min=30,
    resonance_enabled=True,
)
result = run_backtest(config)
```

## 配置驱动

所有版本差异通过 YAML 配置控制，无需修改代码：

| 参数 | v2.1风格 | v2.3风格 | v2.4风格 |
|:----:|:--------:|:--------:|:--------:|
| fund_score_mode | pre_trade | post_trade (默认50) | post_trade (真实) |
| fund_score_min | 30 | 40 | 40 |
| resonance_enabled | True | False | False |
| env_classifier | none | ma | ma |

## 验证

```bash
python3 -c "
from quant_framework.config.config import BacktestConfig
c = BacktestConfig.load_v2_4_default()
print(c)
"
```
