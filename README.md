# ChanFund Fusion — 缠论融合策略

> 基于缠论信号 + 沪深300环境分类 + 波动率标准化仓位的量化股票策略。
> **理想版本**: v3.1 (夏普 0.268) | **真实成本最佳**: v5-C (夏普 0.115)

---

## 快速开始

```bash
# 回测 v3.1（理想环境，无滑点）
python3 scripts/v31_backtest.py

# 回测 v5-C（含滑点 + BEAR空仓，真实成本最佳）
python3 scripts/v5c_backtest.py

# 回测 v4.0（含滑点 + 涨跌停 + 成交量限制）
python3 scripts/v40_backtest.py
```

## 版本谱系

```
v1.4 (3.44) → v2.x → R1 (0.19) → v3.0 → v3.1 (0.268⭐)
→ v3.2 → v4.0 (0.046) → v4.1 → v5-A (0.302) → v5-C (0.115⭐)
→ v5-B (关闭❌)
```

| 版本 | 夏普 | 核心改进 | 说明 |
|:---|:---:|:--------|:----|
| v1.4 | 3.44 | 缠论基线 | 旧框架，不可复现 |
| v2.5 | 0.062 | 真实scores pre过滤 | 修复后基准 |
| **R1** | **0.19** | 环境依赖TP/SL | 里程碑版本 |
| **v3.1** | **0.268** | CSI300分类 + 波动率仓位 | **理想环境定版** |
| v4.0 | 0.046 | 滑点+涨跌停+成交量限制 | 真实成本基准 |
| v5-A | 0.302 | min_hold=3 + 信号拥挤度 | 夏普最高但回撤-33% |
| **v5-C** | **0.115** | BEAR不开新仓 | **真实成本最佳** |
| v5-B | -0.449 | min_bi_len=6 | ❌ 关闭 |

## 方法论

```
缠论信号 → CSI300环境分类器 → 波动率仓位 → 环境依赖TP/SL
   ↓                 ↓                 ↓               ↓
信号引擎       MA三线+ADX+ATR      signal/ATR²     BULL/OSC/BEAR
                                                  差异退出
```

### 环境分类器（基于沪深300真实OHLC）
三层过滤器判断市场状态：
- **趋势方向**：MA20/60/120 三线排列
- **趋势强度**：ADX(14) > 22
- **波动率阻隔**：ATR 比率历史分位数（85%分位→OSCILLATE）

### 波动率仓位公式
```python
vol_forecast = base_signal / atr_ratio      # 信号/波动率
risk_amount = cash × 0.04 × vol_forecast / atr_ratio  # 目标风险4%
```

### 入场过滤
- 股票池：全市场日均量前800只
- 基本面评分：scores.parquet < 30 拒绝
- 冷却期：止损后20天禁止入场

### v5-C 额外过滤
- BEAR 环境下不开新仓（仅 BULL + OSCILLATE 交易）

## 项目结构

```
chanfund-fusion/
├── scripts/              # 策略回测脚本（v2.5 → v5c）
│   ├── v31_backtest.py   # v3.1 回测
│   ├── v40_backtest.py   # v4.0 滑点+限制
│   ├── v5a_backtest.py   # 路径A: 持仓质量
│   ├── v5c_backtest.py   # 路径C: BEAR空仓
│   └── v5b_backtest.py   # 路径B: 缠论参数(已关闭)
├── chanfund_fusion/      # 基础库（缠论引擎、配置）
│   └── tech/
│       ├── chan_objects.py    # 数据结构（含Factor/Event）
│       ├── chan_structure.py  # 增量缠论状态机
│       └── signal_engine.py   # 信号引擎
├── output/               # 回测输出（v21 → v5c）
├── data/                 # 指数数据
│   └── index_sh000300.csv     # 沪深300日线
├── memory/               # 研究记录、缠论引擎分析
├── knowledge_base/       # 完整知识库
│   ├── version_history.md
│   ├── lessons_learned.md
│   ├── closed_paths.md
│   └── reference_chanlun_engines.md
└── v31_report.md         # v3.1 定版研究报告
```

## 数据依赖

1. **Qlib Parquet 数据** — A股日线 OHLCV（缠论信号计算）
2. **CSI300 日线数据** — `data/index_sh000300.csv`（环境分类器）
3. **scores.parquet** — 基本面评分数据

## 已关闭的探索路径

详见 `knowledge_base/closed_paths.md`：
- ❌ 基本面评分重构（LGBM）— 与缠论收益 r≈0
- ❌ 30分钟缠论 — 信号噪声放大，盈亏比<1.5
- ❌ 成交量过滤 — 800只池中阈值过低
- ❌ VOL_CAP单票上限 — 需总仓位而非单票控制
- ❌ fund_score≥60过滤 — 60分档avg_pnl为负
- ❌ min_bi_len=6 — 直接亏损

## 许可

MIT
