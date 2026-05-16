# ChanFund Fusion — 缠论融合策略

> 基于缠论信号 + 沪深300环境分类 + 波动率标准化仓位的量化股票策略。

## 版本变更记录

```
v1.4 ─────── 缠论基线 (夏普 3.44)           [旧框架，不可复现]
v2.0~2.5 ─── 信号因子化/评分模型整合         [修复后基准 v2.5: 0.062]
R1 ───────── 环境依赖TP/SL                   [里程碑 0.19]
v3.1 ─────── CSI300分类 + 波动率仓位         [理想环境定版 0.268]
v4.0 ─────── 滑点+涨跌停+成交量限制          [真实成本基准 0.046]
v5-A ─────── min_hold=3 + 信号拥挤度         [夏普最高 0.302，回撤-33%]
v5-B ─────── min_bi_len=6                    [❌ 关闭 -0.449]
v5-C ─────── BEAR不开新仓                    [⭐ 真实成本最佳 0.115]
```

### master — 稳定版
- v3.1（理想环境定版）、v5-C（真实成本最佳）

### dev — 测试/优化版
- 新因子、参数调优、板块指数过滤等实验性修改

## 快速运行

```bash
# master 分支（稳定版本）
python3 scripts/v31_backtest.py   # 理想环境
python3 scripts/v5c_backtest.py   # 真实成本最佳

# dev 分支（含实验性修改）
```

## 数据依赖

1. **Qlib Parquet** — A股日线 OHLCV（缠论信号计算）
2. **沪深300日线** — `data/index_sh000300.csv`（环境分类器）
3. **scores.parquet** — 基本面评分（可选）

## 结构

```
├── scripts/            # 回测脚本 (v2.5→v5c)
├── chanfund_fusion/    # 缠论引擎、配置、数据层
├── quant_framework/    # 统一框架（环境分类器、回测引擎）
├── memory/             # 研究记录
└── knowledge_base/     # 完整知识库
```

### 已关闭路径
基本面评分(LGBM)重构、30分钟缠论、成交量过滤、VOL_CAP单票上限、min_bi_len=6 → 详见 `knowledge_base/closed_paths.md`

## 许可

MIT
