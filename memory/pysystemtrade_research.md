# pysystemtrade 研究报告 — 框架设计理念

> 日期：2026-05-12
> 目的：借鉴 Rob Carver 系统性交易框架的设计理念

---

## 一、核心框架理念

pysystemtrade 是 Rob Carver 所著《Systematic Trading》的开源实现，核心理念是**系统性风险预算管理**。

### 系统处理流水线

```
Raw Data → Trading Rules → Forecast Scaling → Forecast Combination
→ Position Sizing → Portfolio Construction → P&L Accounting
```

每个阶段独立，通过 System 对象连接。

---

## 二、对我们最有价值的三个概念

### 2.1 波动率标准化（Volatility Normalization）

这是 Carver 框架的基石。信号（forecast）不是原始价格差距，而是：

```
forecast = raw_signal / vol
```

其中 vol 是近期波动率（年化标准差），这样：
- EWMAC(32,128) 的信号值在牛市熊市都是同一量级
- 波动率乘数不直接在仓位公式中，而是提前标准化信号
- **所有交易规则产生 -20 ~ +20 范围的 forecast**

**对我们的启示：**
当前你用的是 `POSITION_SIZE × env_mult × resonance`，其中 resonance 是 0-1 的缩放系数。
Carver 的思路是：**信号本身就应该除以波动率**，这样在高波动期自动轻仓，低波动期自动重仓，比额外的环境乘数更有机。

### 2.2 仓位 = 资本 × 风险预算 / 波动率

Carver 的核心仓位公式：

```
仓位 = 资本 × (目标风险百分比 × 信号强度) / (波动率 × 价格)
```

等价于你的：
```
成本 = cash × POSITION_SIZE × pos_coeff
```

但有一个关键区别：**Carver 的波动率是分母的一部分**，而你的 pos_coeff 依赖于因子评分和共振，和波动率无关。

### 2.3 IDM（Instrument Diversification Multiplier）

多标的持仓时，由于相关性 < 1，实际组合波动率低于各标的波动率之和。Carver 用 IDM 来补偿：

```
IDM = 1 / sqrt(平均相关性)
```

当持仓 5 只且平均相关性 0.3 时，IDM ≈ 1.8，可以安全地增加总暴露。

---

## 三、对你的策略的具体改进建议

### 3.1 信号 → 基于波动率的 forecast

当前：
```python
tech_factor = signal_to_factor(sig.signal_subtype, vol_ratio)
# base * min(vol_ratio, 2.0) → 0.6 ~ 1.6
res = resonance(tech_factor, fund_score)  # 0-1
pos_coeff = min(res, 1.0) * POSITION_SIZE / 0.20 * env_mult
```

改为 Carver 式：
```python
# 1. 缠论信号强度按波动率标准化
atr = calc_atr(stock, 20)  # 20日ATR
raw_signal = {"third_point_buy": 1.0, "hard_divergence": 1.5}.get(subtype, 0.5)
vol_normalized_signal = raw_signal / (atr / entry_price)  # 除以波动率/价格比

# 2. 基本面评分作为乘数因子
fund_mult = fund_score / 100.0  # 0-1

# 3. 仓位计算
target_risk = 0.20  # 目标组合波动率20%
position_risk = (atr / entry_price) * 2.0  # 单只波动率(2 sigma)
position = (capital * target_risk * vol_normalized_signal * fund_mult) 
         / (position_risk * price)
```

### 3.2 IDM 替代 ENV_MULT

当前环境乘数 BULL=1.0 / OSC=0.8 / BEAR=0.6 是手工预设。Carver 的 IDM 是根据持仓相关性自动计算，更客观：

```python
# 周期性计算当前持仓的相关性矩阵
# IDM = 1 / sqrt(avg_correlation)
# 如果持仓 5 只，avg_corr=0.3 → IDM=1.8
# 如果持仓 5 只，avg_corr=0.6 → IDM=1.29

# 环境依赖改为：仅决定IDM上限而非仓位
max_idm = {"BULL": 2.0, "OSCILLATE": 1.5, "BEAR": 1.2}.get(env, 1.5)
effective_idm = min(idm, max_idm)
```

### 3.3 净值曲线 → 目标风险回撤

Carver 的风险管理不仅看价格波动，还定期缩放仓位使「目标风险」保持一致。当组合波动率上升时自动降仓。

---

## 四、总结

| 概念 | 当前做法 | Carver 做法 | 可迁移性 |
|:---|:-------|:----------|:--------:|
| 信号标准化 | 固定 0.6/0.8 | 波动率标准化forecast | 高 |
| 仓位公式 | 20%×env_mult×resonance | 资本×风险×信号/(波动率×价格) | 高 |
| 多标的相关性 | 未考虑 | IDM 补偿 | 中 |
| 目标风险控制 | 固定止损 | 波动率缩放到目标 | 中 |

**最优先采用：** 波动率标准化信号（替换当前的 vol_ratio 乘数）
