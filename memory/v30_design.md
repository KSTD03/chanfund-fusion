# ChanFund Fusion v3.0 设计稿

> 日期：2026-05-12
> 基线版本：R1（环境依赖TP/SL，夏普 0.19）
> 改进来源：Hikyuu 组件化架构 + pysystemtrade 波动率标准化

---

## 一、目标

在 R1 基线（夏普 0.19）基础上，通过两项改进实现夏普 > 0.25：

| 改进 | 预期增益 | 风险 |
|:---|:-------:|:---:|
| 环境分类器升级 | +0.03~0.05 | 低 |
| 波动率标准化仓位 | +0.03~0.08 | 中 |

---

## 二、改动内容

### 改动 1：环境分类器升级

**当前问题：**
- MA20/60 简单交叉 → OSCILLATE 仅覆盖 9% 交易日
- 环境切换滞后（牛市初期仍判 BEAR）
- 高波动率保护（85% 分位）过于粗暴

**改为三层过滤器：**

```python
def classify_environment(date, prices, returns):
    """
    返回：BULL / BEAR / OSCILLATE
    """
    # 层1：趋势方向 — MA三线排列
    ma20 = SMA(prices, 20)
    ma60 = SMA(prices, 60)
    ma120 = SMA(prices, 120)
    
    bull_trend = (ma20 > ma60) and (ma60 > ma120)
    bear_trend = (ma20 < ma60) and (ma60 < ma120)
    
    # 层2：趋势强度 — ADX
    adx = calc_adx(prices, 14)
    strong_trend = adx > 22  # 相比默认25略松
    
    # 层3：波动率阻隔 — ATR%
    atr_ratio = calc_atr(prices, 20) / prices[-1]
    high_vol = atr_ratio > SMA(atr_ratio, 60) * 1.5
    
    # 综合判断
    if bull_trend and strong_trend:
        return "BULL"
    elif bear_trend and strong_trend:
        return "BEAR"
    elif high_vol or (abs(ma20 - ma60) / ma60 < 0.02):
        # 高波动或均线粘合 → 震荡
        return "OSCILLATE"
    else:
        # 有方向但强度弱 → 按趋势走
        return "BULL" if bull_trend else "BEAR"
```

**预期效果：** OSCILLATE 从 9% → 15-20%，减少 BULL/BEAR 的误判。

### 改动 2：波动率标准化仓位

**当前公式（R1）：**
```python
tech_factor = signal_to_factor(signal_subtype, vol_ratio)  # 0.3-1.6
res = resonance(tech_factor, fund_score)                   # 0-1
pos_coeff = min(res, 1.0) * POSITION_SIZE / 0.20 * env_mult
cost = cash * POSITION_SIZE * pos_coeff
```

**改为：**
```python
# 1. 信号强度按波动率标准化（Carver forecast 概念）
atr_ratio = calc_atr(stock, 20) / entry_price  # ATR/价格
base_signal = {"third_point_buy": 0.6, "hard_divergence": 1.0}.get(subtype, 0.5)
vol_forecast = base_signal / max(atr_ratio, 0.01)  # 除波动率，自动缩放

# 2. 叠加基本面评分
fund_mult = fund_score / 100.0

# 3. 仓位计算
target_position_risk = 0.04  # 单只目标风险（4%）
position_amount = (cash * target_position_risk * vol_forecast * fund_mult) 
                / (atr_ratio * entry_price)
position_amount = min(position_amount, cash * 0.40)  # 单只上限40%

# 4. 环境乘数仅作为上限帽
env_cap = {"BULL": 1.0, "OSCILLATE": 0.7, "BEAR": 0.5}.get(env, 1.0)
position_amount *= env_cap

shares = int(position_amount / entry_price / 100) * 100
cost = shares * entry_price
```

**关键变化：** 仓位不再依赖 vol_ratio 乘数而是直接除波动率。高波动期自动降仓（无需判断），低波动期自动升仓。

### 改动 3（可选）：组件化重构

将当前 ~900 行回测循环拆分为独立模块：

```
chanfund_fusion/
├── core/
│   ├── environment.py      ← 环境分类器（改动1）
│   ├── position_sizer.py   ← 仓位计算器（改动2）
│   ├── stoploss.py         ← 止损管理（固定+环境+尾随）
│   ├── takeprofit.py       ← 止盈管理（TP1+TP2）
│   └── signal_engine.py    ← 现有（不改）
├── strategy/
│   └── ...
```

简化版本：不改目录结构，只提取 3 个函数到 `strategy/tech/` 下。

---

## 三、回测验证计划

### Step 1：只改环境分类器（v3.0-a）

- 不改仓位公式
- 只替换 `MAEnvClassifier()` 为三层过滤器
- 验证 OSCILLATE 覆盖率是否从 9% → 15%+
- 预期：夏普 0.20~0.22

### Step 2：波动率标准化 + 环境分类器（v3.0-b）

- 全部改动到位
- 预期：夏普 0.22~0.28

### Step 3：组件化重构（v3.0-c）

- 代码重构，不改逻辑
- 验证回测结果与 v3.0-b 一致
- 只做组织优化，不追求夏普提升

---

## 四、风险点

1. **ATR 计算成本** — 每只股票每天算 ATR 会增加约 20% 计算开销，但 ATR 本就在缠论状态机中有基础数据，可以复用
2. **极低波动率风险** — 如果某只票 ATR% 极低（如银行股），vol_forecast 会被过度放大。需要加 ATR 地板值：`max(atr_ratio, 0.005)`
3. **环境分类器参数** — ADX 阈值 22、ATR 倍数 1.5 需要验证，可做网格搜索

---

## 五、如果升幅不达预期的备选方向

| 方向 | 优先级 | 说明 |
|:---|:----:|:----|
| TP1/SL 网格优化 | P2 | 对 BULL/OSC/BEAR TP 和 SL 做网格搜索 |
| 基本面评分重构 | P3 | 换预测目标（不预测缠论收益，改用其他因子） |
| 分钟频数据 | P3 | 已验证过效果有限 |
| ML 入场过滤 | P3 | LightGBM 效果不显著 |

---

## 六、执行顺序

```
v3.0-a (环境分类器) → 验证 → v3.0-b (波动率仓位) → 验证 → 组件化
  夏普预期 0.20-0.22      如果OK       夏普预期 0.22-0.28        代码质量
```

建议：**先做 v3.0-a**（低风险，最多改 100 行代码），验证先单独看环境覆盖率指标，再跑全量回测。OK 了再做 v3.0-b。
