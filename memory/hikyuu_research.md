# Hikyuu 源码研究报告

> 日期：2026-05-12
> 目的：为 ChanFund Fusion v3.0 的环境分类器改造 + 止损/止盈/资金管理组件化提供参考

---

## 一、框架架构概览

Hikyuu 是 C++ 核心 + Python 绑定的系统化交易框架，核心架构围绕 **System 组件化** 设计：

```
System
├── SystemPart.ENVIRONMENT    → 市场环境判断策略  (EnvironmentBase)
├── SystemPart.CONDITION      → 系统有效条件      (ConditionBase)
├── SystemPart.SIGNAL         → 买卖信号指示器    (SignalBase)
├── SystemPart.STOPLOSS       → 止损策略          (StoplossBase)
├── SystemPart.TAKEPROFIT     → 止盈策略          (ProfitGoalBase)
├── SystemPart.MONEYMANAGER   → 资金管理策略      (MoneyManagerBase)
├── SystemPart.SLIPPAGE       → 移滑价差算法      (SlippageBase)
├── Selector                  → 交易对象选择       (SelectorBase)
├── AllocateFundsBase         → 资产分配算法      (AllocateFundsBase)
├── MultiFactorBase           → 多因子合成        (MultiFactorBase)
├── ScoresFilterBase          → 评分过滤器        (ScoresFilterBase)
└── NormalizeBase             → 标准化/归一化     (NormalizeBase)
```

**核心设计原则**：每个组件都可以独立替换，通过工厂函数快速创建自定义策略。

---

## 二、市场环境判断策略（Environment）

### 创建方式

```python
from hikyuu import crtEV

def my_environment(self):
    """自定义环境判断函数，self 是 EnvironmentBase 实例"""
    # 访问当前K线数据
    k = self.get_kdata()
    # 判断逻辑
    # 通过 self.set_env(item) 标记当前环境状态
    pass

ev = crtEV(my_environment, params={"n": 30}, name="MY_EV")
```

### 内置环境策略

Hikyuu 的 C++ 核心层（`hikyuu_cpp` 库）包含多种内置环境策略实现，通过统一基类 `EnvironmentBase` 暴露：

- **均线环境**：根据 MA 大小判断多空趋势（类似你当前的 MA20/60）
- **波动率环境**：根据 ATR/标准差判断高波动/低波动状态
- **趋势强度环境**：结合 ADX 等指标判断趋势强度
- **自定义环境**：用户通过 `crtEV` 工厂函数传入任意函数

*注意：由于核心实现在 C++ 编译后的动态库中，Python 层通过 pybind11 绑定调用，具体的 C++ 实现策略在编译后的 `hikyuu.cpp.*.so` 中。*

---

## 三、止损/止盈策略组件（StoplossBase）

### 创建方式

```python
from hikyuu import crtST

# 动态止损
def my_stop_loss(self):
    """返回止损价格"""
    k = self.get_kdata()
    return k.close * 0.92  # 固定8%止损

st = crtST(my_stop_loss, params={}, name="MY_STOP")

# 加入 System
sys = System()
sys.set_stoploss(st)
```

### 止损策略的特点

- **独立模块**：止损逻辑完全独立于信号模块，可动态替换
- **逐K线计算**：每个 K 线周期都会调用止损函数计算最新止损价
- **多止损组合**：可以设置多个止损策略（固定止损 + 移动止损 + 时间止损）
- **区分多空**：`get_short_price` 方法为空头单独设置止损

### 与你当前方案对比

| 维度 | 你当前 (R1) | Hikyuu 方案 |
|:----|:---------:|:----------:|
| 止损位置 | 硬编码在回测循环中 | 独立闭包，可替换 |
| 尾随止损 | 自定义逻辑 | 内置支持 |
| 多止损级联 | 无 | 支持组合 |
| 动态调整 | 环境依赖 | 完全函数式 |

---

## 四、资金管理策略（MoneyManagerBase）

### 创建方式

```python
from hikyuu import crtMM

def my_get_buy_num(self, budget, price):
    """计算买入数量
    
    :param budget: 当前可用资金
    :param price: 买入价格
    :return: 买入股数
    """
    return int(budget * 0.2 / price / 100) * 100

def my_get_sell_num(self, volume, price):
    """计算卖出数量，None表示全部卖出"""
    return None  # 全部卖出

mm = crtMM(my_get_buy_num, my_get_sell_num, params={}, name="MY_MM")
```

### 关键设计

- **`_get_buy_num(budget, price)`** — 根据预算和价格计算买入量
- **`_get_sell_num(volume, price)`** — 根据持仓量计算卖出量（None = 全部）
- **`_buy_notify(trade_record)`** — 买入成交通知回调
- **`_sell_notify(trade_record)`** — 卖出成交通知回调

---

## 五、对当前 ChanFund Fusion 的改进启示

### 5.1 环境分类器改造

当前方案的问题：
```
MA20 > MA60 → BULL
MA20 < MA60 → BEAR
波动率 > 85% → OSCILLATE
```
- OSCILLATE 仅覆盖 9% 交易日（识别太保守）
- 环境切换有滞后（牛市初期还在 BEAR）

**Hikyuu 式改进方案**：
```python
# 多维度环境分类器
def advanced_environment(self):
    k = self.get_kdata()
    
    # 维度1：趋势方向 (MA交叉 + 斜率)
    ma20 = MA(k, 20)
    ma60 = MA(k, 60)
    ma120 = MA(k, 120)
    
    # 维度2：趋势强度 (ADX)
    adx = ADX(k)
    
    # 维度3：波动率状态 (ATR/价格)
    atr = ATR(k)
    atr_ratio = atr / k.close
    
    # 维度4：量能状态
    vol_ratio = k.volume / MA(k.volume, 20)
    
    # 综合判断
    trend = "BULL" if ma20 > ma60 and ma60 > ma120 else "BEAR"
    trend_strength = "STRONG" if adx > 25 else "WEAK"
    volatility = "HIGH" if atr_ratio > atr_ratio.mean() * 1.2 else "NORMAL"
    
    if trend == "BULL" and trend_strength == "STRONG":
        self.set_env("BULL") 
    elif trend == "BEAR" and trend_strength == "STRONG":
        self.set_env("BEAR")
    elif volatility == "HIGH" or (trend == "BULL" and trend_strength == "WEAK"):
        self.set_env("OSCILLATE")
    else:
        self.set_env(trend)  # 非强趋势但方向明确
```

### 5.2 组件化重构

当前你的回测代码中，TP/SL/MM/环境判断全部硬编码在 `run_v25()` 函数的循环体内，无法独立测试和替换。

**建议：** 将以下逻辑拆分为独立函数/类：

| 当前 | 改为 |
|:---|:----|
| `env_cls.update()` + `env_mult` | `EnvironmentClassifier.evaluate(date) → str` |
| 止损检查（~60行） | `StoplossManager.get_exit_signal(pos, market_data) → str` |
| 止盈检查（~50行） | `TakeprofitManager.check_tp1/tp2(pos, price, env) → (bool, pct)` |
| 仓位计算（~15行） | `PositionSizer.calculate(cash, price, score, env) → (shares, cost)` |

### 5.3 pysystemtrade 框架设计理念 (Rob Carver)

（本部分将从 pysystemtrade 补充，详见 `pysystemtrade_research.md`）

核心理念：
1. **波动率标准化** — 仓位大小与波动率成反比，而非固定金额
2. **相关性管理** — 多标的持仓之间的相关性影响总风险
3. **Forecast 缩放** — 信号强度映射到目标风险暴露

---

## 六、可迁移方案（v3.0 候选）

### 方案 A：阈值优化版（低风险）

仅改环境分类器的参数和逻辑，不改整体架构：
- MA20/60 + MA120 三层过滤器（BULL: MA20>MA60>MA120）
- ADX 强度过滤（ADX<20 → OSCILLATE）
- ATR 波动率保护（ATR/close > 15% → OSCILLATE）

### 方案 B：组件化重构版（中等风险）

重构回测引擎，保持 R1 的 TP/SL 参数不变：
- 提取 EnvironmentClassifier 为独立类
- 提取 StoplossManager（固定止损 + 环境止损 + 尾随止损）
- 提取 TakeprofitManager（TP1 + TP2）
- 提取 PositionSizer（环境乘数 + 共振 + 基础仓位）

### 方案 C：波动率标准化版（高风险，需学习 pysystemtrade 后）

引入波动率标准化仓位管理：
- 仓位 = 目标风险 / (仓位波动率 × 价格)
- 环境判断仅用于 TP/SL，不直接影响仓位大小
- 需要先完成 pysystemtrade 研究

---

## 七、总结

Hikyuu 框架的组件化设计理念——将环境判断、止损、止盈、资金管理全部作为可独立替换的模块——是当前 ChanFund Fusion 最需要借鉴的改进方向。R1 的夏普 0.19 说明 TP/SL 的参数是合理的，但环境分类器是当前最薄弱的一环。

**最优先迁移：** 环境分类器改造（方案 A 或 B）
**同步学习：** pysystemtrade 的波动率标准化仓位管理
