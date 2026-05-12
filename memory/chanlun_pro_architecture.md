# chanlun-pro 回测架构分析

> 分析基准：chanlun-pro v1.x (yijixiuxin/chanlun-pro)

---

## 一、回测引擎类关系图

```
Backtest (backtest.py)
├── 构造：接收 klines, config, trade_data
├── 主入口：run()
│   ├── 初始化交易器 BackTestTrader
│   ├── 按 K 线循环调用 strategy.on_bar()
│   └── 收集结果
│
├── backtest_klines.py          ← K 线数据管理
│   └── Klines: 多标的OHLC管理
│
├── backtest_trader.py          ← 交易模拟器
│   └── BackTestTrader: 订单执行、持仓、资金管理
│
├── base.py                     ← 策略基类
│   └── StrategyBase: 抽象接口
│
├── optimize / 回测参数
│   └── run_optimization: 参数网格搜索
└── 结果输出
    └── trade_records / equity_curve
```

## 二、Strategy 基类接口对比

| chanlun-pro `StrategyBase` | ChanFund Fusion（我们的） | 差异 |
|:-------------------------|:-----------------------|:----|
| `__init__(self, config)` | — | chanlun-pro 在构造时传入配置 |
| `on_bar(self, kline, trade_data)` | `process_kbar(kbar, emit_signals)` | **核心差异**：chanlun-pro 的 on_bar 仅处理一根 K 线，不嵌在回测循环中；我们的 process_kbar 同样逐 K 线处理 |
| `on_order(self, order)` | —（无独立订单回调） | 我们没有订单回调，所有交易在回测循环中同步执行 |
| `on_trade(self, trade)` | —（无独立成交回调） | 我们没有成交回调 |
| **信号/仓位分离** | 信号计算 + 仓位管理都在回测脚本中硬编码 | chanlun-pro 的 Strategy 是独立的可组合单元 |
| **多标的处理** | 一个 Strategy 一个标的 | 我们一个线程管理全部 800 只 |

**核心差异**：chanlun-pro 是 **单标的→组合成组合** 的架构，每个 Strategy 独立跑一个标的，然后由组合层汇总。我们是 **全局循环**，一个循环管理所有标的。两种方式各有优劣。

## 三、BackTestTrader 模拟器功能对比

| 功能 | chanlun-pro | 我们 |
|:----|:----------|:----|
| **滑点** | ✅ 可配置滑点（买入/卖出不同） | ❌ 无滑点，以开盘价成交 |
| **手续费** | ✅ 可配置（佣金+印花税） | ❌ 无手续费 |
| **涨跌停限制** | ✅ 检查涨停不可买、跌停不可卖 | ❌ 无检查 |
| **停牌复牌** | ✅ 停牌期间跳过交易 | ❌ 无停牌处理 |
| **T+1限制** | ✅ 当日买入不可卖出 | ✅ 通过 `buy_exec_date` 模拟 |
| **成交量限制** | ✅ 检查卖一/买一挂单量 | ❌ 无限制，假设无限流动性 |

**共 6 项差异，每一项 chanlun-pro 都更完整。** 这是我们模拟器最需要补的部分。

## 四、参数优化模块评估

- ✅ **方法**：网格搜索（逐一枚举参数组合）
- ❌ **风险**：全量样本内优化，**没有滚窗验证**
- ❌ **风险**：优化结果直接在测试集上评估，没有样本外保留
- **标注**：@zjb → 明显过拟合风险。在我们的架构中，R1/R2/R3/v3.x 的训练/验证/测试三段拆分是更严谨的做法。**不建议模仿 chanlun-pro 的全样本内优化方法。**

## 五、对我们引擎的改进建议

### 优先级 P0（模拟器精度）

1. **滑点**（最简单，加一行 `entry_price * 1.001`）
2. **涨跌停限制**（从数据中读取涨停/跌停价）
3. **成交量限制**（检查日成交量百分比）

### 优先级 P1（策略接口）

参考 chanlun-pro 的 `StrategyBase`，把我们的回测脚本拆分为：
- `Strategy(init, on_kline, on_signal, on_trade)` 四个接口
- 使得策略和回测引擎解耦

### 优先级 P2（参数优化）

- 保留我们的三段拆分（训练/验证/测试）
- 只在验证集上做网格搜索
- 测试集仅用于最终评估
