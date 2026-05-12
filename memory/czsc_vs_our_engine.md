# czsc 与 ChanFund Fusion 引擎对比分析

> 分析基准：czsc v0.9.69（Python版） vs ChanFund Fusion chan_structure.py

---

## 一、分型/笔识别算法对比

| 对比项 | czsc（v0.9.69） | ChanFund Fusion（我们） | 差异说明 |
|:------|:--------------|:--------------------|:--------|
| **包含处理** | `remove_include(k1, k2, k3)`：按方向（Up/Down）确定取大/取小高低的合并规则 | `_update_current_bi()` 中类似逻辑 | 实现大同小异，czsc 更规范 |
| **分型检测** | `check_fx(k1,k2,k3)`：顶分型 k1.high < k2.high > k3.high **且** k1.low < k2.low > k3.low | `_detect_fx()`：同样正字判断高低点 | **✅ 完全一致**，仅变量名不同 |
| **分型交替校验** | `check_fxs()` 中强制检查顶底交替，连续相同 mark 会 err log 并丢弃 | pending_fx 自然保证交替 | czsc 更严格（显式错误检测） |
| **成笔条件** | `len(bars_a) >= min_bi_len(默认6)` + 顶底之间**无包含关系** | `self.bi_min_kbar = 5` + 笔方向反向突破 | **czsc要求6根K线，我们是5根**，且czsc多了一个"顶底无包含"条件 |
| **笔破坏后处理** | 最后一笔延伸中被反向突破→丢弃旧笔+合并bars_ubi | `_confirm_bi_after_fx()` 中确认新笔并修正旧笔 | czsc 会丢弃被破坏笔，我们保留并更新 |
| **跳空处理** | czsc 无特殊跳空处理（直接当K线处理） | 我们有单独的跳空分型检测逻辑 | **我们的更完整**，跳空可产生无K线分型 |
| **第一笔初始化** | 奇数笔时用最显著分型（选极值） | 等待完整顶底出现 | czsc 更快进入状态，但可能不稳 |
| **缓存机制** | `RawBar.cache` + `BI.cache`，懒计算属性 | 无系统缓存，属性即时计算 | czsc 性能更好（运行时懒加载） |

### 关键差异详解

**差异1：最小笔长度**
- czsc: `MIN_BI_LEN = 6`
- 我们: `bi_min_kbar = 5`
- 影响：czsc 更严格，会过滤掉部分短笔，我们更激进

**差异2：笔破坏后丢弃 vs 保留**
- czsc: 强势反向突破→丢弃当前笔、合并 K 线、在 bars_ubi 中重算
- 我们: 确认反向笔、修正旧笔 end_fx、保留在中枢计算中
- **这可能是我们信号质量差异的一个来源**——czsc 的重算更符合缠论原文"笔被笔破坏"

**差异3：分型交替校验**
- czsc: `check_fxs()` 中显式校验 `fx.mark != fxs[-1].mark`，违反时 log error 并丢弃
- 我们: 由 `pending_fx` 自然保证，无显式校验

**差异4：多级别 K 线合成**
- czsc: 通过 `BarGenerator` 将低频合成高频，原生支持多级别
- 我们: 没有 BarGenerator，所有分析基于日频

---

## 二、因子/信号/事件数据结构对比

### czsc 的 Signal 对象

```python
@dataclass
class Signal:
    signal: str = ""    # 完整信号字符串，格式: "k1_k2_k3_v1_v2_v3_score"
    score: int = 0      # 0~100 信号强度
    k1: str = "任意"    # K线周期
    k2: str = "任意"    # 参数
    k3: str = "任意"    # 信号分类标识
    v1: str = "任意"    # 信号取值（可任意匹配）
    v2: str = "任意"
    v3: str = "任意"
```

### czsc 的 Factor（因子）对象

```python
@dataclass
class Factor:
    signals_all: List[Signal]    # 必须全部满足
    signals_any: List[Signal]    # 满足任一即可（可空）
    signals_not: List[Signal]    # 都不能满足（可空）
    name: str = ""
    # is_match(s: dict) -> bool  — 判断是否匹配
```

### czsc 的 Event（事件）对象

```python
@dataclass
class Event:
    operate: Operate              # 操作类型（开多/开空/平多/平空）
    factors: List[Factor]         # 任一因子满足即可
    signals_all: List[Signal]     # 全局约束
    signals_any: List[Signal]
    signals_not: List[Signal]
    name: str = ""
    sha256: str = ""              # 自动生成的唯一指纹
```

### 字段映射表

| czsc 字段 | 我们的对应变量 | 备注 |
|:---------|:-----------|:----|
| `Signal.k1`（周期） | 无对应（仅日频） | 我们不需要，可忽略 |
| `Signal.k2`（参数） | `Signal.details` | 可放入 details dict |
| `Signal.k3`（分类） | `Signal.signal_subtype` | 如 "third_point_buy" |
| `Signal.v1/v2/v3` | 无对应 | czsc 的"可任意匹配"设计我们没用 |
| `Signal.score` | `Signal.tech_score` | 我们的 0-100 |
| `Factor.signals_all` | 无对应 | 我们无信号组合逻辑 |
| `Factor.signals_any` | 无对应 | 我们无信号组合逻辑 |
| `Event.operate` | `Signal.signal_type` | "buy"/"sell" |
| `Event sha256` | 无 | czsc 防误用设计 |

**核心差异**：czsc 有完整的 **Signal→Factor→Event** 三级体系，支持信号的逻辑组合（AND/OR/NOT）。我们目前只有一层 Signal 对象，信号是"触发→执行"直通，没有组合层。

---

## 三、可复用代码清单

### czsc 中可以直接参考的函数/类

| 函数/类 | 文件路径（v0.9.69） | 行号 | 复用方式 |
|:-------|:----------------|:----:|:--------|
| `remove_include()` | `czsc/analyze.py` | 26-85 | **参考** — 与我们的包含处理逻辑比较 | 
| `check_fx()` | `czsc/analyze.py` | 87-103 | **直接对齐** — 分型判断逻辑完全一致 |
| `check_fxs()` | `czsc/analyze.py` | 105-125 | **参考** — 顶底交替校验更严格 |
| `check_bi()` | `czsc/analyze.py` | 127-165 | **参考** — 成笔条件：min_bi_len=6 vs 我们5 |
| `Signal` dataclass | `czsc/objects.py` | 199-252 | **数据结构参考** — 信号序列化格式 |
| `Factor` dataclass | `czsc/objects.py` | 254-316 | **可直接移植** — signals_all/any/not 三件套 |
| `Event` dataclass | `czsc/objects.py` | 318-404 | **可直接移植** — 事件多因子组合 |
| `CZSC.update()` | `czsc/analyze.py` | 196-240 | **结构参考** — 增量更新的方法论（与我们的 5 事件完全一致） |
| `CZSC.open_in_browser()` | `czsc/analyze.py` | 267-273 | **工具** — echarts 可视化（低优先级） |
| `CZSC.last_bi_extend` | `czsc/analyze.py` | 276-284 | **参考** — 判断笔是否在延伸中 |

### 不可复用

| 部分 | 原因 |
|:----|:-----|
| czsc 1.0+ Rust 核心（`czsc._native`） | 与 Python 接口不兼容，且混合架构部署复杂 |
| Rust TA 算子（ema/sma/boll） | TA 库我们用 numpy 实现，足够 |
| BarGenerator 多级别 K 线合成 | 我们当前只有日频，若有需求可引入但优先级低 |
