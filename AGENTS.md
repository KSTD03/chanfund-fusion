# 技能自动选择规则

> 基于当前已安装的 34 个就绪技能，定义以下自动选择规则。
> 当用户输入指令时，优先匹配最具体的分类，回退到通用技能。

---

## 一、数据获取类

| 触发关键词 | 首选技能 | 备选 | 说明 |
|-----------|---------|------|------|
| A股/沪深/上证/深证/个股日线/K线/行情 | `akshare-cn-market` | `tushare-finance` | AKShare 覆盖A股全量数据，Tushare 用于财报深度 |
| 港股/美股/国际行情 | `stock-info-explorer` | — | 基于 yfinance，适合非A股市场 |
| 个股基本面/财报/ROE/PE | `tushare-finance` | `akshare-cn-market` | Tushare 财报数据最全 |
| 宏观经济/GDP/CPI/PMI/M2 | `akshare-cn-market` | — | 内置宏观数据接口 |
| 自选股/关注列表/持仓汇总 | `stock-watcher` | — | 管理用户的自选股池 |
| 金融计算/复利/折现/贷款 | `financial-calculator` | — | 独立的财务计算器 |

## 二、策略与量化类

| 触发关键词 | 首选技能 | 备选 | 说明 |
|-----------|---------|------|------|
| 策略回测/历史回测/绩效分析 | `backtesting-trading-strategies` | — | 通用回测框架 |
| 因子分析/IC/IR/因子组合 | `quant-research-platform` | — | 100+ alpha 因子 + 多因子分析 |
| 投资组合优化/风险平价/最大夏普 | `quant-research-platform` | — | 组合优化模块 |
| 技术分析/缠论/RSI/MACD/支撑阻力 | `technical-analysis` | — | 含Wyckoff/Dow/缠论 |
| 量化研究平台/多因子 | `quant-research-platform` | — | 高级量化研究 |

## 三、风险与监控类

| 触发关键词 | 首选技能 | 备选 | 说明 |
|-----------|---------|------|------|
| 服务器监控/磁盘/内存/CPU | `system-vigil` | `healthcheck` | 系统健康监控 |
| 安全检查/SSH/防火墙/更新 | `healthcheck` | — | OpenClaw 主机安全审计 |
| 依赖漏洞/安全审计/扫描 | `dep-audit` | — | pip/npm/Cargo/Go 依赖审计 |
| 代码审查/PR/代码质量 | `code-review` | — | 安全+性能+可维护性检查 |
| 技能安全审查 | `skill-vetter` | — | 仅限技能本身的安全性 |

## 四、编码与文档类

| 触发关键词 | 首选技能 | 备选 | 说明 |
|-----------|---------|------|------|
| 代码生成/脚手架/CRUD | `code-generator` | — | 多语言代码生成 |
| 文档生成/README/API文档 | `docs-generator` | — | 自动文档生成 |
| Git/提交/分支/合并 | `git-essentials` | `git-workflows` | 日常+高级Git操作 |
| 代码审查/重构建议 | `code-review` | — | 系统性代码审查 |
| Claude Code/编码助手 | `claude-code` | — | AI辅助编码 |
| 文本摘要/翻译/要点提炼 | `summarize-pro` | — | 20种摘要格式 |

## 五、知识与自动化类

| 触发关键词 | 首选技能 | 备选 | 说明 |
|-----------|---------|------|------|
| 知识库/入库/归档/笔记 | `knowledge` | — | 本地知识库集成 |
| 多步任务/工作流/编排 | `taskflow` | — | 协调多步任务 |
| 浏览器/网页操作/登录 | `browser-automation` | — | 浏览器自动化 |
| 学习改进/历史错误/总结 | `self-improvement` | — | 持续学习 |
| 天气/天气预报 | `weather` | — | 天气查询 |
| 终端/TTY/交互 | `tmux` | — | 终端管理 |

## 六、数据源优先级规则

```
A股行情/财务数据:  akshare-cn-market（首选）> tushare-finance（备选）
美股/全球行情:     stock-info-explorer（yfinance）
策略回测:          backtesting-trading-strategies（回测引擎）
量化研究:          quant-research-platform（因子+组合+回测）
```

## 七、覆盖边界

以下情况不触发自动选择，需人工指定：
- 需要自定义股票池/参数的回测
- 自定义因子表达式编写
- 策略部署到实盘环境
- 需要修改系统配置的操作
