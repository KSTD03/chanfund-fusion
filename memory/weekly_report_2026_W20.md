# 本周工作总结（2026年第20周 | 2026-05-11 ~ 2026-05-16）

> 生成时间：2026-05-16 11:39（周六中午）
> 特别说明：本周核心工作集中在周一(5/11)~周二(5/12)，后续为复盘整理与决策等待期。

---

## ✅ 本周完成事项

### 1. ChanFund Fusion 密集迭代（核心成果）

本周共执行 **12个版本的回测与优化**，含3条独立路径探索：

#### 版本谱系（W20 完整版）

```
v2.5 scores修复 → R1(0.19⭐) → R2(0.065) → R3(-0.233)
  → v2.6(-0.60❌) → v3.0(0.067) → v3.1(0.268⭐定版)
  → v3.2(0.160) → v4.0(0.046真实成本) → v4.1(-0.053)
  → v5-A(0.302⚠️) → v5-C(0.115⭐推荐) → v5-B(-0.449❌关闭)
  → v6-A(-0.029待优化)
```

#### 里程碑版本

| 版本 | 夏普 | 总收益 | 最大回撤 | 核心改动 | 判定 |
|:----|:---:|:-----:|:--------:|---------|:---:|
| **v3.1** | **0.268** | +24.42% | -10.44% | CSI300 + 三层分类器(MA三线+ADX+ATR) + 波动率标准化仓位(Carver) | ⭐ **定版** — 无滑点理想环境最优 |
| **v5-C** | **0.115** | +44.55% | -20.78% | v4.0实成本 + BEAR环境空仓 | ⭐ **推荐** — 含真实成本最优 |
| v5-A | 0.302⚠️ | +74.29% | -33.08% | min_hold=3 + 拥挤度过滤 | 夏普最高但回撤过大，不推荐 |

### 2. scores.parquet 评分模型修复与评估

- **Bug修复**：`load_scores_parquet()` 中 Tushare 代码（`000001.SZ`）未清理后缀即加前缀，导致 Qlib 格式错误
  - 修复前：`used=328` 是假象（实际全用默认50分）
  - 修复后：`used=302, default=0, rejected=59, passed=243` ✅
- **真实评分接入**后夏普 **0.062**，反而不如伪分50的0.11
- **根因分析**：被减仓的 low-score 交易仍有57-62%胜率（正期望），减仓是削正期望仓位

### 3. 三条长期任务执行 + 两条路径正式关闭

| 路径 | 状态 | 结论 | 证据 |
|:----|:---:|:----:|:----:|
| 基本面评分重构(LightGBM) | ❌ **正式关闭** | 高分组avg 1.20% vs 低分组1.55%，差-0.35%，p=0.60 | 146K样本，27K有效标签，无预测能力 |
| 30分钟频切换验证 | ❌ **正式关闭** | 胜率最高53.5%，盈亏比≤1.15，信号密度↑26x但质量不足 | 三种TP/SL配置全部不达标 |
| czsc缠论引擎独立验证 | ✅ **确认有效** | 夏普**0.21**，缠论信源有预测能力 | 独立环境复现，与ChanFund框架互相验证 |
| BI力度过滤(v6-A) | ⏳ **待优化** | 阈值1.6%通过率太严，需调参 | 初版夏普-0.029 |

### 4. 外部研究 + 知识库构建

| 项目 | 产出 |
|:----|:-----|
| Hikyuu 源码研究 | `memory/hikyuu_research.md` — 组件化架构分析 + 环境分类器改进方案 |
| pysystemtrade 研究 | `memory/pysystemtrade_research.md` — Carver 波动率标准化仓位迁移方案 |
| v3.0 设计稿 | `memory/v30_design.md` — 三层分类器设计（MA三线+ADX+ATR） |
| 缠论引擎对比分析 | `memory/czsc_vs_our_engine.md` + `chanlun_pro_architecture.md` + `chanpy_multilevel_summary.md` + `recommended_transplant.md` |
| 学术材料阅读 | LdP(Ch3/4/11-14)、G-K-X论文、Ernest Chan博客、PandaFactor源码、vnpy CTA模块 |
| 知识库入库 | `knowledge_base/` 新增6份参考文件 + `skills/` 新增3份技能笔记 |
| v3.1 研究报告 | `v31_report.md` — 完整方法论+核心数据+实盘建议 |
| GitHub 提交 | `chanfund-fusion` 仓库 47文件推送成功 |

### 5. 系统维护

- **iKuuu VPN** 安装配置（deb安装 + Mihomo内核 + systemd + ufw加固）
- **滑点默认配置更新**：取消（SLIPPAGE_BUY=1.0, SLIPPAGE_SELL=1.0），因实盘资金量级小
- **Matt Pocock Skills** 安装到 Claude Code

---

## 📌 下周待办

优先级从高到低：

1. **高** — 方向A(BI力度过滤)参数调优
   - 当前阈值1.6%通过率太严，降低至3-5%重测
   - 预期夏普0.10~0.15（含滑点）

2. **高** — 方向B(环境+板块共向性)
   - 老板已批准但未执行
   - 需设计板块共向性指标 + 环境加权融合

3. **中** — v3.1 理想环境版本实盘适配讨论
   - 当前最佳定版（夏普0.268），但无滑点
   - 需评估实盘可行性

4. **中** — 组件化重构（v3.0-c）
   - 将R1中硬编码逻辑拆分为独立组件（环境判断/止损/止盈/资金管理）
   - 参考 Hikyuu 设计模式

5. **低** — scores.parquet 评分模型重训
   - 目标函数改为预测缠论信号后收益
   - 等待方向A/B有结果后再决定资源分配

---

## 📊 ChanFund Fusion 重点项目进展

| 里程碑 | 状态 | 进度 | 说明 |
|--------|:----:|:----:|------|
| v2.5 scores接入修复 | ✅ 完成 | 100% | Bug修复+验证通过 |
| 基本面评分重构(LightGBM) | ❌ 已关闭 | 100% | 无预测能力，正式关闭 |
| 30分钟频切换验证 | ❌ 已关闭 | 100% | 信号密度高但质量不足，正式关闭 |
| czsc缠论引擎验证 | ✅ 完成 | 100% | 缠论信源有效(夏普0.21) |
| v3.1 定版 | ✅ 完成 | 100% | 夏普0.268，理想环境最优 |
| v5-C 推荐版本 | ✅ 完成 | 100% | 夏普0.115，含真实成本最优 |
| 方向A BI力度过滤调参 | 🔄 待执行 | 50% | 初版已跑，阈值需放宽 |
| 方向B 环境+板块共向性 | 📥 待启动 | 0% | 设计已完成，待执行 |
| 组件化重构(v3.0-c) | 📥 待启动 | 0% | 设计方案已出 |
| GitHub 仓库发布 | ✅ 完成 | 100% | 47文件推送，含README+版本谱系 |

---

## 📚 本周入库知识点概览

| 知识点 | 位置 | 类型 |
|--------|------|------|
| López de Prado 读书记录(Ch3/4/11-14) | `knowledge_base/reference_lopez_de_prado_2018.md` | 学术参考 |
| Gu-Kelly-Xiu 2020 论文摘要 | `knowledge_base/reference_gu_kelly_xiu_2020.md` | 学术参考 |
| Ernest Chan 回测系统读书记录 | `knowledge_base/reference_ernest_chan.md` | 学术参考 |
| PandaFactor 工具分析 | `knowledge_base/tool_pandafactor.md` | 工具评估 |
| vnpy CTA 模块分析 | `knowledge_base/tool_vnpy_cta.md` | 工具评估 |
| 缠论引擎对比(czsc/chanlun-pro/chan.py/ZenPlot) | `knowledge_base/reference_chanlun_engines.md` | 技术评估 |
| 三重屏障标签+样本权重 | `skills/triple_barrier_labeling.md` | 方法论 |
| Deflated Sharpe Ratio | `skills/deflated_sharpe_ratio.md` | 方法论 |
| 版本谱系/经验教训/已关闭路径 | `knowledge_base/version_history.md` 等 | 策略沉淀 |
| czsc 验证结论 | `knowledge_base/czsc_validation_conclusion.md` | 验证报告 |
| Hikyuu 源码分析 | `memory/hikyuu_research.md` | 技术研究 |
| pysystemtrade 研究 | `memory/pysystemtrade_research.md` | 技术研究 |

> **⚠️ 风险提示：** 以上所有回测结果均为历史数据模拟，不代表未来收益。v3.1夏普0.268为无滑点理想环境，实盘需考虑交易成本。v5-C夏普0.115虽含滑点，但回撤-20.78%需风控配合。多条路径已正式关闭，剩余优化空间有限，建议聚焦方向A/B。
