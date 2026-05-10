#!/bin/bash
# ============================================================
# ChanFund Fusion v2.2 优化冲刺 — 流水线主控脚本
# 执行范围：数据补全 → 基线验证 → 环境乘数调优 → 
#           因子门槛松绑 → 盈亏比修复 → 全量验证 → 工程收尾
# 
# 用法：
#   cd /home/quant/.openclaw/workspace
#   nohup bash scripts/run_full_optimization_pipeline.sh > scripts/pipeline_cron.log 2>&1 &
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG="$SCRIPT_DIR/pipeline_cron.log"

log() { echo "[$(date '+%H:%M:%S')] $1"; echo "[$(date '+%H:%M:%S')] $1" >> "$LOG"; }
log_sep() { log "========================================"; log " $1"; log "========================================"; }
fail() { log "❌ 步骤失败: $1 (退出码=$2)"; exit "$2"; }

cd "$WORKSPACE"
log_sep "ChanFund Fusion v2.2 优化冲刺启动"
log "工作目录: $WORKSPACE"

# Step 1: 财务数据补全
log_sep "Step 1: 财务数据补全 (Tushare 全量)"
if [ ! -f "financial_data/scores.parquet" ]; then
    log "📥 开始下载财务数据..."
    python3 scripts/fix_and_run_financial_pipeline.py --resume || {
        ret=$?; log "⚠️ Step 1 返回码=$ret, 尝试续传..."
        python3 scripts/fix_and_run_financial_pipeline.py --resume || fail "Step 1 (续传也失败)" $?
    }
else
    log "✅ scores.parquet 已存在，跳过"
fi

# Step 2: v2.1-datafull 基线回测
log_sep "Step 2: v2.1-datafull 基线回测"
if [ ! -f "backtest_results/v2.1_datafull_summary.md" ]; then
    log "🚀 启动 v2.1-datafull 基线回测..."
    python3 run_chanfund_v20_backtest.py --config chanfund_fusion/config_v2.1_datafull.yaml --batched || fail "Step 2" $?
else
    log "✅ v2.1 基线回测报告已存在，跳过"
fi

# Step 3: 自动对比 v2.0 vs v2.1
log_sep "Step 3: v2.0 vs v2.1 对比"
mkdir -p comparison
python3 scripts/compare_versions.py --baseline backtest_results/v2.0_full_summary.md --new backtest_results/v2.1_datafull_summary.md || log "⚠️ 对比执行（部分文件可能缺失）"

# Step 4: 环境乘数网格搜索
log_sep "Step 4: 环境乘数网格搜索"
if [ ! -f "optimization/env_multiplier_results.csv" ]; then
    log "🔍 搜索中... (8组 × ~3min = ~24min)"
    python3 scripts/grid_search_env_multipliers.py || fail "Step 4" $?
else
    log "✅ env_multiplier_results.csv 已存在，跳过"
fi

# Step 5: 因子门槛扫描
log_sep "Step 5: 因子门槛网格搜索"
if [ ! -f "optimization/factor_threshold_results.csv" ]; then
    log "🔍 搜索中... (6组 × ~3min = ~18min)"
    python3 scripts/grid_search_factor_threshold.py || fail "Step 5" $?
else
    log "✅ factor_threshold_results.csv 已存在，跳过"
fi

# Step 6: 止盈参数调优
log_sep "Step 6: 止盈参数网格搜索"
if [ ! -f "optimization/tp_param_results.csv" ]; then
    log "🔍 搜索中... (16组 × ~3min = ~48min)"
    python3 scripts/grid_search_tp_params.py || fail "Step 6" $?
else
    log "✅ tp_param_results.csv 已存在，跳过"
fi

# Step 7: 最优组合全量回测
log_sep "Step 7: 最优组合全量回测"
python3 scripts/build_best_config.py || fail "Step 7" $?

# Step 8: 最终报告生成
log_sep "Step 8: 最终报告生成"
python3 scripts/generate_final_report.py || log "⚠️ Step 8 执行（部分数据可能缺失）"

log_sep "🎉 v2.2 优化冲刺完成!"
log "唯一需查看文件: output/v2.2_final_report.md"
log "完整内容: output/v2.2_final_report.md"
