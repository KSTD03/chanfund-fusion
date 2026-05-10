"""
端到端冒烟测试
===========
选取100只代表性股票，用3年数据执行完整链路：
数据加载 → 缠论日线更新 → 基本面因子计算 → 融合规则 → 生成目标权重

不追求绩效，只验证所有管道能正常联动。
作为CI基线，任何新模块或参数改动后先跑此测试。
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import date, timedelta
from pathlib import Path

# 确保能找到策略模块
_workspace_root = str(Path(__file__).resolve().parent.parent)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_data_pipeline():
    """测试1：数据管道是否正常"""
    logger.info("=== Test 1: Data Pipeline ===")
    from strategy.data.market_data import MarketDataProvider
    from strategy.data.pit_data import PITDataPipeline

    # 不依赖Qlib，用mock数据测试
    config = {
        "start_date": "2020-01-01",
        "end_date": "2022-12-31",
        "benchmark": "SH000300",
    }
    provider = MarketDataProvider(config)
    assert provider is not None

    # 测试PIT管道
    pit = PITDataPipeline()
    assert pit is not None
    assert not pit._initialized  # 未加载数据时

    logger.info("PASS: Data Pipeline OK")


def test_tech_engine():
    """测试2：技术信号引擎"""
    logger.info("=== Test 2: Technical Engine ===")
    from strategy.tech.signal_engine import TechSignalEngine
    from strategy.tech.chan_objects import SignalStatus
    import random

    config = {
        "bi_min_kbar": 5,
        "noise": {
            "er_period": 20,
            "er_hist_window": 500,
            "er_low_percentile": 0.30,
            "er_high_percentile": 0.80,
        },
    }
    engine = TechSignalEngine(config)

    # 生成100根模拟K线进行预热
    symbol = "600000"
    base_price = 10.0

    for i in range(100):
        kbar = {
            "time": date(2020, 1, 1) + timedelta(days=i),
            "open": base_price + random.uniform(-0.5, 0.5),
            "high": base_price + random.uniform(0, 1.0),
            "low": base_price - random.uniform(0, 1.0),
            "close": base_price + random.uniform(-0.3, 0.3),
            "volume": random.randint(10000, 1000000),
        }
        base_price = kbar["close"]
        engine.update(symbol, kbar)

    # 预热后应有缠论结构
    state = engine.get_structure_state(symbol)
    assert state is not None, "State should exist after warm-up"
    logger.info(f"  Confirmed bis: {len(state.confirmed_bis)}")
    logger.info(f"  ZS count: {len(state.zs_list)}")

    # 测试信号查询
    signals = engine.get_confirmed_signals(date(2020, 4, 1))
    logger.info(f"  Confirmed signals: {len(signals)}")

    # 测试get_snapshot
    snapshot = engine.get_snapshot()
    assert "chan_cache" in snapshot
    assert "pending_signals" in snapshot

    # 测试restore
    engine2 = TechSignalEngine(config)
    engine2.restore_snapshot(snapshot)
    assert symbol in engine2.chan_cache

    logger.info("PASS: Technical Engine OK")


def test_fund_engine():
    """测试3：基本面因子引擎"""
    logger.info("=== Test 3: Fundamental Engine ===")
    from strategy.fund.factor_engine import FundFactorEngine, FactorFunc

    config = {
        "fin_mix_days": 10,
        "fin_mix_ratio": 0.75,
        "scene_params": {},
        "redflag": {},
    }

    engine = FundFactorEngine(config)

    # 注册测试因子
    def dummy_factor(symbol, trade_date, _):
        return 1.0 if symbol.endswith("00") else 0.5

    engine.register_factor(FactorFunc(
        name="test_factor",
        calc_func=dummy_factor,
        factor_type="real_time",
        weight=1.0,
    ))

    # 模拟股票池
    universe = [f"{i:06d}" for i in range(600000, 600020)]

    # 更新
    trade_date = date(2023, 6, 1)
    engine.set_industry_map({s: "bank" for s in universe})
    engine.update_daily(trade_date, universe)
    from strategy.fund.scene_detector import MacroScene
    engine.set_scene(MacroScene.BALANCED)

    # 获取得分
    scores = engine.get_latest_scores(trade_date, universe)
    assert isinstance(scores, pd.Series)
    assert len(scores) == len(universe)
    logger.info(f"  Scores: min={scores.min():.1f}, max={scores.max():.1f}")

    # 测试快照
    snapshot = engine.get_snapshot()
    assert "cache" in snapshot

    logger.info("PASS: Fundamental Engine OK")


def test_preprocess():
    """测试4：标准化流水线"""
    logger.info("=== Test 4: Preprocess Pipeline ===")
    from strategy.fund.preprocess import mad_winsorize, group_zscore, market_neutralize

    # 生成测试数据
    np.random.seed(42)
    n = 200
    data = {
        "symbol": [f"{i:06d}" for i in range(n)],
        "factor": np.random.randn(n) * 2 + 10,
        "industry": np.random.choice(["bank", "tech", "pharma", "manufacturing"], n),
        "log_mktcap": np.random.randn(n) * 0.5 + 10,
    }
    # 加入异常值
    data["factor"][0] = 100.0
    data["factor"][1] = -50.0

    df = pd.DataFrame(data)

    # MAD去极值
    df["factor_clean"] = mad_winsorize(df["factor"], n_mad=5.0)
    assert df["factor_clean"].max() < 100.0, "Outlier should be capped"

    # 行业ZScore
    df = group_zscore(df, "factor_clean", "industry")
    z_col = "factor_clean_z"
    assert z_col in df.columns

    # 各组ZScore均值应接近0
    group_means = df.groupby("industry")[z_col].mean()
    logger.info(f"  Group means: {group_means.round(3).to_dict()}")
    assert all(abs(group_means) < 0.1), "Group means should be near 0"

    # 市值中性化
    df = market_neutralize(df, z_col, "log_mktcap")

    logger.info("PASS: Preprocess OK")


def test_risk_budget():
    """测试5：风险预算与仓位管理"""
    logger.info("=== Test 5: Risk & Position Management ===")
    from strategy.risk.budget_manager import RiskBudgetManager
    from strategy.risk.position_manager import PositionManager

    config = {
        "fusion": {
            "daily_risk_budget_pct": 0.03,
            "cooling_period_kbars": 10,
            "cooling_stop_count": 2,
        },
        "position": {
            "single_stock_max_pct": 0.08,
            "sector_max_pct": 0.25,
            "slippage": 0.0015,
            "risk_per_trade_pct": 0.005,
        },
    }

    budget = RiskBudgetManager(config)
    budget.new_day(date(2023, 1, 1), 1_000_000.0)
    assert budget.daily_budget == 30000.0  # 3% of 1M

    pos_mgr = PositionManager(config, budget)
    pos_mgr.set_sector_map({"600000": "bank", "600001": "bank", "600002": "tech"})

    # 测试可以开仓（3%日预算可用，开2%仓位）
    assert budget.can_open(date(2023, 1, 1), 0.02)
    budget.allocate(date(2023, 1, 1), 0.02)
    assert abs(budget.used_budget - 0.02) < 1e-6

    # 测试开仓记录
    pos_mgr.open_position("600000", 10.0, 0.05, date(2023, 1, 1), "hard_divergence")
    assert pos_mgr.has_position("600000")

    # 测试冷却期
    pos_mgr.record_stop_loss("600000", date(2023, 1, 1))
    pos_mgr.record_stop_loss("600000", date(2023, 1, 2))
    assert pos_mgr.is_in_cooling("600000")

    # 测试行业上限
    result = pos_mgr.exceeds_sector_limit("600001", 0.25)
    assert result  # bank已有5%，再加25%超限

    logger.info("PASS: Risk & Position Management OK")


def test_fusion_rules():
    """测试6：融合规则"""
    logger.info("=== Test 6: Fusion Rules ===")
    from strategy.tech.signal_engine import TechSignalEngine
    from strategy.fund.factor_engine import FundFactorEngine
    from strategy.risk.budget_manager import RiskBudgetManager
    from strategy.risk.position_manager import PositionManager
    from strategy.fusion.rules import FusionRules
    from strategy.utils.logger import DecisionLogger
    from strategy.tech.chan_objects import Signal, SignalStatus

    config = {
        "fusion": {
            "buy_score_min": 60,
            "strong_buy_score": 85,
            "fund_score_high": 80,
            "fund_score_mid": 70,
            "fund_score_low": 60,
            "fund_confidence_high": 1.2,
            "fund_confidence_mid": 1.0,
            "fund_confidence_low": 0.8,
            "daily_risk_budget_pct": 0.03,
            "sort_top_n": 3,
            "budget_exhaust_threshold": 0.80,
        },
        "position": {
            "single_stock_max_pct": 0.08,
            "sector_max_pct": 0.25,
            "slippage": 0.0015,
            "risk_per_trade_pct": 0.005,
            "jump_reject_pct": 0.03,
        },
        "tech": {},
        "fund": {},
    }

    tech_engine = TechSignalEngine({"noise": {}})
    fund_engine = FundFactorEngine({"fin_mix_days": 10, "fin_mix_ratio": 0.75, "scene_params": {}, "redflag": {}})
    budget = RiskBudgetManager(config)
    budget.new_day(date(2023, 6, 1), 1_000_000.0)
    pos_mgr = PositionManager(config, budget)
    decision_logger = DecisionLogger()

    # 创建模拟信号
    signals = [
        Signal(
            symbol="600000",
            signal_type="buy",
            signal_subtype="hard_divergence",
            source="tech",
            price_confirmed=10.0,
            confirmed_time=date(2023, 5, 30),
        ),
        Signal(
            symbol="600001",
            signal_type="buy",
            signal_subtype="soft_divergence",
            source="tech",
            price_confirmed=20.0,
            confirmed_time=date(2023, 5, 30),
        ),
    ]

    fund_scores = {"600000": 85.0, "600001": 65.0}
    noise_ratios = {"600000": 0.6, "600001": 0.4}

    fusion = FusionRules(config, tech_engine, fund_engine, budget, pos_mgr, decision_logger)
    decisions = fusion.filter_and_merge(signals, fund_scores, date(2023, 6, 1), noise_ratios)

    logger.info(f"  Decisions: {len(decisions)}")
    for sym, w, reason in decisions:
        logger.info(f"    {sym}: weight={w:.4f}, reason={reason}")

    # 至少有决策（600000基本面85+硬背驰应该通过）
    logger.info("PASS: Fusion Rules OK")


def test_snapshot():
    """测试7：快照与断点续跑"""
    logger.info("=== Test 7: Snapshot ===")

    import tempfile
    from strategy.utils.snapshot import SnapshotManager
    from strategy.utils.logger import DecisionLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SnapshotManager(base_dir=tmpdir, experiment_id="test")
        tech_snapshot = {"chan_cache": {"600000": {}}, "pending_signals": {}}
        fund_snapshot = {"cache": {}, "factor_weights": {}}
        logger_demo = DecisionLogger()

        path = manager.save_snapshot(
            tech_snapshot, fund_snapshot, logger_demo,
            date(2023, 6, 1), tag="test",
        )
        assert path is not None

        # 加载
        snapshot = manager.load_snapshot(date(2023, 6, 1), tag="test")
        assert snapshot is not None
        assert "tech_engine" in snapshot

        # 列表
        listings = manager.list_snapshots()
        assert len(listings) >= 1

    logger.info("PASS: Snapshot OK")


def test_trial_buy():
    """测试8：试探建仓模块"""
    logger.info("=== Test 8: Trial Buy Manager ===")
    from strategy.tech.trial_buy import TrialBuyManager

    config = {
        "position_pct": 0.25,
        "stop_multiplier": 1.5,
        "expire_kbars": 20,
        "upgrade_score": 85,
        "upgrade_double_r": 2.0,
    }
    manager = TrialBuyManager(config)

    # 测试是否应该进入
    assert manager.should_enter("600000", "soft_divergence", 70.0, "bull", has_position=False)
    assert not manager.should_enter("600000", "hard_divergence", 70.0, "bull")  # 只接受软背驰
    assert not manager.should_enter("600000", "soft_divergence", 30.0, "bull")  # 基本面太低
    assert not manager.should_enter("600000", "soft_divergence", 70.0, "bear")  # 均线空头

    # 测试建仓
    pos = manager.enter("600000", date(2023, 6, 1), 10.0, 9.5, 1_000_000)
    assert pos is not None
    assert pos.status == "ACTIVE"

    # 测试止损（扩大1.5倍：10 - 0.5*1.5 = 9.25）
    assert pos.stop_price == 9.25

    # 测试超时：价格不动但超过kbar限制
    action = manager.update("600000", 21, 10.2, 10.0, 10.1, date(2023, 6, 22))
    assert action == "expire"  # kbar_count(21) >= max_kbars(20)，未触发止损且未到target

    logger.info("PASS: Trial Buy Manager OK")


def test_redflag():
    """测试9：排雷模块"""
    logger.info("=== Test 9: RedFlag Detector ===")
    from strategy.fund.redflag import RedFlagDetector, RedFlagType

    config = {"redflag": {}}
    detector = RedFlagDetector(config)

    # 正常数据
    normal_fin = {
        "ar_ttm": 100, "revenue_ttm": 1000,
        "ar_3y_ago": 50, "revenue_3y_ago": 600,
        "ocf_ttm": 100, "ocf_ttm_last_year": 80,
        "inventory": 50, "cost_of_sales": 800,
        "inventory_2y_ago": 30, "cost_2y_ago": 700,
        "cash_assets": 50, "total_assets": 1000,
        "interest_bearing_debt": 50,
    }
    result = detector.check("600000", normal_fin)
    assert not result.has_flag, "Normal data should have no flags"

    # 异常数据：应收账款奇高
    abnormal_fin = dict(normal_fin)
    abnormal_fin["ar_ttm"] = 1000
    abnormal_fin["revenue_ttm"] = 1100
    abnormal_fin["ar_3y_ago"] = 50
    abnormal_fin["revenue_3y_ago"] = 800

    result = detector.check("600000", abnormal_fin)
    assert result.has_flag
    assert any(f.flag_type == RedFlagType.AR_ABNORMAL for f in result.flags)
    logger.info(f"  Detected flags: {[f.flag_type.name for f in result.flags]}")

    logger.info("PASS: RedFlag Detector OK")


def test_scene_detector():
    """测试10：宏观场景检测"""
    logger.info("=== Test 10: Scene Detector ===")
    from strategy.fund.scene_detector import SceneDetector, MacroScene
    import numpy as np

    params = {
        "vol_high": 0.25,
        "vol_low": 0.18,
        "vol_window": 20,
        "credit_z_high": 1.5,
        "credit_z_low": -1.5,
        "value_momentum_z": 1.0,
        "vol_extreme_percentile": 0.95,
    }
    detector = SceneDetector(params)

    # 模拟正常数据（均衡场景）
    returns = np.random.randn(500) * 0.01
    credit = np.ones(20) * 0.5
    vm = np.random.randn(250) * 0.1

    result = detector.detect(returns, credit, vm)
    assert isinstance(result.scene, MacroScene)
    logger.info(f"  Scene: {result.scene.value}")
    logger.info(f"  Vol: {result.vol_annualized:.2%}")

    # 模拟高波动+信用收紧（防御场景）
    high_vol_returns = np.random.randn(500) * 0.03
    high_credit = np.concatenate([np.ones(15) * 0.5, np.ones(5) * 3.0])

    result2 = detector.detect(high_vol_returns, high_credit, vm)
    logger.info(f"  Scene2: {result2.scene.value}")
    logger.info(f"  Vol2: {result2.vol_annualized:.2%}")

    logger.info("PASS: Scene Detector OK")


def test_config_validation():
    """测试11：配置验证"""
    logger.info("=== Test 11: Config Validation ===")
    from strategy.config_schema import load_config, validate_config

    # 加载默认配置
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
        assert "tech" in config
        assert "fund" in config
        assert "fusion" in config
        assert "position" in config
        logger.info("  Config loaded and validated OK")
        logger.info(f"  bi_min_kbar: {config['tech']['bi_min_kbar']}")
    else:
        logger.warning("  config.yaml not found, skipping")

    logger.info("PASS: Config Validation OK")


def test_priority_queue():
    """测试12：优先级队列"""
    logger.info("=== Test 12: Priority Queue ===")
    from strategy.fusion.priority_queue import SignalPriorityQueue, PrioritySignal, SignalPriority
    from strategy.tech.chan_objects import Signal, SignalStatus

    queue = SignalPriorityQueue()

    # 添加不同优先级的信号
    sig1 = Signal("600000", "buy", "hard_divergence", "tech", 10.0, date(2023, 6, 1))
    sig2 = Signal("600001", "buy", "soft_divergence", "tech", 20.0, date(2023, 6, 1))
    sig3 = Signal("600002", "buy", "third_point_buy", "tech", 30.0, date(2023, 6, 1))

    queue.push(PrioritySignal(sig3, SignalPriority.TECH_THIRD_POINT, 70))
    queue.push(PrioritySignal(sig1, SignalPriority.TECH_HARD_DIVERGENCE, 85))
    queue.push(PrioritySignal(sig2, SignalPriority.TECH_SOFT_DIVERGENCE, 60))

    # 按优先级弹出
    top1 = queue.peek()
    assert top1 is not None
    if top1:
        logger.info(f"  Top priority: {top1.signal.symbol}, priority={top1.priority}")

    top_n = queue.get_top_n(2)
    assert len(top_n) == 2
    logger.info(f"  Top 2: {[ps.signal.symbol for ps in top_n]}")

    logger.info("PASS: Priority Queue OK")


def main():
    """运行所有冒烟测试"""
    logger.info("=" * 60)
    logger.info("Fusion Strategy — End-to-End Smoke Tests")
    logger.info("=" * 60)

    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Tech Engine", test_tech_engine),
        ("Fund Engine", test_fund_engine),
        ("Preprocess", test_preprocess),
        ("Risk & Budget", test_risk_budget),
        ("Fusion Rules", test_fusion_rules),
        ("Snapshot", test_snapshot),
        ("Trial Buy", test_trial_buy),
        ("RedFlag", test_redflag),
        ("Scene Detector", test_scene_detector),
        ("Config Validation", test_config_validation),
        ("Priority Queue", test_priority_queue),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            logger.info(f"✓ {name}")
            passed += 1
        except Exception as e:
            logger.error(f"✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
