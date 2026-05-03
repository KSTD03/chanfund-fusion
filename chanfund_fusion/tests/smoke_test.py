"""
ChanFund Fusion — 冒烟测试
=====================
验证所有核心模块可导入、配置可加载。

运行方式：
    cd /home/quant/.openclaw/workspace
    python3 -m chanfund_fusion.tests.smoke_test
    或
    cd /home/quant/.openclaw/workspace
    python3 chanfund_fusion/tests/smoke_test.py
"""

import os
import sys

# 添加项目根目录（workspace）到路径，保证包级相对导入正常工作
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
_WORKSPACE = os.path.dirname(_PACKAGE_DIR)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)


def test_imports():
    """测试所有模块导入"""
    from chanfund_fusion.config_schema import load_config
    from chanfund_fusion.tech.chan_objects import Signal, SignalStatus, Bi, ZS, FX, FxType
    from chanfund_fusion.tech.chan_structure import ChanStructureState
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.tech.noise import NoiseEstimator, EfficiencyRatio, ActiveStockFilter
    from chanfund_fusion.tech.trial_buy import TrialBuyManager
    from chanfund_fusion.tech.macd_area import calc_macd_area_until_confirmed
    from chanfund_fusion.fund.factor_engine import FundFactorEngine, FactorFunc
    from chanfund_fusion.fund.redflag import RedFlagDetector, RedFlagType, CardLevel
    from chanfund_fusion.fund.preprocess import standardize_pipeline
    from chanfund_fusion.fund.scene_detector import SceneDetector, MacroScene
    from chanfund_fusion.fusion.rules import FusionRules
    from chanfund_fusion.fusion.priority_queue import SignalPriorityQueue, SignalPriority
    from chanfund_fusion.risk.budget_manager import RiskBudgetManager
    from chanfund_fusion.risk.position_manager import PositionManager
    from chanfund_fusion.utils.logger import DecisionLogger, RejectReason
    from chanfund_fusion.utils.snapshot import SnapshotManager
    from chanfund_fusion.utils.cache import SignalCache
    from chanfund_fusion.data.market_data import MarketDataProvider
    from chanfund_fusion.data.pit_data import PITDataPipeline

    print("✅ All modules imported successfully")


def test_config():
    """测试配置加载"""
    from chanfund_fusion.config_schema import load_config

    config_path = os.path.join(_PACKAGE_DIR, "config.yaml")
    config = load_config(config_path)
    assert config["strategy"]["name"] == "ChanFund Fusion"
    assert config["position"]["stop_loss"]["mode"] == "trailing"
    assert config["position"]["risk_per_trade_pct"] == 0.01
    print(f"✅ Config loaded: {config['strategy']['name']} v{config['strategy']['version']}")


def test_new_reject_reasons():
    """测试新拒绝原因枚举"""
    from chanfund_fusion.utils.logger import RejectReason

    reasons = [r.value for r in RejectReason]
    assert "trend_direction" in reasons, "Missing TREND_DIRECTION"
    assert "red_flag" in reasons, "Missing RED_FLAG"
    print(f"✅ RejectReasons: {len(list(RejectReason))} values")


def test_redflag_types():
    """测试排雷模块新类型"""
    from chanfund_fusion.fund.redflag import RedFlagType, CardLevel

    types = list(RedFlagType)
    type_names = [t.name for t in types]
    assert "ST" in type_names
    assert "INVESTIGATION" in type_names
    assert "PLEDGE_HIGH" in type_names
    assert "GOODWILL_HIGH" in type_names
    print(f"✅ RedFlagTypes: {len(types)} types including ST/INVESTIGATION/PLEDGE")


def test_stop_loss_config():
    """测试保护性止损配置"""
    from chanfund_fusion.config_schema import load_config

    config_path = os.path.join(_PACKAGE_DIR, "config.yaml")
    config = load_config(config_path)
    sl = config["position"]["stop_loss"]
    assert sl["trail_bi_low"] is True
    assert sl["trail_zs_zg"] is True
    assert sl["initial_stop_atr_mult"] == 2.0
    print(f"✅ Trailing stop: bi_low={sl['trail_bi_low']}, zs_zg={sl['trail_zs_zg']}")


def test_position_coeff_config():
    """测试分级仓位系数配置"""
    from chanfund_fusion.config_schema import load_config

    config_path = os.path.join(_PACKAGE_DIR, "config.yaml")
    config = load_config(config_path)
    coeff = config["fusion"]["position_coeff"]
    assert coeff["hard_divergence"] == 1.0
    assert coeff["third_point_buy"] == 0.8
    assert coeff["soft_divergence"] == 0.25
    print(f"✅ Position coeff: HD={coeff['hard_divergence']}, "
          f"3B={coeff['third_point_buy']}, Soft={coeff['soft_divergence']}")


def test_direction_filter_config():
    """测试方向过滤器配置"""
    from chanfund_fusion.config_schema import load_config

    config_path = os.path.join(_PACKAGE_DIR, "config.yaml")
    config = load_config(config_path)
    df = config["tech"]["direction_filter"]
    assert df["ema_fast"] == 12
    assert df["ema_slow"] == 26
    assert df["use_long_term_filter"] is True
    print(f"✅ Direction filter: EMA{df['ema_fast']}/{df['ema_slow']}, "
          f"250MA={'on' if df['use_long_term_filter'] else 'off'}")


if __name__ == "__main__":
    print("=" * 50)
    print("ChanFund Fusion — 冒烟测试")
    print("=" * 50)
    test_imports()
    test_config()
    test_new_reject_reasons()
    test_redflag_types()
    test_stop_loss_config()
    test_position_coeff_config()
    test_direction_filter_config()
    print()
    print("🎯 All smoke tests passed!")
