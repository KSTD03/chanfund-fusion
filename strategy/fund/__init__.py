"""
基本面子系统：因子引擎、去极值标准化、宏观场景识别、排雷模块
"""

from .factor_engine import FundFactorEngine
from .preprocess import mad_winsorize, group_zscore, market_neutralize
from .scene_detector import SceneDetector, MacroScene
from .redflag import RedFlagDetector, RedFlagType

__all__ = [
    "FundFactorEngine",
    "mad_winsorize", "group_zscore", "market_neutralize",
    "SceneDetector", "MacroScene",
    "RedFlagDetector", "RedFlagType",
]
