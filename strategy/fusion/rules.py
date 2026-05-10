"""
融合规则与仲裁 — FusionRules
======================
信号融合过滤器 + 优先级排序 + 仓位计算公式

核心流程：
    1. 技术信号 + 基本面得分 → 构建优先级队列
    2. 按优先级逐次计算目标仓位
    3. 受风险预算约束
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

from ..tech.chan_objects import Signal, SignalStatus
from ..tech.signal_engine import TechSignalEngine
from ..fund.factor_engine import FundFactorEngine
from ..risk.budget_manager import RiskBudgetManager
from ..risk.position_manager import PositionManager
from .priority_queue import SignalPriorityQueue, PrioritySignal, SignalPriority
from ..utils.logger import DecisionLogger, RejectReason

logger = logging.getLogger(__name__)


class FusionRules:
    """融合规则与仲裁

    核心职责：
    1. 技术信号 + 基本面得分 → 融合筛选
    2. 仓位计算
    3. 信号优先级排序
    """

    def __init__(
        self,
        config: dict,
        tech_engine: TechSignalEngine,
        fund_engine: FundFactorEngine,
        budget_manager: RiskBudgetManager,
        position_manager: PositionManager,
        decision_logger: DecisionLogger,
    ):
        self.config = config
        self.fusion_config = config.get("fusion", {})
        self.tech_engine = tech_engine
        self.fund_engine = fund_engine
        self.budget_manager = budget_manager
        self.position_manager = position_manager
        self.logger = decision_logger

        # 融合参数
        self.buy_score_min = self.fusion_config.get("buy_score_min", 60)
        self.strong_buy_score = self.fusion_config.get("strong_buy_score", 85)
        self.sort_top_n = self.fusion_config.get("sort_top_n", 3)
        self.budget_exhaust_threshold = self.fusion_config.get(
            "budget_exhaust_threshold", 0.80
        )

        # 基本面信心调节因子
        self.confidence_high = self.fusion_config.get("fund_confidence_high", 1.2)
        self.confidence_mid = self.fusion_config.get("fund_confidence_mid", 1.0)
        self.confidence_low = self.fusion_config.get("fund_confidence_low", 0.8)

    def filter_and_merge(
        self,
        tech_signals: List[Signal],
        fund_scores: Dict[str, float],
        trade_date,
        noise_ratios: Dict[str, float] = None,
    ) -> List[Tuple[str, float, str]]:
        """融合技术信号与基本面得分

        Args:
            tech_signals: 当日可执行的买入信号列表
            fund_scores: {symbol: score 0-100}
            trade_date: 当前交易日
            noise_ratios: {symbol: noise_level 0-1} (可选)

        Returns:
            decisions: [(symbol, target_weight, reason)]
        """
        if noise_ratios is None:
            noise_ratios = {}

        # 1. 构建优先级队列
        pq = SignalPriorityQueue()

        for sig in tech_signals:
            symbol = sig.symbol
            fund_score = fund_scores.get(symbol, 0.0)

            # --- 过滤 ---

            # 检查噪声
            noise = noise_ratios.get(symbol, 0.5)
            if noise < 0.15:  # 过高的噪声
                self.logger.log_rejected(
                    symbol, RejectReason.HIGH_NOISE,
                    f"noise_ratio={noise:.3f}", trade_date
                )
                continue

            # 计算技术模式优先级
            tech_priority = self._get_tech_priority(sig.signal_subtype)

            # 计算融合分
            fused_score = self._calc_fused_score(sig, fund_score)

            if fused_score < self.buy_score_min:
                self.logger.log_rejected(
                    symbol, RejectReason.LOW_FUSED_SCORE,
                    f"fused_score={fused_score:.1f} < {self.buy_score_min}",
                    trade_date,
                )
                continue

            # 标记融合分
            sig.fund_score = fund_score
            sig.fused_score = fused_score

            # 入队列
            ps = PrioritySignal(sig, tech_priority, fund_score)
            pq.push(ps)

        # 2. 按优先级依次处理（受预算约束）
        decisions: List[Tuple[str, float, str]] = []

        for _ in range(min(self.sort_top_n, len(pq))):
            if self.budget_manager.is_exhausted():
                break

            ps = pq.pop()
            if ps is None:
                break

            symbol = ps.signal.symbol
            fund_score = ps.fund_score

            # 计算技术仓位系数
            tech_coeff = self._get_tech_coeff(ps.signal.signal_subtype)

            # 计算基本面信心调节因子
            fund_coeff = self._get_fund_coeff(fund_score)

            # 计算组合风险调节因子
            portfolio_coeff = self._get_portfolio_coeff()

            # 最终目标仓位
            target_weight = (tech_coeff * fund_coeff * portfolio_coeff)

            # 应用单只股票上限
            max_single = self.config.get("position", {}).get(
                "single_stock_max_pct", 0.08
            )
            target_weight = min(target_weight, max_single)

            # 跳空过滤
            if self._check_jump_gap(symbol, ps.signal, trade_date):
                self.logger.log_rejected(
                    symbol, RejectReason.JUMP_GAP,
                    f"price gap > 3%", trade_date
                )
                continue

            # 冷却期检查
            if self.position_manager.is_in_cooling(symbol):
                self.logger.log_rejected(
                    symbol, RejectReason.COOLING_PERIOD,
                    "in cooling period", trade_date
                )
                continue

            # 预算检查
            if not self.budget_manager.can_open(trade_date, target_weight):
                self.logger.log_rejected(
                    symbol, RejectReason.BUDGET_EXHAUSTED,
                    f"budget used {self.budget_manager.used_budget:.1%}",
                    trade_date
                )
                continue

            # 行业集中度检查
            if self.position_manager.exceeds_sector_limit(symbol, target_weight):
                self.logger.log_rejected(
                    symbol, RejectReason.SECTOR_LIMIT,
                    "sector exposure exceeded", trade_date
                )
                continue

            # 通过所有检查，接受信号
            decisions.append((symbol, target_weight, ps.signal.subtype))
            self.budget_manager.allocate(trade_date, target_weight)

            # 日志
            reason_detail = (
                f"tech_coeff={tech_coeff:.3f}, fund_coeff={fund_coeff:.3f}, "
                f"portfolio_coeff={portfolio_coeff:.3f}"
            )
            self.logger.log_accepted(
                symbol, ps.signal.signal_subtype, target_weight,
                fund_score, ps.signal.fused_score, trade_date,
                reason_detail,
            )

        return decisions

    def _get_tech_priority(self, signal_subtype: str) -> int:
        """根据技术信号子类型确定优先级"""
        mapping = {
            "hard_divergence": SignalPriority.TECH_HARD_DIVERGENCE,
            "third_point_buy": SignalPriority.TECH_THIRD_POINT,
            "second_class_buy": SignalPriority.TECH_THIRD_POINT,
            "soft_divergence": SignalPriority.TECH_SOFT_DIVERGENCE,
        }
        return mapping.get(signal_subtype, SignalPriority.TECH_SOFT_DIVERGENCE)

    def _get_tech_coeff(self, signal_subtype: str) -> float:
        """技术仓位系数

        模式A（强趋势）= 1.0
        模式B（盘整）= 根据盘整子状态降级
        """
        # 硬背驰+强趋势 = 1.0
        if signal_subtype in ("hard_divergence", "trend_follow"):
            return 1.0
        # 三类买点 = 0.8
        if signal_subtype in ("third_point_buy", "second_class_buy"):
            return 0.8
        # 软背驰 = 0.4（由试探仓管理）
        if signal_subtype == "soft_divergence":
            return 0.4

        return 0.5

    def _get_fund_coeff(self, fund_score: float) -> float:
        """基本面信心调节因子"""
        if fund_score > self.fusion_config.get("fund_score_high", 80):
            return self.confidence_high
        elif fund_score > self.fusion_config.get("fund_score_mid", 70):
            return self.confidence_mid
        elif fund_score > self.fusion_config.get("fund_score_low", 60):
            return self.confidence_low
        else:
            return 0.0

    def _get_portfolio_coeff(self) -> float:
        """组合风险调节因子"""
        used_ratio = self.budget_manager.used_budget
        if used_ratio > self.budget_exhaust_threshold:
            return 0.5
        return 1.0

    def _check_jump_gap(self, symbol: str, signal: Signal, trade_date) -> bool:
        """检查跳空"""
        jump_reject_pct = self.config.get("position", {}).get("jump_reject_pct", 0.03)
        # 需要外部传入实际价格来判断跳空
        # 简化：placeholder，默认不过滤
        return False

    def _calc_fused_score(self, signal: Signal, fund_score: float) -> float:
        """计算融合分（0-100）

        融合规则：
        - 技术信号自身置信度（基于信号类型）
        - 基本面得分
        - 两者加权
        """
        tech_score_map = {
            "hard_divergence": 90,
            "third_point_buy": 80,
            "second_class_buy": 75,
            "soft_divergence": 55,
            "trend_follow": 70,
        }
        tech_base = tech_score_map.get(signal.signal_subtype, 50)
        # 融合 = 0.6 * tech_base + 0.4 * fund_score
        fused = 0.6 * tech_base + 0.4 * fund_score
        return fused
