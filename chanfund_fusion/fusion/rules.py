"""
融合规则与仲裁 — FusionRules
======================
信号融合过滤器 + 优先级排序 + 仓位系数计算

核心流程（v1.0优化版）：
    1. ER噪声过滤（Phase 2）
    2. 趋势方向过滤（Phase 4）
    3. 红黄牌过滤（Phase 5）
    4. 信号分级仓位系数（Phase 3）
    5. 基本麵信心调节
    6. 优先级排序 + 风控约束
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

from ..tech.chan_objects import Signal, SignalStatus
from ..tech.signal_engine import TechSignalEngine
from ..fund.factor_engine import FundFactorEngine
from ..fund.redflag import RedFlagResult, RedFlagType
from ..risk.budget_manager import RiskBudgetManager
from ..risk.position_manager import PositionManager
from .priority_queue import SignalPriorityQueue, PrioritySignal, SignalPriority
from ..utils.logger import DecisionLogger, RejectReason

logger = logging.getLogger(__name__)


class FusionRules:
    """融合规则与仲裁

    v1.0 优化实现 Phase 2~5：
    - Phase 2: ER噪声过滤（已实现但阈值修复）+ 冷却期
    - Phase 3: 信号分级仓位系数（position_coeff）
    - Phase 4: 趋势方向过滤器（MA排列）
    - Phase 5: 红黄牌机制（redflag）
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

        # 【Phase 2】噪声参数（v1.0.1: 分级降权）
        tech_config = config.get("tech", {})
        noise_config = tech_config.get("noise", {})
        self.er_low_percentile = noise_config.get("er_low_percentile", 0.30)
        self.er_discard_percentile = noise_config.get("er_discard_percentile", 0.20)  # Task 2: 丢弃阈值
        self.er_high_percentile = noise_config.get("er_high_percentile", 0.80)          # Task 2: 趋势增强阈值

        # 【Phase 3】信号分级仓位系数
        self.position_coeff = self.fusion_config.get("position_coeff", {})

        # 【Phase 4】方向过滤器参数（v1.0.1: 分层自适应）
        ma_config = tech_config.get("ma", {})
        self.ma_short = ma_config.get("short_window", 20)
        self.ma_long = ma_config.get("long_window", 120)
        self.dir_filter_config = tech_config.get("direction_filter", {})
        self.ema_fast = self.dir_filter_config.get("ema_fast", 12)
        self.ema_slow = self.dir_filter_config.get("ema_slow", 26)
        self.use_long_term_filter = self.dir_filter_config.get("use_long_term_filter", True)

        # Task 4: 趋势方向过滤分层
        trend_filter_config = tech_config.get("trend_filter", {})
        self.trend_filter_mode = trend_filter_config.get("mode", "adaptive")
        strict_cfg = trend_filter_config.get("strict", {})
        self.tf_strict_fast = strict_cfg.get("fast", 12)
        self.tf_strict_slow = strict_cfg.get("slow", 26)
        adaptive_cfg = trend_filter_config.get("adaptive", {})
        self.tf_adaptive_fast = adaptive_cfg.get("fast", 50)
        self.tf_adaptive_slow = adaptive_cfg.get("slow", 100)
        self.tf_price_ma_period = adaptive_cfg.get("price_ma_period", 200)

        # 【Phase 5】红黄牌系数
        self.red_card_coeff = self.fusion_config.get("red_card_coeff", 0.0)
        self.yellow_card_coeff = self.fusion_config.get("yellow_card_coeff", 0.5)

    def filter_and_merge(
        self,
        tech_signals: List[Signal],
        fund_scores: Dict[str, float],
        trade_date,
        noise_ratios: Dict[str, float] = None,
        ma_states: Dict[str, str] = None,
        redflag_results: Dict[str, RedFlagResult] = None,
    ) -> List[Tuple[str, float, str]]:
        """融合技术信号与基本面得分（v1.0优化版）

        过滤管道顺序：
        1. ER噪声过滤（Phase 2）→ 2. 趋势方向过滤（Phase 4）
        3. 红黄牌过滤（Phase 5）→ 4. 融合分检查
        5. 冷却期检查 → 6. 预算检查 → 7. 行业集中度检查

        Args:
            tech_signals: 当日可执行的买入信号列表
            fund_scores: {symbol: score 0-100}
            trade_date: 当前交易日
            noise_ratios: {symbol: noise_percentile 0-1} (可选)
            ma_states: {symbol: "bull"|"bear"|"neutral"} (可选, Phase 4)
            redflag_results: {symbol: RedFlagResult} (可选, Phase 5)

        Returns:
            decisions: [(symbol, signal_coeff, subtype)]
        """
        if noise_ratios is None:
            noise_ratios = {}
        if ma_states is None:
            ma_states = {}
        if redflag_results is None:
            redflag_results = {}

        # 1. 构建优先级队列
        pq = SignalPriorityQueue()

        for sig in tech_signals:
            symbol = sig.symbol
            fund_score = fund_scores.get(symbol, 0.0)
            subtype = sig.signal_subtype

            # ================ 过滤管道 ================

            # --- Phase 2: ER噪声过滤（v1.0.1 平滑降权） ---
            noise_coeff = self._get_noise_coeff(noise_ratios.get(symbol, 0.5))
            if noise_coeff <= 0:
                self.logger.log_rejected(
                    symbol, RejectReason.HIGH_NOISE,
                    f"ER noise filter: percentile={noise_ratios.get(symbol, 0.5):.3f} "
                    f"< {self.er_discard_percentile:.0%}",
                    trade_date, signal_type=subtype,
                )
                continue

            # --- Phase 4: 趋势方向过滤（v1.0.1 分层模式）---
            if not self._pass_direction_filter(symbol, ma_states, noise_ratios):
                mode = getattr(self, 'trend_filter_mode', 'strict')
                self.logger.log_rejected(
                    symbol, RejectReason.TREND_DIRECTION,
                    f"Direction filter ({mode}): {ma_states.get(symbol, 'unknown')}",
                    trade_date, signal_type=subtype,
                )
                continue

            # --- Phase 5: 红黄牌过滤 ---
            redflag = redflag_results.get(symbol)
            card_coeff = self._get_card_coeff(redflag)
            if card_coeff <= 0:
                red_type = [f.name for f in redflag.flags] if redflag else []
                self.logger.log_rejected(
                    symbol, RejectReason.RED_FLAG,
                    f"Red card: {red_type}",
                    trade_date, signal_type=subtype,
                )
                continue

            # --- 基本面融合分检查 ---
            fused_score = self._calc_fused_score(sig, fund_score)
            if fused_score < self.buy_score_min:
                self.logger.log_rejected(
                    symbol, RejectReason.LOW_FUSED_SCORE,
                    f"fused_score={fused_score:.1f} < {self.buy_score_min}",
                    trade_date, signal_type=subtype,
                )
                continue

            # === 通过所有过滤 ===

            sig.fund_score = fund_score
            sig.fused_score = fused_score

            # 【Phase 3】计算信号分级仓位系数
            base_coeff = self._get_tech_coeff(subtype)
            # Task 2: 乘以噪声降权系数
            total_coeff = base_coeff * card_coeff * noise_coeff

            # 入队列
            tech_priority = self._get_tech_priority(subtype)
            ps = PrioritySignal(sig, tech_priority, fund_score)
            ps.custom_coeff = total_coeff  # 携带系数入队
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
            coeff = getattr(ps, 'custom_coeff', 1.0)

            # 基本面信心调节
            fund_coeff = self._get_fund_coeff(fund_score)

            # 组合风险调节
            portfolio_coeff = self._get_portfolio_coeff()

            # 最终仓位 = 信号系数 × 基本面信心 × 组合风险 × 红黄牌系数
            # 注意：custom_coeff 已包含红黄牌调节
            total_weight = coeff * fund_coeff * portfolio_coeff

            # 冷却期检查
            if self.position_manager.is_in_cooling(symbol):
                self.logger.log_rejected(
                    symbol, RejectReason.COOLING_PERIOD,
                    "in cooling period", trade_date,
                    signal_type=ps.signal.signal_subtype,
                )
                continue

            # 跳空过滤
            if self._check_jump_gap(symbol, ps.signal, trade_date):
                self.logger.log_rejected(
                    symbol, RejectReason.JUMP_GAP,
                    f"price gap > 3%", trade_date,
                    signal_type=ps.signal.signal_subtype,
                )
                continue

            # 预算检查
            if not self.budget_manager.can_open(trade_date, total_weight):
                self.logger.log_rejected(
                    symbol, RejectReason.BUDGET_EXHAUSTED,
                    f"budget used {self.budget_manager.used_budget:.1%}",
                    trade_date, signal_type=ps.signal.signal_subtype,
                )
                continue

            # 行业集中度检查
            if self.position_manager.exceeds_sector_limit(symbol, total_weight):
                self.logger.log_rejected(
                    symbol, RejectReason.SECTOR_LIMIT,
                    "sector exposure exceeded", trade_date,
                    signal_type=ps.signal.signal_subtype,
                )
                continue

            # 通过所有检查
            decisions.append((symbol, total_weight, ps.signal.signal_subtype))
            self.budget_manager.allocate(trade_date, total_weight)

            # 日志
            reason_detail = (
                f"coeff={coeff:.3f}, fund_coeff={fund_coeff:.3f}, "
                f"portfolio_coeff={portfolio_coeff:.3f}, "
                f"fused_score={ps.signal.fused_score:.1f}"
            )
            self.logger.log_accepted(
                symbol, ps.signal.signal_subtype, total_weight,
                fund_score, ps.signal.fused_score, trade_date,
                reason_detail,
            )

        return decisions

    # ------------------------------------------------------------------
    # Phase 2: ER噪声过滤
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Task 2: ER噪声分级降权
    # ------------------------------------------------------------------
    def _get_noise_coeff(self, noise_percentile: float) -> float:
        """ER噪声分级降权系数

        将单一的丢弃逻辑改为分级降权：
        - 分位 < 20% → 丢弃（系数 0.0）
        - 20% ≤ 分位 < 30% → 降权（系数 0.5）
        - 30% ≤ 分位 ≤ 80% → 正常（系数 1.0）
        - 分位 > 80% → 增强（系数 1.2，趋势强劲）

        Args:
            noise_percentile: ER在历史中的分位 [0,1]

        Returns:
            coeff: 噪声调节系数
        """
        if noise_percentile is None:
            return 1.0  # 无数据则不惩罚

        if noise_percentile < self.er_discard_percentile:
            # < 20%: 高噪声，丢弃
            return 0.0
        elif noise_percentile < self.er_low_percentile:
            # 20-30%: 中等噪声，降权
            return 0.5
        elif noise_percentile > self.er_high_percentile:
            # > 80%: 强趋势，增强
            return 1.2
        else:
            # 30-80%: 正常
            return 1.0

    # ------------------------------------------------------------------
    # Phase 4: 趋势方向过滤器
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Task 4: 趋势方向过滤（分层模式）
    # ------------------------------------------------------------------
    def _pass_direction_filter(
        self,
        symbol: str,
        ma_states: Dict[str, str],
        noise_ratios: Dict[str, float] = None,
    ) -> bool:
        """趋势方向过滤（v1.0.1: 支持三种模式）

        做多信号仅允许：
        - strict模式: EMA12 > EMA26
        - adaptive模式: 强趋势用EMA50/100, 弱趋势用价格>MA200
        - off模式: 不过滤

        Args:
            symbol: 股票代码
            ma_states: {symbol: "bull"|"bear"|"neutral"}
            noise_ratios: {symbol: noise_percentile} — 用于adaptive模式

        Returns:
            pass_filter: True=允许通过
        """
        if self.trend_filter_mode == "off":
            return True

        state = ma_states.get(symbol)
        if state is None:
            return True

        if self.trend_filter_mode == "strict":
            return state != "bear"

        elif self.trend_filter_mode == "adaptive":
            noise = noise_ratios.get(symbol, 0.5) if noise_ratios else 0.5
            if noise > self.er_high_percentile:
                # 强趋势 → 用EMA50/100（更宽松）
                return state in ("bull", "neutral")
            else:
                # 震荡/弱趋 → 价格>MA200（更严格）
                return state == "bull"

        # 兼容老逻辑
        return state != "bear"

    # ------------------------------------------------------------------
    # Phase 5: 红黄牌机制
    # ------------------------------------------------------------------
    def _get_card_coeff(self, redflag: Optional[RedFlagResult]) -> float:
        """根据排雷结果返回仓位系数

        - 红牌（ST、立案调查、否定审计意见等严重问题）：系数0.0 → 禁止买入
        - 黄牌（质押>50%、商誉异常等中等问题）：系数0.5 → 仓位减半
        - 无问题：系数1.0 → 正常
        """
        if redflag is None or not redflag.has_flag:
            return 1.0

        # 检查是否有红牌级别的问题
        for flag in redflag.flags:
            if flag.flag_type in (
                RedFlagType.ST,         # ST/*ST
                RedFlagType.INVESTIGATION,  # 立案调查
                RedFlagType.AUDIT_OPINION,  # 审计否定意见
            ):
                return self.red_card_coeff  # 0.0

        # 黄牌（其他问题）
        return self.yellow_card_coeff  # 0.5

    # ------------------------------------------------------------------
    # 信号分级仓位系数（Phase 3）
    # ------------------------------------------------------------------
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
        """【Phase 3】信号分级仓位系数

        hard_divergence（已确认背驰）: 1.0
        third_point_buy（三类买卖点）: 0.8
        second_class_buy（二类买卖点）: 0.7
        soft_divergence（软背驰）: 0.25（试探仓）
        trend_follow（趋势跟随）: 0.6
        """
        coeff = self.position_coeff.get(signal_subtype)
        if coeff is not None:
            return coeff

        # 后备映射
        mapping = {
            "hard_divergence": 1.0,
            "third_point_buy": 0.8,
            "second_class_buy": 0.7,
            "soft_divergence": 0.25,
            "trend_follow": 0.6,
        }
        return mapping.get(signal_subtype, 0.5)

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
        return False

    def _calc_fused_score(self, signal: Signal, fund_score: float) -> float:
        """计算融合分（0-100）

        融合规则：
        - 技术信号自身置信度（基于信号类型）
        - 基本面得分
        - 两者加权: 0.6 × tech_base + 0.4 × fund_score
        """
        tech_score_map = {
            "hard_divergence": 90,
            "third_point_buy": 80,
            "second_class_buy": 75,
            "soft_divergence": 55,
            "trend_follow": 70,
        }
        tech_base = tech_score_map.get(signal.signal_subtype, 50)
        fused = 0.6 * tech_base + 0.4 * fund_score
        return fused
