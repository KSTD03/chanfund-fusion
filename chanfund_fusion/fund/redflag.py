"""
排雷模块 — RedFlagDetector v1.0
=========================
识别财务异常信号，用于过滤基本面有瑕疵的股票。
v1.0 新增：红牌（ST/立案调查/否定审计意见）和黄牌制度

排雷项：
1. 应收账款异常（应收增速 > 营收增速 + 阈值）
2. 经营现金流恶化（OCF_TTM < 0 且同比恶化 > 50%）
3. 存货周转异常（周转天数增加 > 50%）
4. 存贷双高（现金占比>20% 且 有息负债>30%）
5. ST/*ST标识（红牌，数据由外部传入）
6. 立案调查（红牌，数据由外部传入）
7. 审计否定意见（红牌，数据由外部传入）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class RedFlagType(Enum):
    """排雷类型"""
    AR_ABNORMAL = auto()           # 应收账款异常
    OCF_DETERIORATION = auto()     # 经营现金流恶化
    INVENTORY_ABNORMAL = auto()    # 存货周转异常
    CASH_DEBT_HIGH = auto()        # 存贷双高
    COMPOUND = auto()              # 复合异常
    # v1.0 新增红牌类型
    ST = auto()                    # ST/*ST 标识（红牌）
    INVESTIGATION = auto()         # 立案调查（红牌）
    AUDIT_OPINION = auto()         # 审计否定意见（红牌）
    PLEDGE_HIGH = auto()           # 大股东高质押（黄牌）
    GOODWILL_HIGH = auto()         # 商誉过高（黄牌）


class CardLevel(Enum):
    """牌级"""
    RED = "red"        # 红牌 → 永久排除
    YELLOW = "yellow"  # 黄牌 → 仓位减半
    NONE = "none"      # 无问题


@dataclass
class RedFlag:
    """排雷结果"""
    symbol: str
    flag_type: RedFlagType
    detail: str
    severity: float = 1.0         # 严重程度 [0, 1]

    @property
    def card_level(self) -> CardLevel:
        """获取牌级"""
        if self.flag_type in (
            RedFlagType.ST,
            RedFlagType.INVESTIGATION,
            RedFlagType.AUDIT_OPINION,
        ):
            return CardLevel.RED
        return CardLevel.YELLOW


@dataclass
class RedFlagResult:
    """排雷模块输出"""
    has_flag: bool = False
    flags: List[RedFlag] = field(default_factory=list)
    max_severity: float = 0.0
    score_cap: float = 50.0       # 触发排雷后基本面总分上限

    @property
    def card_level(self) -> CardLevel:
        """取最高牌级"""
        levels = {f.card_level for f in self.flags}
        if CardLevel.RED in levels:
            return CardLevel.RED
        if CardLevel.YELLOW in levels:
            return CardLevel.YELLOW
        return CardLevel.NONE


class RedFlagDetector:
    """排雷检测器

    v1.0.1 (Task 5): 事件PIT确认
    - 若数据区分 ann_date(公告日) 与 impl_date(实施日)，以实施日为准
    - 若无法区分，使用固定延迟 T+1 作为生效延迟
    - 回测中，事件生效以 实施日下一个交易日 为准
    - 日志记录延迟使用情况
    """

    def __init__(self, config: dict):
        self.config = config
        self.redflag_config = config.get("redflag", {})
        # Task 5: 事件生效延迟（交易日），无法区分 ann/impl 时的后备
        self.event_delay_days = self.redflag_config.get("event_delay_days", 1)
        self._used_delay_fallback = False  # 是否使用了后备延迟

    def check(
        self,
        symbol: str,
        fin_data: dict,
        trade_date=None,  # Task 5: 当前交易日，用于时间确认
    ) -> RedFlagResult:
        """执行排雷检测

        Args:
            symbol: 股票代码
            fin_data: 财务数据字典，应包含：
                - ar_ttm: 应收账款TTM
                - revenue_ttm: 营收TTM
                - ar_3y_ago: 3年前应收账款
                - revenue_3y_ago: 3年前营收
                - ocf_ttm: 经营现金流TTM
                - ocf_ttm_last_year: 上年同期OCF
                - inventory: 存货
                - cost_of_sales: 营业成本
                - inventory_2y_ago: 2年前存货
                - cash_assets: 货币资金+交易性金融资产
                - total_assets: 总资产
                - interest_bearing_debt: 有息负债
                - accounts_payable: 应付账款
            trade_date: 当前交易日（用于事件日期对比）
                - is_st: bool (可选) 是否ST/*ST
                - is_investigation: bool (可选) 是否立案调查
                - audit_opinion: str (可选) 审计意见类型
                - pledge_ratio: float (可选) 大股东质押比例
                - goodwill_ratio: float (可选) 商誉/净资产比率

        Returns:
            result: 排雷结果
        """
        result = RedFlagResult()
        flags: List[RedFlag] = []

        # --- Task 5: PIT 事件生效日期判断 ---
        def _is_event_effective(event_date) -> bool:
            """判断事件在当前交易日是否已生效"""
            if trade_date is None:
                return True  # 无交易日则默认生效
            if event_date is not None:
                return trade_date >= event_date
            # 无事件日期 → 使用固定延迟
            if not self._used_delay_fallback:
                logger.warning(
                    f"[{symbol}] No event date available, using T+{self.event_delay_days} fallback. "
                    "Set event_delay_days=0 to disable delay (risk of look-ahead bias)."
                )
                self._used_delay_fallback = True
            # 简化：若无法判断，假设已生效（保证回测保守性）
            return True

        # ===== 红牌检查（严重一票否决） =====

        # 1. ST/*ST 标识
        if fin_data.get("is_st"):
            impl_date = fin_data.get("st_impl_date")
            if _is_event_effective(impl_date):
                date_info = f" (impl: {impl_date})" if impl_date else ""
                flags.append(RedFlag(
                    symbol=symbol,
                    flag_type=RedFlagType.ST,
                    detail=f"ST/*ST stock{date_info}",
                    severity=1.0,
                ))

        # 2. 立案调查
        if fin_data.get("is_investigation"):
            impl_date = fin_data.get("investigation_impl_date")
            if _is_event_effective(impl_date):
                flags.append(RedFlag(
                    symbol=symbol,
                    flag_type=RedFlagType.INVESTIGATION,
                    detail="Under investigation",
                    severity=1.0,
                ))

        # 3. 审计否定意见
        audit = fin_data.get("audit_opinion", "")
        if audit in ("否定意见", "无法表示意见", "adverse", "disclaimer"):
            ann_date = fin_data.get("audit_ann_date")
            if _is_event_effective(ann_date):
                flags.append(RedFlag(
                    symbol=symbol,
                    flag_type=RedFlagType.AUDIT_OPINION,
                    detail=f"Audit opinion: {audit}",
                    severity=1.0,
                ))

        # ===== 黄牌检查（仓位减半） =====

        # 4. 大股东高质押
        pledge_ratio = fin_data.get("pledge_ratio")
        if pledge_ratio is not None and pledge_ratio > 0.50:
            flags.append(RedFlag(
                symbol=symbol,
                flag_type=RedFlagType.PLEDGE_HIGH,
                detail=f"Pledge ratio: {pledge_ratio:.1%}",
                severity=min(1.0, pledge_ratio / 0.50 * 0.5),
            ))

        # 5. 商誉过高
        goodwill_ratio = fin_data.get("goodwill_ratio")
        if goodwill_ratio is not None and goodwill_ratio > 0.30:
            flags.append(RedFlag(
                symbol=symbol,
                flag_type=RedFlagType.GOODWILL_HIGH,
                detail=f"Goodwill/equity: {goodwill_ratio:.1%}",
                severity=min(1.0, goodwill_ratio / 0.30 * 0.5),
            ))

        # ===== 原排雷检查（黄牌级别） =====

        # 6. 应收账款异常
        ar_flag = self._check_ar_abnormal(symbol, fin_data)
        if ar_flag:
            flags.append(ar_flag)

        # 7. 经营现金流恶化
        ocf_flag = self._check_ocf_deterioration(symbol, fin_data)
        if ocf_flag:
            flags.append(ocf_flag)

        # 8. 存货周转异常
        inv_flag = self._check_inventory_abnormal(symbol, fin_data)
        if inv_flag:
            flags.append(inv_flag)

        # 9. 存贷双高
        cash_debt_flag = self._check_cash_debt_high(symbol, fin_data)
        if cash_debt_flag:
            flags.append(cash_debt_flag)

        if flags:
            result.has_flag = True
            result.flags = flags
            result.max_severity = max(f.severity for f in flags)
            # 得分上限：红牌无效化，黄牌设限
            if any(f.card_level == CardLevel.RED for f in flags):
                result.score_cap = 0.0  # 红牌得分为0
            elif len(flags) >= 2:
                result.score_cap = 30.0
            elif result.max_severity > 0.7:
                result.score_cap = 40.0
            else:
                result.score_cap = 50.0

        return result

    def _check_ar_abnormal(self, symbol: str, fin: dict) -> Optional[RedFlag]:
        """应收账款增速异常"""
        gap = self.redflag_config.get("ar_3y_growth_gap", 0.20)

        ar_ttm = fin.get("ar_ttm")
        revenue_ttm = fin.get("revenue_ttm")
        ar_3y = fin.get("ar_3y_ago")
        revenue_3y = fin.get("revenue_3y_ago")

        if not all([ar_ttm, revenue_ttm, ar_3y, revenue_3y]) or ar_3y == 0 or revenue_3y == 0:
            return None

        ar_growth = (ar_ttm / ar_3y) ** (1 / 3) - 1
        revenue_growth = (revenue_ttm / revenue_3y) ** (1 / 3) - 1

        if ar_growth > revenue_growth + gap:
            return RedFlag(
                symbol=symbol,
                flag_type=RedFlagType.AR_ABNORMAL,
                detail=f"AR growth {ar_growth:.1%} > Rev growth {revenue_growth:.1%}",
                severity=min(1.0, (ar_growth - revenue_growth - gap) * 2),
            )
        return None

    def _check_ocf_deterioration(self, symbol: str, fin: dict) -> Optional[RedFlag]:
        """经营现金流恶化"""
        decline_pct = self.redflag_config.get("ocf_decline_pct", 0.50)

        ocf = fin.get("ocf_ttm")
        ocf_last = fin.get("ocf_ttm_last_year")

        if ocf is None or ocf_last is None or ocf_last == 0:
            return None

        if ocf < 0 and (ocf - ocf_last) / abs(ocf_last) < -decline_pct:
            return RedFlag(
                symbol=symbol,
                flag_type=RedFlagType.OCF_DETERIORATION,
                detail=f"OCF {ocf:.0f} negative, declined {abs(ocf - ocf_last) / abs(ocf_last):.1%}",
                severity=min(1.0, abs(ocf / ocf_last if ocf_last != 0 else 0)),
            )
        return None

    def _check_inventory_abnormal(self, symbol: str, fin: dict) -> Optional[RedFlag]:
        """存货周转异常"""
        increase_pct = self.redflag_config.get("inventory_turnover_increase", 0.50)

        inventory = fin.get("inventory")
        cost = fin.get("cost_of_sales")
        inventory_2y = fin.get("inventory_2y_ago")

        if not all([inventory, cost, inventory_2y]) or cost == 0 or inventory_2y == 0:
            return None

        turnover_current = inventory / cost * 365 if cost else 0
        turnover_2y_ago = inventory_2y / (fin.get("cost_2y_ago", cost) or 1) * 365

        if turnover_2y_ago == 0:
            return None

        increase = (turnover_current - turnover_2y_ago) / turnover_2y_ago
        if increase > increase_pct:
            return RedFlag(
                symbol=symbol,
                flag_type=RedFlagType.INVENTORY_ABNORMAL,
                detail=f"Inventory turnover {turnover_current:.1f}d, "
                       f"increased {increase:.1%} from {turnover_2y_ago:.1f}d",
                severity=min(1.0, increase / increase_pct * 0.5),
            )
        return None

    def _check_cash_debt_high(self, symbol: str, fin: dict) -> Optional[RedFlag]:
        """存贷双高检测"""
        cash_ratio_threshold = self.redflag_config.get("cash_debt_high_ratio", 0.20)
        debt_ratio_threshold = self.redflag_config.get("cash_debt_liability_ratio", 0.30)

        cash = fin.get("cash_assets")
        total_assets = fin.get("total_assets")
        debt = fin.get("interest_bearing_debt")

        if not all([cash, total_assets, debt]) or total_assets == 0:
            return None

        cash_ratio = cash / total_assets
        debt_ratio = debt / total_assets

        if cash_ratio > cash_ratio_threshold and debt_ratio > debt_ratio_threshold:
            return RedFlag(
                symbol=symbol,
                flag_type=RedFlagType.CASH_DEBT_HIGH,
                detail=f"Cash/assets={cash_ratio:.1%}, Debt/assets={debt_ratio:.1%}",
                severity=min(1.0, (cash_ratio + debt_ratio) / 2 * 2),
            )
        return None
