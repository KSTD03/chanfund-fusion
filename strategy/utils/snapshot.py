"""
快照与断点续跑 — SnapshotManager
===========================
支持按交易日为单位，序列化/反序列化引擎状态。
快照按回测参数隔离存储，避免不同实验互相覆盖。
"""

from __future__ import annotations

import pickle
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from .logger import DecisionLogger

logger = logging.getLogger(__name__)


class SnapshotManager:
    """快照管理器

    支持策略引擎的快照保存/加载，实现断点续跑。
    快照按实验ID隔离（避免不同参数实验互相覆盖）。
    """

    def __init__(self, base_dir: str = "snapshots", experiment_id: str = "default"):
        self.base_dir = Path(base_dir) / experiment_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        tech_engine_snapshot: dict,
        fund_engine_snapshot: dict,
        decision_logger: DecisionLogger,
        trade_date: date,
        tag: str = "",
    ) -> str:
        """保存完整快照

        Args:
            tech_engine_snapshot: TechSignalEngine.get_snapshot()
            fund_engine_snapshot: FundFactorEngine.get_snapshot()
            decision_logger: DecisionLogger 实例
            trade_date: 当前交易日
            tag: 可选标签

        Returns:
            snapshot_path: 快照文件路径
        """
        tag_str = f"_{tag}" if tag else ""
        filename = f"snapshot_{trade_date.isoformat()}{tag_str}.pkl"
        path = self.base_dir / filename

        snapshot = {
            "date": trade_date,
            "tech_engine": tech_engine_snapshot,
            "fund_engine": fund_engine_snapshot,
            "decision_log": decision_logger.to_dict(),
        }

        with open(path, "wb") as f:
            pickle.dump(snapshot, f)

        logger.info(f"Snapshot saved: {path}")
        return str(path)

    def load_snapshot(self, trade_date: date, tag: str = "") -> Optional[dict]:
        """加载指定日期的快照

        Args:
            trade_date: 目标日期（加载该日或之前最近的快照）
            tag: 标签

        Returns:
            snapshot: 快照数据，或 None
        """
        tag_str = f"_{tag}" if tag else ""
        filename = f"snapshot_{trade_date.isoformat()}{tag_str}.pkl"
        path = self.base_dir / filename

        if path.exists():
            with open(path, "rb") as f:
                snapshot = pickle.load(f)
            logger.info(f"Snapshot loaded: {path}")
            return snapshot

        # 找更早但最近的回测快照
        return self._find_nearest(trade_date, tag)

    def _find_nearest(self, trade_date: date, tag: str = "") -> Optional[dict]:
        """找指定日期之前最近的快照"""
        tag_str = f"_{tag}" if tag else ""
        pattern = f"snapshot_*{tag_str}.pkl"

        snapshots = []
        for f in self.base_dir.glob(pattern):
            try:
                parts = f.stem.replace("snapshot_", "").split(tag_str)[0]
                snap_date = date.fromisoformat(parts)
                snapshots.append((snap_date, f))
            except (ValueError, IndexError):
                continue

        if not snapshots:
            return None

        # 按日期排序，找不超过trade_date的最远日期
        candidates = [(d, p) for d, p in snapshots if d <= trade_date]
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, path = candidates[0]

        with open(path, "rb") as f:
            snapshot = pickle.load(f)
        logger.info(f"Nearest snapshot loaded: {path}")
        return snapshot

    def list_snapshots(self) -> list:
        """列出所有快照"""
        results = []
        for f in sorted(self.base_dir.glob("snapshot_*.pkl")):
            date_str = f.stem.replace("snapshot_", "").split("_")[0]
            try:
                snap_date = date.fromisoformat(date_str)
                size_mb = f.stat().st_size / (1024 * 1024)
                results.append({
                    "date": snap_date,
                    "path": str(f),
                    "size_mb": round(size_mb, 2),
                })
            except ValueError:
                continue
        return results

    def clean_old(self, keep_last_n: int = 5):
        """仅保留最近N个快照"""
        snapshots = self.list_snapshots()
        if len(snapshots) <= keep_last_n:
            return

        snapshots.sort(key=lambda x: x["date"], reverse=True)
        for old in snapshots[keep_last_n:]:
            Path(old["path"]).unlink()
            logger.info(f"Removed old snapshot: {old['path']}")

    def experiment_exists(self, experiment_id: str) -> bool:
        """检查某个实验的快照是否存在"""
        exp_dir = self.base_dir.parent / experiment_id
        return exp_dir.exists() and any(exp_dir.glob("snapshot_*.pkl"))
