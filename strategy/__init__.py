"""
[已迁移] 旧策略目录 — 档案保留
=========================
策略已升级重命名为 ChanFund Fusion v1.0.0
新路径: /home/quant/.openclaw/workspace/chanfund-fusion/

请不要在此目录进行任何修改，所有新开发在 chanfund-fusion/ 中进行。
"""

# 向后兼容：尝试导入新策略
import warnings
warnings.warn(
    "⚠️  'strategy' 目录已归档，请使用 'chanfund-fusion' 目录。"
    " 导入: from chanfund_fusion.strategy import Strategy",
    DeprecationWarning,
    stacklevel=2,
)
