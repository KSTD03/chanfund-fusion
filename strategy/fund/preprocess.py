"""
去极值、标准化、中性化
=================
处理流程：MAD去极值 → 行业分组ZScore → 市值中性化
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def mad_winsorize(
    series: pd.Series,
    n_mad: float = 5.0,
) -> pd.Series:
    """MAD去极值

    将超过 median ± n*mad 的值截断到边界。

    Args:
        series: 因子值序列
        n_mad: MAD倍数

    Returns:
        winsorized: 截断后的序列
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    lower = median - n_mad * mad
    upper = median + n_mad * mad
    return series.clip(lower=lower, upper=upper)


def group_zscore(
    factor_df: pd.DataFrame,
    factor_name: str,
    group_field: str = "industry",
) -> pd.DataFrame:
    """行业分组ZScore标准化

    注意：必须在去极值之后执行。
    在每个申万一级行业内部计算均值和标准差进行标准化，
    避免将行业间的系统性差异带入后续步骤。

    Args:
        factor_df: 因子DataFrame，包含 factor_name 和 group_field 列
        factor_name: 因子列名
        group_field: 分组字段，如 "industry"

    Returns:
        factor_df: 添加了标准化列 factor_name_z 的DataFrame
    """
    factor_name_z = f"{factor_name}_z"

    def _zscore_group(group):
        mean = group[factor_name].mean()
        std = group[factor_name].std()
        if std == 0:
            group[factor_name_z] = 0.0
        else:
            group[factor_name_z] = (group[factor_name] - mean) / std
        return group

    factor_df = factor_df.groupby(group_field, group_keys=False).apply(_zscore_group)
    return factor_df


def market_neutralize(
    factor_df: pd.DataFrame,
    factor_name: str,
    mktcap_col: str = "log_mktcap",
) -> pd.DataFrame:
    """市值中性化

    用截面回归去除市值影响：
        factor_z = alpha + beta * log_mktcap + epsilon
    取残差 epsilon 作为最终因子值。
    避免模型隐含做多小盘。

    Args:
        factor_df: 标准化的因子DataFrame
        factor_name: 标准化后的因子列名（如 factor_name_z）
        mktcap_col: 对数市值列名

    Returns:
        factor_df: 添加中性化列 factor_name_neutral 的DataFrame
    """
    factor_neutral = f"{factor_name}_neutral"
    factor_df[factor_neutral] = np.nan

    # 逐日回归（如果有日期列）
    if "date" in factor_df.columns:
        for dt, group in factor_df.groupby("date"):
            idx = group.dropna(subset=[factor_name, mktcap_col]).index
            if len(idx) < 20:  # 样本太少直接用zscore
                factor_df.loc[idx, factor_neutral] = group.loc[idx, factor_name]
                continue
            x = group.loc[idx, mktcap_col].values
            y = group.loc[idx, factor_name].values
            x_mean = x.mean()
            y_mean = y.mean()
            beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2 + 1e-10)
            residual = y - beta * x
            factor_df.loc[idx, factor_neutral] = residual
    else:
        # 单截面回归
        idx = factor_df.dropna(subset=[factor_name, mktcap_col]).index
        if len(idx) >= 20:
            x = factor_df.loc[idx, mktcap_col].values
            y = factor_df.loc[idx, factor_name].values
            x_mean = x.mean()
            y_mean = y.mean()
            beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2 + 1e-10)
            residual = y - beta * x
            factor_df.loc[idx, factor_neutral] = residual
        else:
            factor_df[factor_neutral] = factor_df[factor_name]

    return factor_df


def standardize_pipeline(
    factor_df: pd.DataFrame,
    factor_name: str,
    group_field: str = "industry",
    mktcap_col: str = "log_mktcap",
    n_mad: float = 5.0,
    do_neutralize: bool = True,
) -> pd.DataFrame:
    """完整三步标准化流水线

    流程：
        1. MAD去极值
        2. 行业分组ZScore
        3. 市值中性化（可选）

    Args:
        factor_df: 因子数据
        factor_name: 因子列名
        group_field: 行业分类字段
        mktcap_col: 对数市值列
        n_mad: MAD倍数
        do_neutralize: 是否执行市值中性化

    Returns:
        factor_df: 处理后的DataFrame
    """
    # 1. 去极值
    col_clean = f"{factor_name}_clean"
    factor_df[col_clean] = mad_winsorize(factor_df[factor_name], n_mad)

    # 2. 行业ZScore
    factor_df = group_zscore(factor_df, col_clean, group_field)
    z_col = f"{col_clean}_z"

    # 3. 市值中性化
    if do_neutralize:
        factor_df = market_neutralize(factor_df, z_col, mktcap_col)

    return factor_df
