"""
solver.py

实现对房屋数据的特征工程、缺失值处理、异常值处理、特征缩放与结果输出
使用：
    python solver.py

主要步骤：
 1. 读取数据并输出缺失值统计
 2. 对特定列进行缺失值填充（GarageCars, BedroomAbvGr, KitchenQual, MiscFeature）
 3. 处理 SalePrice 缺失（删除）
 4. 检查 LotArea 的负值与极端值并进行修正（winsorize/填充）
 5. 对数值型特征执行 StandardScaler，打印缩放前后均值与标准差
 6. 输出前 5 行并保存处理后的 DataFrame

Author: GitHub Copilot (生成代码示例)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple


def load_data(path: str = 'house_data.csv') -> pd.DataFrame:
    """
    读取house_data.csv数据集
    
    :param path: 文件路径
    :type path: str
    :return: 文件对象
    :rtype: DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件 {path} 未找到，请先运行 `data-generater.py` 生成数据。")
    df = pd.read_csv(path)
    return df


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计缺失值数量和比例. 保持 shape 不变, 将 df 中 NaN 值变为 True, 其他值变为 False. 使用 sum 对列求和
    
    :param df: 原数据
    :type df: pd.DataFrame
    :return: 缺失值统计，包括数量和百分比
    :rtype: DataFrame
    """
    total = df.isna().sum()
    percent = (total / len(df)) * 100
    summary = pd.DataFrame({'missing_count': total, 'missing_percent': percent})
    return summary


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    填充缺失值. 
    
    :param df: 原数据
    :type df: pd.DataFrame
    :return: 填充缺失值后的数据
    :rtype: DataFrame
    """
    df = df.copy()

    # 车库中的车辆数: 
    # NaN 认为是没有车而不填, 用 0 填充
    # 异常值认为是错误数据, 用众数填充
    def clean_garage_value(x):
        if pd.isna(x):
            return 0
        if x > 5 or x < 0:
            return np.nan
        return x
    df['GarageCars'] = df['GarageCars'].apply(clean_garage_value)
    garage_modes = df['GarageCars'].mode()
    fill_garage = int(garage_modes.median()) if len(garage_modes) > 0 else 0
    df['GarageCars'] = df['GarageCars'].fillna(fill_garage)

    # 卧室数量:
    # NaN 和异常值都认为是错误数据, 用众数填充
    def clean_bedroom_value(x):
        if pd.isna(x):
            return np.nan
        if x > 7 or x < 0:
            return np.nan
        return x
    df['BedroomAbvGr'] = df['BedroomAbvGr'].apply(clean_bedroom_value)
    bed_modes = df['BedroomAbvGr'].mode()
    fill_bedroom = int(bed_modes.median()) if len(bed_modes) > 0 else 0
    df['BedroomAbvGr'] = df['BedroomAbvGr'].fillna(fill_bedroom)

    # 厨房质量:
    # 先映射为有序数, NaN 认为是错误数据, 用众数填充
    mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    reverse_mapping = {v: k for k, v in mapping.items()}

    df['KitchenQual_Ordinal'] = df['KitchenQual'].map(mapping).astype(int)
    ordinal_modes = df['KitchenQual_Ordinal'].mode(dropna=True)
    fill_kitchen_ordinal = int(ordinal_modes.median()) if len(ordinal_modes) > 0 else 3
    fill_kitchen = reverse_mapping[fill_kitchen_ordinal]
    df['KitchenQual'] = df['KitchenQual'].fillna(fill_kitchen)

    # 其他特征:
    # NaN 认为是没有特征, 用 'None' 填充
    df['MiscFeature'] = df['MiscFeature'].fillna('None')

    # 售价:
    # 由于是目标值, 直接删除含有 NaN 的行
    # 异常值不处理
    before_rows = df.shape[0]
    df = df.dropna(subset=['SalePrice']).reset_index(drop=True)
    after_rows = df.shape[0]
    print(f"由于缺失售价删除 {before_rows - after_rows} 行")

    return df


def handle_lotarea_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理土地面积的异常值.
    
    :param df: 原数据
    :type df: pd.DataFrame
    :return: 处理后的数据
    :rtype: DataFrame
    """
    df = df.copy()

    # 将负值替换为 NaN, 并删除
    def clean_lotarea_value(x):
        if pd.isna(x):
            return np.nan
        if x < 0:
            return np.nan
        return x
    df['LotArea'] = df['LotArea'].apply(clean_lotarea_value)
    before_rows = df.shape[0]
    df = df.dropna(subset=['LotArea']).reset_index(drop=True)
    after_rows = df.shape[0]
    print(f"由于土地面积异常值或者缺失值删除 {before_rows - after_rows} 行")

    # Winsorize: 限制下 1 百分位和上 99 百分位
    lower = df['LotArea'].quantile(0.01)
    upper = df['LotArea'].quantile(0.99)
    print(f"土地面积1%: {lower:.2f}, 99%: {upper:.2f}")
    df['LotArea'] = df['LotArea'].clip(lower, upper)

    return df

def scale_numerical_features(df: pd.DataFrame, exclude: Optional[List[str]] = None, include_target: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#def scale_numerical_features(df: pd.DataFrame, exclude: list = None, include_target: bool = False) -> tuple:
    """
    标准化数值型特征.
    
    :param df: 原数据
    :type df: pd.DataFrame
    :param exclude: 排除的列名列表
    :type exclude: Optional[List[str]]
    :param include_target: 是否包含目标变量 SalePrice
    :type include_target: bool
    :return: 标准化后的数据，标准化前的统计信息，标准化后的统计信息
    :rtype: Tuple[DataFrame, DataFrame, DataFrame]
    """
    df = df.copy()
    
    # 需要标准化的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude is None:
        exclude = ['Id']
    if not include_target:
        exclude += ['SalePrice']
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    # 标准化前的均值和标准差, agg for aggregate
    before_stats = df[numeric_cols].agg(['mean', 'std']).T
    print(f"标准化前: {before_stats}")

    # 标准化后的均值和标准差
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    after_stats = df[numeric_cols].agg(['mean', 'std']).T
    print(f"标准化后: {after_stats}")

    return df, before_stats, after_stats


def print_stats(before_shape, after_shape, missing_summary, before_stats, after_stats):
    print('\n=== 形状 ===')
    print(f"处理前形状: {before_shape}, 处理后形状: {after_shape}")
    print('\n=== 缺失值汇总 ===')
    print(missing_summary)
    print('\n=== 标准化前的数值统计 ===')
    print(before_stats)
    print('\n=== 标准化后的数值统计 ===')
    print(after_stats)


def main():
    df = load_data('house_data.csv')
    before_shape = df.shape

    print('\n任务 1: 缺失值分析和处理:')
    miss_summary = missing_value_summary(df)
    print(miss_summary)
    df = impute_missing_values(df)

    print('\n任务 2: 异常值检测与处理:')
    df = handle_lotarea_outliers(df)

    print('\n任务 3: 特征缩放:')
    df_scaled, before_stats, after_stats = scale_numerical_features(df, exclude=['Id', 'KitchenQual_Ordinal'], include_target=False)
    after_shape = df_scaled.shape

    print('\n任务 4: 输出结果和总结:')
    print_stats(before_shape, after_shape, miss_summary, before_stats, after_stats)

    print('\n处理后数据集的前 5 行:')
    print(df_scaled.head(5))

    # 保存处理后的数据
    processed_path = 'house_data_processed.csv'
    df_scaled.to_csv(processed_path, index=False)
    print(f'保存到 {processed_path}')


if __name__ == '__main__':
    main()
