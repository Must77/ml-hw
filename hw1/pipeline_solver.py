import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 预处理函数 (Pre-pipeline)
# ==========================================
def initial_row_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    在进入 Pipeline 之前执行的行级清洗。
    """
    df = df.copy()
    
    # 删除售价缺失的行
    if 'SalePrice' in df.columns:
        df = df.dropna(subset=['SalePrice'])
    
    # 删除土地面积为负值的行
    mask = (df['LotArea'] >= 0) | (df['LotArea'].isna()) 
    df = df[mask]
    
    return df.reset_index(drop=True)

# ==========================================
# 2. 自定义 Transformers
# ==========================================

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    自定义 Winsorize 转换器：
    限制数据在 1% 和 99% 分位数之间。
    """
    def __init__(self, lower_percentile=0.01, upper_percentile=0.99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
    def fit(self, X, y=None):
        X = np.array(X)
        self.lower_bound_ = np.nanquantile(X, self.lower_percentile, axis=0)
        self.upper_bound_ = np.nanquantile(X, self.upper_percentile, axis=0)
        return self
    
    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    针对厨房质量的特定映射转换器
    """
    def __init__(self, mapping=None):
        self.mapping = mapping if mapping else {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        col_name = X.columns[0]
        return X[col_name].map(self.mapping).values.reshape(-1, 1)

class GarageBedroomCleaner(BaseEstimator, TransformerMixin):
    """
    针对汽车和卧室数量的特殊异常清洗：
    超出范围的值设为 NaN (以便后续 Imputer 填充)
    """
    def __init__(self, max_val=None, min_val=0):
        self.max_val = max_val
        self.min_val = min_val
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        # 将不合法的值设为 NaN
        mask = (X > self.max_val) | (X < self.min_val)
        X[mask] = np.nan
        return X

# ==========================================
# 3. 构建 Pipeline
# ==========================================

def build_pipeline():
    
    # --- 分支 A: LotArea (数值 + 截断 + 标准化) ---
    lot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('winsorizer', OutlierClipper(0.01, 0.99)),
        ('scaler', StandardScaler())
    ])
    
    # --- 分支 B: GarageCars (异常清洗 + 填 0 + 标准化) ---
    garage_pipeline = Pipeline([
        ('cleaner', GarageBedroomCleaner(max_val=5)),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    # --- 分支 C: BedroomAbvGr (异常清洗 + 填众数 + 标准化) ---
    bedroom_pipeline = Pipeline([
        ('cleaner', GarageBedroomCleaner(max_val=8)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])
    
    # --- 分支 D: KitchenQual (填众数 + 映射 + 标准化) ---
    kitchen_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('mapper', CustomOrdinalEncoder()),
        ('scaler', StandardScaler())
    ])
    
    # --- 分支 E: MiscFeature (填 None) ---
    misc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None'))
    ])

    # --- 组合所有分支: ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('lot', lot_pipeline, ['LotArea']),
            ('garage', garage_pipeline, ['GarageCars']),
            ('bedroom', bedroom_pipeline, ['BedroomAbvGr']),
            ('kitchen', kitchen_pipeline, ['KitchenQual']),
            ('misc', misc_pipeline, ['MiscFeature']),
            ('year', StandardScaler(), ['YearBuilt']) 
        ],
        remainder='passthrough' # 其他未列出的列（如 Id）原样保留
        # remainder='drop' # 或者是删除
    )
    
    return preprocessor

def main():
    # 1. 加载
    df = pd.read_csv('house_data.csv')
    print("原始形状:", df.shape)
    
    # 2. 初始清洗 (删除行)
    df_clean = initial_row_clean(df)
    
    # 分离 X (SalePrice 已经在 initial_row_clean 中处理过对齐问题，这里只取特征)
    # 注意：Pipeline 只处理 X，不处理 y
    X = df_clean.drop(columns=['SalePrice'])
    
    print("清洗后用于 Pipeline 的形状:", X.shape)
    
    # 3. 构建并运行 Pipeline
    pipeline = build_pipeline()
    X_processed = pipeline.fit_transform(X)
    
    # 4. 结果整理 (修复 TypeError)
    # ColumnTransformer 会重新排列列的顺序：先是转换器处理的列，最后是 remainder(passthrough) 的列
    # 我们可以手动指定列名以便查看（基于 build_pipeline 中的顺序）
    new_columns = [
        'LotArea_Scaled',    # 分支 A
        'GarageCars_Scaled', # 分支 B
        'Bedroom_Scaled',    # 分支 C
        'Kitchen_Scaled',    # 分支 D
        'MiscFeature',       # 分支 E (这里是字符串！)
        'YearBuilt_Scaled',  # 分支 F
        # remainder='passthrough' 的列 (剩下没被指定的列)
        'Id', 'PoolArea'     
    ]
    
    # 将 numpy 数组转回 DataFrame，这样可以同时容纳数字和字符串
    df_result = pd.DataFrame(X_processed, columns=new_columns)
    
    print("\n处理后的数据 (DataFrame 前 5 行):")
    # 使用 pandas 的 option 来控制显示精度，而不是暴力使用 np.round
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(df_result.head())
    
    # 验证一下类型，你会发现 MiscFeature 是 object，其他是 float
    print("\n各列数据类型:")
    print(df_result.dtypes)
    
    print("\nPipeline 重构完成！")

if __name__ == '__main__':
    main()