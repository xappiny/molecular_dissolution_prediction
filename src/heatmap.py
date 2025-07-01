from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle  # 正确导入 Wedge 和 Circle

# 读取数据
file_path = r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\dissolution_features.csv'
data = pd.read_csv(file_path, encoding='unicode_escape')

# 目标变量
target_col = 'diss_fraction'

# 分离特征和目标变量
X_selected = data.drop(columns=[target_col,'formulation_id'])
y = data[target_col]

# 打印每列 NaN 值的数量
nan_counts = X_selected.isna().sum()
print("每列 NaN 值数量：")
print(nan_counts[nan_counts > 0])  # 仅打印含 NaN 的列

# 统计总共的 NaN 值数量
total_nans = X_selected.isna().sum().sum()
print(f"数据集中总共的 NaN 数量: {total_nans}")

# 找出含 NaN 的行数
nan_rows = X_selected.isna().any(axis=1).sum()
print(f"包含 NaN 值的行数: {nan_rows}")

# **处理缺失值（如果有）**
X_selected = X_selected.fillna(X_selected.mean())

# **3️⃣ 绘制相关系数矩阵（下三角数字，上三角实心圆圈，纵坐标从下到上）**
# 计算特征之间的相关系数矩阵
corr_matrix = X_selected.corr()

# 绘图设置
plt.figure(figsize=(15, 12))
ax = plt.gca()

# 获取相关系数矩阵的大小
n_features = len(corr_matrix)
feature_names = corr_matrix.columns

# 创建颜色映射
cmap = plt.get_cmap('RdYlBu')  # 使用红-黄-蓝颜色映射，表示从负到正的相关系数

# 遍历相关系数矩阵的每个元素
for i in range(n_features):
    for j in range(n_features):
        if i != j:  # 忽略对角线（自相关）
            corr_value = corr_matrix.iloc[i, j]
            # 纵坐标从下到上，i 从 0 到 n-1 表示从下到上
            i_mapped = i
            center = (j, i_mapped)  # 调整坐标

            if i < j:  # 上三角（右半边）：绘制实心圆圈
                # 圆圈半径根据相关系数绝对值调整（绝对值越大，圆圈越大）
                radius = 0.6 * abs(corr_value)  # 最大半径为 0.4，按绝对值缩放
                # 确定颜色（正相关偏红，负相关偏蓝）
                color = cmap((corr_value + 1) / 2)  # 映射到颜色范围
                ax.add_patch(Circle(center, radius, color=color, edgecolor='w', linewidth=0.5))
            else:  # 下三角（左半边）：显示数字
                ax.text(j, i_mapped, f'{corr_value:.2f}',
                        ha='center', va='center', fontsize=8)

# 设置坐标轴
ax.set_xlim(-0.5, n_features - 0.5)
ax.set_ylim(-0.5, n_features - 0.5)
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))
ax.set_xticklabels(feature_names, rotation=45, ha='right')  # 横坐标从左到右
ax.set_yticklabels(feature_names)  # 纵坐标从下到上
ax.set_xlabel('Features')
ax.set_ylabel('Features')
ax.set_aspect('equal')

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
plt.colorbar(sm, ax=ax, label='Correlation Coefficient', fraction=0.046, pad=0.04)

# 设置标题
plt.title('Correlation Matrix')

# 保存和显示
output_plot = r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\correlation_matrix_filled_circles.png"
plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.show()

print(f"相关系数矩阵图已保存为 {output_plot}")