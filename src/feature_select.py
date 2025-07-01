import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 读取数据
file_path = r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\descriptor2.csv'
data = pd.read_csv(file_path, encoding='unicode_escape')

# 目标变量
target_col = 'Output_diss_fraction'
test_size = 0.2
random_state = 42
num_features_to_select = 50
corr_threshold = 0.1
high_corr_threshold = 0.7
mi_threshold = 0.01

# 分组数据集拆分（固定数据划分，避免特征选择不稳定）
gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
for train_idx, test_idx in gss.split(data, groups=data['formulation_id']):
    train_data = data.iloc[train_idx].copy()
    test_data = data.iloc[test_idx].copy()

print(f"训练集样本数: {len(train_data)}")
print(f"测试集样本数: {len(test_data)}")

# 分离特征和目标变量（仅使用训练集进行特征选择）
X_train = train_data.drop(columns=[target_col, 'formulation_id']).copy()
y_train = train_data[target_col]
X_test = test_data.drop(columns=[target_col, 'formulation_id']).copy()
y_test = test_data[target_col]

# 处理特征数据类型，确保所有特征为 float
error_columns = []
for col in X_train.columns:
    try:
        X_train[col] = X_train[col].astype(float)
    except ValueError:
        error_columns.append(col)

if error_columns:
    print(f"无法转换为 float 的列: {error_columns}")
    X_train.drop(columns=error_columns, inplace=True)
else:
    print("所有列均可成功转换为 float")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# 1. 皮尔逊相关系数筛选（仅在训练集上）
corr_matrix = X_train_scaled.corrwith(y_train).abs()
selected_corr_features = corr_matrix[corr_matrix > corr_threshold].index.tolist()
print(f"通过皮尔逊相关系数筛选的特征数: {len(selected_corr_features)}")
print(f"被皮尔逊相关系数筛选掉的特征数: {len(X_train.columns) - len(selected_corr_features)}")

# 2. 互信息筛选（仅在训练集上）
mi_scores = mutual_info_regression(X_train_scaled[selected_corr_features], y_train)
mi_scores_series = pd.Series(mi_scores, index=selected_corr_features)
mi_selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
print(f"互信息筛选后特征数: {len(mi_selected_features)}")

# 3. RFE 递归特征消除（仅在训练集上）
rfe_model = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
rfe = RFE(estimator=rfe_model, n_features_to_select=num_features_to_select, step=10, verbose=1)
rfe.fit(X_train_scaled[mi_selected_features], y_train)

best_features_rfe = X_train[mi_selected_features].columns[rfe.support_].tolist()
print(f"通过 RFE 筛选的特征数: {len(best_features_rfe)}")

# 4. 相关性分析去除冗余特征
X_rfe = X_train_scaled[best_features_rfe].copy()
corr_matrix = X_rfe.corr().abs()

# 计算所有 RFE 选中特征的互信息分数
mi_scores_final = mutual_info_regression(X_train_scaled[best_features_rfe], y_train)
mi_scores_final_series = pd.Series(mi_scores_final, index=best_features_rfe)

# 去除高相关性的特征
to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > high_corr_threshold:
            feature_i = best_features_rfe[i]
            feature_j = best_features_rfe[j]
            mi_i = mi_scores_final_series[feature_i]
            mi_j = mi_scores_final_series[feature_j]
            to_drop.add(feature_j if mi_i > mi_j else feature_i)

final_features = [f for f in best_features_rfe if f not in to_drop]
print(f"被相关性分析移除的特征: {list(to_drop)}")
print(f"通过相关性分析后保留的特征数: {len(final_features)}")
print(f"最终特征列表: {final_features}")

# 可视化最终特征的相关性矩阵
X_final = X_train_scaled[final_features].copy()
final_corr_matrix = X_final.corr().abs()

plt.figure(figsize=(12, 10))
sns.heatmap(final_corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt='.2f')
plt.title("Final Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\corr_map.png", dpi=600)
plt.close()


# 应用特征选择到测试集（保持特征一致）
X_test = X_test[final_features]  # 确保测试集特征与训练集一致
missing_features = [f for f in final_features if f not in X_test.columns]
if missing_features:
    raise ValueError(f"测试集中缺少特征: {missing_features}")

# 仅保留最终选择的特征用于后续分析
final_selected_data_train = train_data[['formulation_id', target_col] + final_features].copy()
final_selected_data_test = test_data[['formulation_id', target_col] + final_features].copy()

# 保存筛选后的数据
output_file_train = r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\dissolution_train.csv"
output_file_test = r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\dissolution_test.csv"

final_selected_data_train.to_csv(output_file_train, index=False)
final_selected_data_test.to_csv(output_file_test, index=False)

print(f"训练集筛选后的数据已保存为 {output_file_train}")
print(f"测试集筛选后的数据已保存为 {output_file_test}")
