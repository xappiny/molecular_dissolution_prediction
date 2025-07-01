from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from pathlib import Path
import joblib
import pandas as pd
from tabpfn import TabPFNRegressor
import warnings
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import random
import logging
from datetime import datetime
from sklearn.exceptions import DataConversionWarning
import torch  # Added for device management

# 抑制特定警告
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning)

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)

# 设置日志
base_path = Path(r"C:\Users\熊萍\Desktop\ecg-age-prediction-main\In_situ_gel_project\原位凝胶\凝胶_模型结果数据\final_paper_model\first_round\withoutday1")
log_filename = base_path / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting script...")

# 检查 CUDA 可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")
print(f"Using device: {device}")

# 加载数据
data = pd.read_csv(base_path / "scaler_features.csv", encoding='unicode_escape')
X = data.iloc[:, 1:-1]  # 特征
y = data['cumulative_release']  # 目标变量 (连续值)
groups = data['formulation_id']  # 用于分组交叉验证

# 初始化 GroupKFold
gkf = GroupKFold(n_splits=5)

performance_list = []  # 存储每一折的性能
shap_values_all = []   # 存储所有折叠的 SHAP 值
predictions_all = []   # 存储所有折叠的预测值
X_test_all = []        # 存储所有折叠的测试集特征
y_test_all = []        # 存储所有折叠的测试集真实值

# 交叉验证循环
for fold_idx, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups)):
    logging.info(f"\nProcessing Fold {fold_idx + 1}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    groups_test = groups.iloc[test_index]  # 测试集的分组信息

    # 初始化 TabPFNRegressor
    reg = TabPFNRegressor(device=device)  # Use dynamically determined device
    reg.fit(X_train, y_train)

    # 预测测试集
    y_test_pred = reg.predict(X_test)

    # 计算回归性能指标
    metrics = {
        "MSE": mean_squared_error(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "R2": r2_score(y_test, y_test_pred)
    }
    performance_list.append(metrics)

    # 记录每一折的性能
    logging.info(f"Fold {fold_idx + 1} Performance:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    # SHAP 分析：对测试集按分组采样计算 SHAP 值
    n_samples = 50  # 每个折叠采样 30 个样本
    if len(test_index) <= n_samples:
        X_test_sample = X_test
        y_test_sample = y_test
    else:
        # 计算每个组的样本比例
        group_counts = groups_test.value_counts()
        total_samples = len(groups_test)
        group_ratios = {group: count / total_samples for group, count in group_counts.items()}

        # 计算每个组应采样的样本数（每组至少1个）
        samples_per_group = {}
        remaining_samples = n_samples
        for group in group_counts.index:
            # 按照比例分配，但不超过该组的样本数
            available_samples = group_counts[group]
            ratio = group_ratios[group]
            proposed_samples = max(1, int(n_samples * ratio))
            samples_per_group[group] = min(proposed_samples, available_samples)
            remaining_samples -= samples_per_group[group]

        # 分配剩余样本（避免负数）
        if remaining_samples > 0:
            # 按比例重新分配剩余样本
            sorted_groups = sorted(group_counts.index, key=lambda g: group_ratios[g], reverse=True)
            i = 0
            while remaining_samples > 0 and i < len(sorted_groups):
                group = sorted_groups[i]
                available = group_counts[group] - samples_per_group[group]
                if available > 0:
                    add_samples = min(remaining_samples, available)
                    samples_per_group[group] += add_samples
                    remaining_samples -= add_samples
                i += 1

        # 如果仍然不足 n_samples，则从剩余样本中随机补充
        total_assigned = sum(samples_per_group.values())
        if total_assigned < n_samples:
            logging.warning(f"Fold {fold_idx + 1}: Total assigned samples ({total_assigned}) less than n_samples ({n_samples}). Adjusting...")
            remaining_indices = list(set(X_test.index) - set(groups_test[groups_test.isin(samples_per_group.keys())].index))
            if remaining_indices and len(remaining_indices) >= (n_samples - total_assigned):
                extra_indices = np.random.choice(remaining_indices, size=n_samples - total_assigned, replace=False)
                X_test_sample = pd.concat([X_test_sample, X_test.loc[extra_indices]])
                y_test_sample = pd.concat([y_test_sample, y_test.loc[extra_indices]])

        # 按分组采样
        X_test_sample_list = []
        y_test_sample_list = []
        np.random.seed(42 + fold_idx)  # 确保可重复性
        for group, n in samples_per_group.items():
            # 获取该组的样本索引
            group_indices = groups_test[groups_test == group].index

            if len(group_indices) == 0:
                continue  # 防止空组导致的随机采样报错

            # 确保 n 不超过可用样本数
            n = min(n, len(group_indices))
            if n == len(group_indices):
                sampled_indices = group_indices
            else:
                sampled_indices = np.random.choice(group_indices, size=n, replace=False)

            X_test_sample_list.append(X_test.loc[sampled_indices])
            y_test_sample_list.append(y_test.loc[sampled_indices])

        # 合并采样结果
        X_test_sample = pd.concat(X_test_sample_list)
        y_test_sample = pd.concat(y_test_sample_list)

        # 如果采样后样本数仍不等于 n_samples，随机调整
        if len(X_test_sample) > n_samples:
            extra_indices = np.random.choice(X_test_sample.index, size=len(X_test_sample) - n_samples, replace=False)
            X_test_sample = X_test_sample.drop(extra_indices)
            y_test_sample = y_test_sample.drop(extra_indices)
        elif len(X_test_sample) < n_samples:
            remaining_indices = list(set(X_test.index) - set(X_test_sample.index))
            if remaining_indices and len(remaining_indices) >= (n_samples - len(X_test_sample)):
                extra_indices = np.random.choice(remaining_indices, size=n_samples - len(X_test_sample), replace=False)
                X_test_sample = pd.concat([X_test_sample, X_test.loc[extra_indices]])
                y_test_sample = pd.concat([y_test_sample, y_test.loc[extra_indices]])

    # 定义预测函数
    def tabpfn_predict(X_input):
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        if len(X_input.shape) == 1:
            X_input = X_input.reshape(1, -1)
        return reg.predict(X_input)

    # 创建 SHAP 解释器
    explainer = shap.Explainer(tabpfn_predict, X_test_sample)
    shap_values = explainer(X_test_sample)

    # 存储 SHAP 值、预测值和测试集数据
    shap_values_all.append(shap_values.values)  # Extract raw SHAP values
    predictions_all.append(tabpfn_predict(X_test_sample))
    X_test_all.append(X_test_sample)
    y_test_all.append(y_test_sample)

    # 清空 GPU 内存（如果使用 CUDA）
    if device == 'cuda':
        torch.cuda.empty_cache()

# 打印每一折的性能
def print_performance(performance_list, title):
    df = pd.DataFrame(performance_list)
    mean_std = df.agg(['mean', 'std'])
    print(f"\n=== {title} Performance ===")
    for metric in df.columns:
        print(f"{metric}: {mean_std.loc['mean', metric]:.3f}±{mean_std.loc['std', metric]:.3f}")
    logging.info(f"\n=== {title} Performance ===")
    for metric in df.columns:
        logging.info(f"{metric}: {mean_std.loc['mean', metric]:.3f}±{mean_std.loc['std', metric]:.3f}")

print_performance(performance_list, "5-Fold Cross-Validation")

# 保存性能指标到 CSV
performance_df = pd.DataFrame(performance_list)
performance_df.to_csv(base_path / "performance_metrics.csv", index=False)
logging.info("Performance metrics saved to performance_metrics.csv")

# 使用完整数据集训练最终模型并保存
final_model = TabPFNRegressor(device=device)
final_model.fit(X, y)
model_path = base_path / "tabpfn_regressor.pkl"
joblib.dump(final_model, model_path)
logging.info(f"Final model saved to {model_path}")

# 合并所有折叠的 SHAP 结果
shap_values_all = np.concatenate(shap_values_all, axis=0)
predictions_all = np.concatenate(predictions_all, axis=0)
X_test_all = pd.concat(X_test_all, axis=0)
y_test_all = np.concatenate(y_test_all, axis=0)

# 创建 SHAP 数据框
shap_dict = {
    'predicted_value': predictions_all,
    'true_value': y_test_all
}
for i, col in enumerate(X.columns):
    shap_dict[f'shap_{col}'] = shap_values_all[:, i]
    shap_dict[col] = X_test_all[col].values

shap_values_df = pd.DataFrame(shap_dict)

# 保存 SHAP 值到 CSV
shap_values_df.to_csv(base_path / "shap_values.csv", index=False)
logging.info("SHAP values saved to shap_values.csv")

# 创建 SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_all, X_test_all, plot_type="dot", show=False)
plt.title("SHAP Summary Plot", fontsize=16)
plt.tight_layout()
plt.savefig(base_path / "shap_summary_plot.png", dpi=600)
plt.close()
logging.info("SHAP summary plot saved to shap_summary_plot.png")

# 计算 SHAP 值重要性
mean_shap_values = np.abs(shap_values_all).mean(axis=0)
shap_importance = pd.Series(mean_shap_values, index=X.columns)
total_shap = shap_importance.sum()
shap_percentage = (shap_importance / total_shap * 100).round(2)
feature_importance_df = pd.DataFrame({
    'SHAP Value': shap_importance,
    'Percentage': shap_percentage
}).sort_values(by='SHAP Value', ascending=False)

# 保存特征重要性到 CSV
feature_importance_df.to_csv(base_path / "feature_importance.csv", index=False)
logging.info("Feature importance saved to feature_importance.csv")

# 绘制 SHAP 特征重要性图
cmap = plt.get_cmap('coolwarm')
plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance_df.index, feature_importance_df['SHAP Value'],
                color=cmap(feature_importance_df['SHAP Value'] / feature_importance_df['SHAP Value'].max()))

for i, (idx, row) in enumerate(feature_importance_df.iterrows()):
    percentage = row['Percentage']
    plt.text(row['SHAP Value'], i, f'{percentage:.2f}%', va='center', ha='left', fontsize=10)

plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
plt.ylabel('Features')
plt.title('Feature Importance Visualization')
plt.gca().invert_yaxis()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, feature_importance_df['SHAP Value'].max()))
plt.colorbar(sm, ax=plt.gca(), label='SHAP Value')
plt.tight_layout()
plt.savefig(base_path / "shap_feature_importance_test.png", dpi=600)
plt.close()
logging.info("SHAP feature importance plot saved to shap_feature_importance_test.png")
logging.info("Script completed successfully!")