from sklearn.preprocessing import StandardScaler
import pandas as pd
from tabpfn import TabPFNRegressor
import joblib
import numpy as np
from pathlib import Path

# 定义常量
DATA_PATH = r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\data_with_descriptors.csv'
SCALER_PATH = r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\scaler.pkl"
TABPFN_MODEL_PATH = r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\external_val\tabpfn_regressor.pkl"
RF_MODEL_PATH = r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\external_val\best_rf_fold5_global.pkl"
LGBM_MODEL_PATH = r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\external_val\best_lgb_fold1_global.pkl"
OUTPUT_PATH = r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\external_val\external_prediction.csv'

# 加载标准化器和模型
scaler = joblib.load(SCALER_PATH)
tabpfn_model = joblib.load(TABPFN_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
lgbm_model = joblib.load(LGBM_MODEL_PATH)

# 加载新的数据
data_new = pd.read_csv(DATA_PATH, encoding='unicode_escape')

# 分离特征（假设特征从第3列开始，与训练数据一致）
X_new = data_new.iloc[:, 2:]  # 所有特征从第3列开始

# 标准化新的特征数据
X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)

# 使用模型进行预测
predictions_tabpfn = tabpfn_model.predict(X_new_scaled)
predictions_rf = rf_model.predict(X_new_scaled)
predictions_lgbm = lgbm_model.predict(X_new_scaled)

# 将预测结果添加到原始数据中
data_new['Predicted_diss_TabPFN'] = predictions_tabpfn
data_new['Predicted_diss_RF'] = predictions_rf
data_new['Predicted_diss_LGBM'] = predictions_lgbm

# 创建 three_drug_predictions 字典，包含18个处方的预测结果
# 检查是否存在 Output_time 列
has_output_time = 'Output_time' in data_new.columns

# 按模型分别存储预测结果
three_drug_predictions = {}
for formulation_id in data_new['formulation_id'].unique():
    # 筛选特定处方的数据
    formulation_data = data_new[data_new['formulation_id'] == formulation_id]

    # 获取该处方所有时间点的预测结果（如果有 Output_time）
    if has_output_time:
        time_points = formulation_data['Output_time'].values
        predicted_tabpfn = formulation_data['Predicted_diss_TabPFN'].values
        predicted_rf = formulation_data['Predicted_diss_RF'].values
        predicted_lgbm = formulation_data['Predicted_diss_LGBM'].values

        # 按模型存储预测结果（包含时间点）
        three_drug_predictions[formulation_id] = {
            'TabPFN': dict(zip(time_points, predicted_tabpfn)),
            'RF': dict(zip(time_points, predicted_rf)),
            'LGBM': dict(zip(time_points, predicted_lgbm))
        }
    else:
        # 如果没有 Output_time，按索引存储预测结果
        predicted_tabpfn = formulation_data['Predicted_diss_TabPFN'].values
        predicted_rf = formulation_data['Predicted_diss_RF'].values
        predicted_lgbm = formulation_data['Predicted_diss_LGBM'].values

        # 按模型存储预测结果（不包含时间点）
        three_drug_predictions[formulation_id] = {
            'TabPFN': predicted_tabpfn,
            'RF': predicted_rf,
            'LGBM': predicted_lgbm
        }

# 输出预测结果
print("=== Prediction Results for 18 Formulations ===")
for formulation_id, predictions_dict in three_drug_predictions.items():
    print(f"Formulation ID: {formulation_id}")
    for model_name, model_predictions in predictions_dict.items():
        print(f"  Model: {model_name}")
        if has_output_time:
            # 如果有 Output_time，按时间点输出
            for time_point, diss in model_predictions.items():
                print(f"    Time Point: {time_point}, Predicted Diss: {diss:.3f}")
        else:
            # 如果没有 Output_time，按索引输出
            for idx, diss in enumerate(model_predictions):
                print(f"    Index: {idx}, Predicted AUC: {diss:.3f}")
    print("-" * 50)

# 将结果保存为CSV文件
if has_output_time:
    predictions_df = pd.DataFrame({
        'formulation_id': data_new['formulation_id'],
        'time_point': data_new['Output_time'],
        'Predicted_diss_TabPFN': predictions_tabpfn,
        'Predicted_diss_RF': predictions_rf,
        'Predicted_diss_LGBM': predictions_lgbm
    })
else:
    predictions_df = pd.DataFrame({
        'formulation_id': data_new['formulation_id'],
        'Predicted_AUC_TabPFN': predictions_tabpfn,
        'Predicted_AUC_RF': predictions_rf,
        'Predicted_AUC_LGBM': predictions_lgbm
    })

predictions_df.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to '{OUTPUT_PATH}'")