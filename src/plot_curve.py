import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 定义文件路径（替换为你的实际结果文件路径）
RESULT_PATH = r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\three_drug_predictions.csv'
# 指定要绘制的药物（替换为你的目标 drug 值，例如 'DrugA'）
TARGET_DRUG = 'posaconazole'  # 请根据你的数据中的 drug 值修改
#posaconazole,tacrolimus,griseofulvin

# 加载数据
data = pd.read_csv(RESULT_PATH)

# 确保每个 formulation_id 和 drug 组合在 time_point = 0 时 Predicted_diss 为 0
# 遍历所有 formulation_id 和 drug 的唯一组合
for formulation_id, drug in data[['formulation_id', 'API_name']].drop_duplicates().values:
    # 筛选当前组合的数据
    mask = (data['formulation_id'] == formulation_id) & (data['API_name'] == drug)
    formulation_data = data[mask]

    # 检查是否已有 time_point = 0 的数据点
    if not (formulation_data['time_point'] == 0).any():
        # 如果没有 time_point = 0 的点，添加一个 Predicted_diss = 0 的点
        new_row = pd.DataFrame({
            'formulation_id': [formulation_id],
            'time_point': [0],
            'Predicted_diss': [0],
            'API_name': [drug]
        })
        data = pd.concat([data, new_row], ignore_index=True)

# 按 drug, formulation_id 和 time_point 排序
data = data.sort_values(by=['API_name', 'formulation_id', 'time_point'])

# 筛选目标药物的数据
target_data = data[data['API_name'] == TARGET_DRUG]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制目标药物下每个 formulation_id 的平滑曲线
for formulation_id in target_data['formulation_id'].unique():
    formulation_data = target_data[target_data['formulation_id'] == formulation_id]

    # 获取原始数据
    x = formulation_data['time_point'].values
    y = formulation_data['Predicted_diss'].values

    # 增加插值点以平滑曲线（例如生成 100 个点）
    x_smooth = np.linspace(x.min(), x.max(), 100)
    interpolator = interp1d(x, y, kind='cubic')  # 使用三次样条插值
    y_smooth = interpolator(x_smooth)

    # 绘制平滑曲线
    plt.plot(x, y, label=f'{formulation_id}')

    # 可选：绘制原始数据点
    plt.scatter(x, y, color='red', s=20, zorder=5)


# 添加标题和标签
plt.title(f'Molecular Dissolution Profiles for {TARGET_DRUG}', fontsize=14)
plt.xlabel('Time Point(min)', fontsize=12)
plt.ylabel('Mass.(mg)', fontsize=12)

# 添加图例
plt.legend()

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 可选：保存图形
plt.savefig(r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\dissolution_curves.png', dpi=600)
print("Dissolution curves saved as 'dissolution_curves.png'")