import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
data = pd.read_csv(r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\model_2\dissolution_features.csv', encoding='unicode_escape')

# 提取目标列
output_diss_fraction = data['Polymer_tg']

# 去除NaN值
output_diss_fraction = output_diss_fraction.dropna()

# 确定数据范围并创建自定义bins
min_value = np.floor(np.min(output_diss_fraction))  # 取最小值的整数部分
max_value = np.ceil(np.max(output_diss_fraction))  # 取最大值的整数部分
bins = [min_value]
bins.extend(np.arange(max(0, min_value), 100, 10))  # 小于1000的部分，间隔100
if max_value > 100:
    bins.extend(np.arange(100, int(max_value) + 1000, 1000))  # 大于1000的部分，间隔1000
bins = np.unique(bins).tolist()  # 去除重复值

# 计算频率
hist, edges = np.histogram(output_diss_fraction, bins=bins)

# 调试输出
print("Bins:", bins)
print("Edges:", edges)
print("Histogram:", hist)

# 绘制直方图，使用plt.bar确保等宽且无缝衔接
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
bar_width = 1  # 所有柱子宽度设为1，确保无缝衔接
positions = np.arange(len(hist))  # 柱子位置，从0开始递增
plt.bar(positions, hist, width=bar_width, color='lightblue', edgecolor='black', alpha=0.7)

# 添加标题和标签
plt.title('Distribution of Dissolution Time', fontsize=20)
plt.xlabel('Dissolution Time', fontsize=20)
plt.ylabel('Frequency', fontsize=20)

# 设置刻度字体大小
plt.tick_params(axis='both', labelsize=16)

# 去掉顶部和右侧边框，仅保留底边和左边
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 设置x轴刻度为实际的bins值
tick_labels = [int(edge) for edge in edges[:-1]]
plt.xticks(positions, tick_labels, rotation=45)
#plt.xticks(positions, [int(edge) for edge in edges[:-1]], rotation=45)
plt.xlim(-1, len(hist)-1)
# 显示网格
plt.grid(axis='y', linestyle='', alpha=0.7)
plt.savefig(r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\model_2\diss_time_distribution.png", dpi=600)
# 显示图像
plt.show()