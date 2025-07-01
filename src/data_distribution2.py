import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\AUC_Cmax\AUC\AUC_features.csv')

# 设置图形风格
sns.set(style="whitegrid")

# 绘制直方图 + 核密度估计
plt.figure(figsize=(10, 6))
ax = sns.histplot(df['Drug_dose'].dropna(), bins=10, kde=True, color='teal')

# 自定义边框线（移除上方和右侧的线）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)  # 左侧边框加粗（可选）
ax.spines['bottom'].set_linewidth(1.5)  # 底部边框加粗（可选）

# 添加标签和标题
plt.grid(linestyle='', alpha=0.7)  # 修改网格线为虚线
plt.tick_params(axis='both', labelsize=16)
plt.xlabel('Drug dose (mg)', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Distribution of Drug dose', fontsize=20)

# 保存高清图片
plt.savefig(r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\Drug_dose.png", dpi=600, bbox_inches='tight')

# 显示图形
plt.show()