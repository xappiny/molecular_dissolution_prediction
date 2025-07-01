import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv(r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\model_2\AUC\AUC_features.csv')

# 将pH 1.2, 1.6, 2合并为"pH ≤ 2"
df['Diss_pH_grouped'] = df['Diss_pH'].apply(lambda x: 'pH ≤ 2' if x in [1.2, 1.6, 2] else f'pH:{x}')

# 统计合并后的分布
process_counts = df['Diss_pH_grouped'].value_counts()
print(process_counts)

# 设置珠光色系配色方案（根据合并后的类别数量调整）
colors = ['#70AED4', '#B0D1E4', '#FFF2CD', '#FFF8E5', '#D3E2B7', '#90A8B6', '#E6C9C9', '#D4B5B0']

# 绘制圆环图
plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=0.05, right=0.65, top=0.8, bottom=0.2)

# 使用自定义autopct函数显示百分比（解决小数位数问题）
def my_autopct(pct):
    return f'{pct:.1f}%' if pct > 5 else ''

wedges, texts, autotexts = plt.pie(
    process_counts,
    colors=colors[:len(process_counts)],
    autopct=my_autopct,
    wedgeprops={'edgecolor': 'white', 'width': 0.5},
    pctdistance=0.8,
    textprops={'fontsize': 16}
)

# 调整百分比标签位置和大小
for autotext in autotexts:
    autotext.set_fontsize(20)

plt.title('Dissolution pH', fontsize=20, pad=20)

# 添加图例并移除外框
plt.legend(
    process_counts.index,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    frameon=False,
    fontsize=14
)

# 确保圆环图为圆形
plt.axis('equal')

# 保存图表
plt.savefig(
    r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\Diss_pH_pie_chart_grouped.png',
    dpi=600,
    bbox_inches='tight',
    transparent=True  # 背景透明
)
plt.show()