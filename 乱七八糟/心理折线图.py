import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据（严格按照您提供的表格复制）
dates = ['5.23', '5.24', '5.25', '5.26', '5.27', '5.28', '5.29', '5.30', '5.31', '6.1', '6.2', '6.3', '6.4', '6.5']
mood = [6, 5, 7, 4, 7, 8, 7, 6, 4, 7, 7, 6, 8, 7]
loneliness = [4, 5, 3, 6, 2, 2, 3, 4, 7, 3, 3, 4, 2, 3]
connection = [6, 5, 7, 4, 8, 8, 7, 6, 3, 7, 7, 6, 8, 7]
stress = [5, 6, 4, 8, 4, 3, 4, 5, 7, 5, 4, 6, 3, 5]

# 创建画布
plt.figure(figsize=(12, 7), dpi=120)

# 绘制四条折线（不同颜色、线型、标记点，确保清晰可辨）
plt.plot(dates, mood, color='#1f77b4', linestyle='-', marker='o', linewidth=2, markersize=6, label='整体情绪')
plt.plot(dates, loneliness, color='#ff7f0e', linestyle='--', marker='o', linewidth=2, markersize=6, label='孤独感')
plt.plot(dates, connection, color='#2ca02c', linestyle=':', marker='s', linewidth=2, markersize=6, label='社会连接感')
plt.plot(dates, stress, color='#d62728', linestyle='-.', marker='^', linewidth=2, markersize=6, label='压力水平')

# 设置坐标轴
plt.ylim(1, 10)  # 固定纵轴范围1-10
plt.yticks(range(1, 11))  # 纵轴刻度间隔为1
plt.xticks(rotation=45)  # 横轴日期倾斜45度，避免重叠

# 添加标题和标签
plt.title('5月23日-6月5日心理状态四项指标变化趋势图', fontsize=14, pad=20)
plt.xlabel('日期', fontsize=12)
plt.ylabel('评分指数（1-10分）', fontsize=12)

# 添加网格线（浅色，不遮挡数据）
plt.grid(True, linestyle=':', alpha=0.6)

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 调整布局，防止标签被截断
plt.tight_layout()

# 显示图表
plt.show()