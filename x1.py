import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('all.csv', encoding='gbk')

# 统计每个标签的数量并打印
print(df.groupby('label')['label'].count())

# 计算每条评论的长度
df['length'] = df['evaluation'].apply(lambda x: len(x) if isinstance(x, str) else 0)

# 按评论长度分组统计出现的频数
len_df = df.groupby('length').count()
sent_length = len_df.index.tolist()
sent_freq = len_df['evaluation'].tolist()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei 是黑体，适用于中文
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示问题

# 绘制句子长度及频数的柱状图
plt.bar(sent_length, sent_freq, color='blue')
plt.title('评论长度及出现频数统计图')
plt.xlabel('评论长度')
plt.ylabel('评论长度出现的频数')

# 显示图像
plt.show()
