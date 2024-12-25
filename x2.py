import pandas as pd
import matplotlib.pyplot as plt
from itertools import accumulate

# 读取数据
df = pd.read_csv('all.csv', encoding='gbk')

# 计算每条评论的长度
df['length'] = df['evaluation'].apply(lambda x: len(x) if isinstance(x, str) else 0)

# 按评论长度分组统计频数
len_df = df.groupby('length').count()
sent_length = len_df.index.tolist()
sent_freq = len_df['evaluation'].tolist()

# 计算累积百分比
sent_percentage_list = [(count / sum(sent_freq)) for count in accumulate(sent_freq)]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False   # 避免负号显示问题

# 绘制CDF
plt.plot(sent_length, sent_percentage_list, label='累积分布函数')

# 寻找分位点为 quantile 的评论长度
quantile = 0.9
index = None
for length, percentage in zip(sent_length, sent_percentage_list):
    if percentage >= quantile:  # 确保找到的值覆盖目标分位点
        index = length
        break

# 打印分位点结果
if index is not None:
    print(f"\n分位点为 {quantile:.2f} 的微博长度: {index}.")

# 在图上标注分位点
if index is not None:
    plt.hlines(quantile, 0, index, colors="c", linestyles="dashed", label='分位点线')
    plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
    plt.text(index, 0, f"{index}", color="blue")
    plt.text(0, quantile, f"{quantile:.2f}", color="blue")

# 设置图表标题和坐标轴
plt.title("评论长度累积分布函数图")
plt.xlabel("评论长度")
plt.ylabel("评论长度累积频率")
plt.legend()
plt.show()
