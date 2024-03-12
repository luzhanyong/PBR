import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

data = [2,134,24,21,38,39,103]

## 查看数据是否支持正太分布

# 绘制直方图
plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True, color='skyblue')
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 进行正态性检验
stat, p_value = shapiro(data)

print(f'Shapiro-Wilk正态性检验结果：\n统计值={stat}, p-value={p_value}')

# 根据p-value判断
alpha = 0.05
if p_value > alpha:
    print('数据可能符合正态分布')
else:
    print('数据不符合正态分布')
