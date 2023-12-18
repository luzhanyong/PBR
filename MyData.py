import random
import time
# 设置随机数种子
random.seed(0)

# # 定义机器和作业集合
# M = list(range(1, 8))  # 7台机器
# J = list(range(1, 41))  # 40个作业
#
# # 随机生成作业权重和处理时间
# w = {j: random.randint(1, 5) for j in J}
# p = {j: {i: random.randint(1, 10) for i in M} for j in J}


M_Type = list(range(1, 13))  # 12台机器
J_Type = list(range(1, 8))  # 7种作业

# 权重直接设置为任务的优先级 1 2 3  幸存者检测：3    路径规划：2   灾难检测:1
w_J_Type = {1:3,2:3,3:3,4:2,5:2,6:1,7:1}

p_J_Type_By_M_Type = {1:[12.04,6.20,4.54,9.49,4.75,2.54,2.26,1.88,1.21,2.39,39.65,37.6],2:[476.69,245.47,179.74,375.73,187.865,134.53,89.55,74.47,48.02,94.50,1569.71,1488.66],
                      3:[117.95,60.73,44.47,92.97,46.485,24.91,22.15,18.42,11.88,23.38,388.40,368.35],4:[182.60,94.03,68.85,79.71,39.855,21.35,19.00,15.70,10.19,20.05,333.01,315.82],
                      5:[182.59,94.03,68.85,143.92,71.96,38.57,34.30,28.53,18.39,36.20,601.26,570.22,],6:[186.03,95.79,70.15,146.63,73.32,9.29,34.95,29.06,18.74,36.88,612.60,580.96],
                      7:[487.98,251.28,184.00,384.625,192.31,103.07,91.67,76.24,49.16,96.74,96.74,1523.90]}



#从M_J_Type中随机选择m台机器
# m = 7

#从p_J_Type_By_M_Type中随机选择n个作业
# n = 41


import random

# 随机抽取m台机器
m = 8  # 你可以根据需要设置m的值
M = random.sample(M_Type, m)
print(M)
# 抽取n次任务，每次可以重复抽取
n = 20  # 你可以根据需要设置n的值

# 抽取任务
J_T = [random.choice(J_Type) for _ in range(n)]


# 构建字典w，表示抽取任务的权重
w = {i+1: w_J_Type[J_T[i]] for i in range(len(J_T))}

# 构建字典p，表示任务在机器上的执行时间
p = {i+1: {m_: p_J_Type_By_M_Type[J_T[i]][m_-1] for m_ in M} for i in range(len(J_T))}

# 重新编号列表J，表示抽取的任务编号
J = list(range(1, len(J_T)+1))


print(J)

# 计算任务权重

print(w)
# 计算任务在机器上的执行时间

print(p)

