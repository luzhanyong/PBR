from scipy.optimize import linprog
import numpy as np
import random
import time

# 设置随机数种子，以便结果可复现
random.seed(0)

# 机器和作业的集合
M = list(range(1, 8))  # 7台机器
J = list(range(1, 41))  # 40个作业

# 随机生成作业权重和处理时间
w = {j: random.randint(1, 5) for j in J}
p = {j: {i: random.randint(1, 10) for i in M} for j in J}
print(p)

# 总时间上限T，取所有机器上的最大处理时间之和
T = sum(max(p[j].values()) for j in J)


# 目标函数的系数
c = []

# 构建目标函数系数
for i in M:
    for j in J:
        for s in range(T + 1):
            c.append(w[j] * (s + p[j][i]))

# 约束条件的矩阵和向量初始化
A_eq = np.zeros((len(J), len(c)))
b_eq = np.ones(len(J))
A_ub = np.zeros((len(M) * T, len(c)))
b_ub = np.ones(len(M) * T)

# 每个作业在所有机器上只能被调度一次
for j in J:
    for i in M:
        for s in range(T + 1):
            A_eq[j-1, (i-1) * len(J) * (T + 1) + (j-1) * (T + 1) + s] = 1

# 在任何时间，每个机器只能处理一个作业
for i in M:
    for t in range(T):
        for j in J:
            for s in range(max(0, t - p[j][i] + 1), min(T, t + p[j][i]) + 1):
                A_ub[(i-1) * T + t, (i-1) * len(J) * (T + 1) + (j-1) * (T + 1) + s] = 1


start_time1 = time.time()

# 求解线性规划问题
bounds = [(0, None) for _ in c]
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')




# # 定义三维矩阵来存储结果
x_matrix = np.zeros((len(M)+1, len(J)+1, T + 1))

#输出结果
print('Optimal value:', res.fun)
if res.success:
    # 输出非零决策变量以查看任务的调度
    for index, x in enumerate(res.x):
        if x > 1e-5:  # 使用小于1e-5作为零的阈值
            machine = index // (len(J) * (T + 1)) + 1
            job = (index % (len(J) * (T + 1))) // (T + 1) + 1
            start_time = index % (T + 1)
            print(f'Job {job} starts on machine {machine} at time {start_time} with intensity {x}')
            x_matrix[machine,job,start_time] = x
else:
    print('No optimal solution found.')



print(x_matrix[1])


# 记录结束时间
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time1

print(f"linprog线性规划求解时间：{execution_time} 秒")