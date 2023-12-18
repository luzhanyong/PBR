import time

import pulp

# 假设我们有一组作业（J）和一组机器（M），以及作业的权重（w），处理时间（p）和最大时间（T）
# 你需要用你的具体问题数据来填充这些
# J = ['Job1', 'Job2', 'Job3']
# M = ['Machine1', 'Machine2']
# w = {'Job1': 3, 'Job2': 2, 'Job3': 4}
# p = {'Job1': {'Machine1': 1, 'Machine2': 2}, 'Job2': {'Machine1': 1, 'Machine2': 1}, 'Job3': {'Machine1': 2, 'Machine2': 2}}
# T = 10  # 这是一个示例值
import random
# import LP
#
start_time1 = time.time()
# # 设置随机数种子，以便结果可复现
random.seed(0)
#
# # 机器和作业的集合
# M = LP.M  # 7台机器
# J = LP.J # 40个作业
#
# # 随机生成作业权重和处理时间
# w = LP.w
# p = LP.p

M = list(range(1, 8))  # 7台机器
J = list(range(1, 41))  # 40个作业

# 随机生成作业权重和处理时间
w = {j: random.randint(1, 5) for j in J}
p = {j: {i: random.randint(1, 10) for i in M} for j in J}
print(p)

# 总时间上限T，取所有机器上的最大处理时间之和
T = sum(max(p[j].values()) for j in J)
# 定义问题
prob = pulp.LpProblem("Job_Scheduling", pulp.LpMinimize)

# 决策变量
x = pulp.LpVariable.dicts("x", ((i, j, s) for i in M for j in J for s in range(T)), cat='Binary')

# 目标函数
prob += pulp.lpSum(w[j] * x[(i, j, s)] * (s + p[j][i]) for i in M for j in J for s in range(T))

# 约束条件
for j in J:
    prob += pulp.lpSum(x[(i, j, s)] for i in M for s in range(T)) == 1, f"每个作业_{j}_开始一次"

for i in M:
    for s in range(T):
        prob += pulp.lpSum(x[(i, j, s)] for j in J if s < T - p[j][i]) <= 1, f"每个机器_{i}_在时刻_{s}_只能处理一个作业"

# 求解问题
prob.solve()

# 输出结果
for v in prob.variables():
    if v.varValue != 0:
        print(v.name, "=", v.varValue)


# 记录结束时间
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time1

print(f"程序执行时间：{execution_time} 秒")