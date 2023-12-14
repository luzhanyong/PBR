import pulp
import random

# 机器和作业的集合
M = list(range(1, 8))  # 10台机器
J = list(range(1, 41))  # 10个作业

# 随机生成作业权重和处理时间
random.seed(0)  # 设置种子确保结果一致
w = {j: random.randint(1, 5) for j in J}
p = {j: {i: random.randint(1, 10) for i in M} for j in J}

# 创建线性规划问题实例
prob = pulp.LpProblem("SchedulingProblem", pulp.LpMinimize)

# 决策变量 x_ijs 表示作业 j 在机器 i 上是否在时间 s 开始处理
# 注意，我们需要定义一个处理时间的合理上限，这里我们使用所有作业在所有机器上处理时间的总和
T = sum(max(p[j].values()) for j in J)
x_ijs = {
    (i, j, s): pulp.LpVariable(f'x_{i}_{j}_{s}', cat='Continuous', lowBound=0)
    for i in M for j in J for s in range(T)
}

# 目标函数：最小化总加权完成时间
prob += pulp.lpSum(
    w[j] * pulp.lpSum((s + p[j][i]) * x_ijs[i, j, s] for i in M for s in range(T))
    for j in J
)

# 约束条件：每个作业在所有机器上只能开始一次
for j in J:
    prob += pulp.lpSum(x_ijs[i, j, s] for i in M for s in range(T)) == 1

# 约束条件：在任何时间点，每台机器上只能有一个作业在处理
for i in M:
    for t in range(T):
        prob += pulp.lpSum(x_ijs[i, j, s] for j in J for s in range(max(0, t - p[j][i]), min(t + 1, T))) <= 1

# 求解问题
prob.solve()

# 输出结果
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)

# 计算总加权完成时间
total_weighted_completion_time = sum(
    w[j] * (s + p[j][i]) * x_ijs[i, j, s].varValue
    for i in M for j in J for s in range(T)
    if x_ijs[i, j, s].varValue > 0
)
print("Total Weighted Completion Time:", total_weighted_completion_time)
