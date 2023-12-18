from gurobipy import Model, GRB
import numpy as np
import random
import MyData

# 设置随机数种子
random.seed(0)

# 定义机器和作业集合
M = MyData.M  # 7台机器
J = MyData.J  # 40个作业

# 随机生成作业权重和处理时间
w = MyData.w
p = MyData.p

# 总时间上限T
T = sum(max(p[j].values()) for j in J)
# 创建模型
model = Model("SchedulingProblem")
# 添加变量
x = model.addVars(len(M), len(J), range(int(T) + 1), vtype=GRB.CONTINUOUS, name="x")

# 目标函数
model.setObjective(
    sum(w[j] * (s + p[j][i]) * x[i, j, s] for i in range(len(M)) for j in range(len(J)) for s in range(int(T) + 1)),
    GRB.MINIMIZE
)

# 约束：每个作业在所有机器上只能被调度一次
for j in range(len(J)):
    model.addConstr(
        sum(x[i, j, s] for i in range(len(M)) for s in range(int(T) + 1)) == 1
    )

# 约束：在任何时间，每个机器只能处理一个作业
for i in range(len(M)):
    for t in range(int(T)):
        model.addConstr(
            sum(x[i, j, s] for j in range(len(J)) for s in range(max(0, t - p[j][i] + 1), min(int(T), t + p[j][i]) + 1)) <= 1
        )

# 求解问题
model.optimize()

# Define the three-dimensional matrix to store the results
x_matrix = np.zeros((len(M) + 1, len(J) + 1, T + 1))

# 输出结果
print('Optimal value:', model.ObjVal)
for v in model.getVars():
    if v.X > 1e-5:
        print(f'{v.VarName} = {v.X}')
        # Extract machine, job, and start time from the variable name
        name_parts = v.VarName.split('[')[1].split(']')[0].split(',')
        machine = int(name_parts[0])
        job = int(name_parts[1])
        start_time = int(name_parts[2])
        x_matrix[machine, job, start_time] = v.X

# Output the matrix or process it as needed
print(x_matrix)
