from gurobipy import Model, GRB
import numpy as np
import random
import time
from data import MyData

### 使用gurobi来求解线性规划问题



# 设置随机数种子
random.seed(0)

# 定义机器和作业集合
M = MyData.M # 7台机器
J = MyData.J  # 40个作业

# 随机生成作业权重和处理时间
w = MyData.w
p = MyData.p




# 总时间上限T
T = sum(max(p[j].values()) for j in J)
print(T)

# 创建模型
model = Model("SchedulingProblem")
model.setParam('Threads', 0)  # 使用所有可用的线程


# 添加变量
x = model.addVars(M, J, range(T + 1), vtype=GRB.CONTINUOUS, name="x")

# 目标函数
model.setObjective(
    sum(w[j] * (s + p[j][i]) * x[i, j, s] for i in M for j in J for s in range(T + 1)),
    GRB.MINIMIZE
)

# 约束：每个作业在所有机器上只能被调度一次
for j in J:
    model.addConstr(
        sum(x[i, j, s] for i in M for s in range(T + 1)) == 1
    )

#约束：在任何时间，每个机器只能处理一个作业
for i in M:
    for t in range(T):
        model.addConstr(
            sum(x[i, j, s] for j in J for s in range(max(0, t - p[j][i] + 1), min(T, t + p[j][i]) + 1)) <= 1
        )

# 求解问题
model.optimize()

# Define the three-dimensional matrix to store the results
x_matrix = np.zeros((len(M)+1, len(J)+1, T + 1))

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


