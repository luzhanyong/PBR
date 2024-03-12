from gurobipy import Model, GRB
import numpy as np
import random
import time
from data import MyData
import gurobipy as gp


# # 定义机器和作业集合
# M = MyData.M  # 7台机器
# J = MyData.J  # 40个作业
#
# # 随机生成作业权重和处理时间
# w = MyData.w
# p = MyData.p
def LP(M,J,w,p):
    T = 0
    for j in J:
        T += max(p[j].values())
    # print(T)

    # 创建模型
    model = Model("SchedulingProblem")
    # model.setParam('Threads', 0)  # 使用所有可用的线程

    # 添加变量 类型是整数
    x = model.addVars(M, J, range(T + 1), vtype=GRB.INTEGER, name="x")
    # # Objective function
    # obj_expr = gp.quicksum(w[j] * gp.quicksum((s + p[j][i]) * x[i, j, s] for s in range(T + 1)) for i in M for j in J)
    # model.setObjective(obj_expr, GRB.MINIMIZE)
    #
    # # Constraint: Each job is scheduled exactly once on all machines
    # for j in J:
    #     constr_expr = gp.quicksum(x[i, j, s] for i in M for s in range(T + 1))
    #     model.addConstr(constr_expr == 1)

    # 目标函数
    obj_expr = 0
    for i in M:
        for j in J:
            for s in range(T + 1):
                obj_expr += w[j] * (s + p[j][i]) * x[i, j, s]
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # 约束：每个作业在所有机器上只能被调度一次
    for j in J:
        constr_expr = 0
        for i in M:
            for s in range(T + 1):
                constr_expr += x[i, j, s]
        model.addConstr(constr_expr == 1)

    # 约束：在任何时间，每个机器只能处理一个作业


    for i in M:
        for t in range(int(T/(len(M)))):
            model.addConstr(gp.quicksum(x[i, j, s] for j in J for s in range(max(0, t - p[j][i] + 1), t)) <= 1)

    # for i in M:
    #     for j in J:
    #         for s in range(T - p[j][i], T):
    #             model.addConstr(x[i, j, s] == 0, f'completion_{i}_{j}_{s}')


    # 求解问题
    model.optimize()

    # 记录结束时间
    solved_time = model.getAttr('Runtime')

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
    # print(x_matrix)
    return  x_matrix,T,model.ObjVal,solved_time
def process_LP(M,J,w,p):
    x_matrix,T,optimal_value,solved_time = LP(M,J,w,p)
    return optimal_value,solved_time

if __name__ == '__main__':
    M0, J0, w0, p0 = MyData.data()
    optimal_value,solved_time = process_LP(M0, J0, w0, p0)
    print(optimal_value,solved_time)