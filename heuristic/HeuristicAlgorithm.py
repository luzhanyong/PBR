# 启发式算法分配作业到机器
# 启发式方法：优先考虑权重最高的作业，并将其分配给当前能最早完成该作业的机器
import random

# 设置随机数种子，以便结果可复现


# # 机器和作业的集合
# M = list(range(1, 8))  # 7台机器
# J = list(range(1, 41))  # 40个作业
#
# # 随机生成作业权重和处理时间
# w = {j: random.randint(1, 5) for j in J}
# p = {j: {i: random.randint(1, 10) for i in M} for j in J}
from data import MyData


def henuristic(M,J,w,p):

    # 对作业按权重降序排序
    sorted_jobs = sorted(J, key=lambda j: w[j], reverse=True)

    # 初始化每台机器的完成时间
    machine_completion = {i: 0 for i in M}

    # 分配作业到机器
    job_assignment = {}
    for j in sorted_jobs:
        # 选择当前能最早完成作业j的机器
        i = min(M, key=lambda i: machine_completion[i] + p[j][i])
        job_assignment[j] = i
        # 更新机器的完成时间
        machine_completion[i] += p[j][i]

    # 计算加权完成时间
    weighted_completion_time = 0
    for j in J:
        i = job_assignment[j]
        weighted_completion_time += w[j] * machine_completion[i]

    return weighted_completion_time

