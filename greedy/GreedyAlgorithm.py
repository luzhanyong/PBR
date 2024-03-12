# 反转贪心算法分配作业到机器
# 启发式方法：优先考虑权重最高的作业，并将其分配给当前完成时间最长的机器

import random

# # 设置随机数种子，以便结果可复现
# # random.seed(0)
#
# # 机器和作业的集合
# M = list(range(1, 8))  # 7台机器
# J = list(range(1, 41))  # 40个作业
#
# # 随机生成作业权重和处理时间
# w = {j: random.randint(1, 5) for j in J}
# p = {j: {i: random.randint(1, 10) for i in M} for j in J}
from data import MyData

import time


def greedy(M,J,w,p):

    # 每次选择当前完成时间最短的机器，而不是根据比率进行排序

    # 贪心算法分配作业到机器
    # 按照权重/处理时间的比率进行升序排序

    # 优化后的贪心算法分配作业到机器
    # 按照权重/处理时间的比率进行升序排序，并优化机器选择过程

    # 计算每个作业的权重/处理时间比率，并取最大值
    job_ratios = {j: w[j] / max(p[j].values()) for j in J}

    # 根据比率进行升序排序
    sorted_jobs = sorted(job_ratios, key=job_ratios.get)

    # 初始化每台机器的完成时间
    machine_completion = {i: 0 for i in M}

    # 分配作业到机器
    job_assignment = {}
    for j in sorted_jobs:
        # 选择能最早完成作业 j 的机器
        i = min(M, key=lambda m: machine_completion[m] + p[j][m])
        job_assignment[j] = i
        # 更新机器的完成时间
        machine_completion[i] += p[j][i]

    # 计算加权完成时间
    weighted_completion_time = 0
    for j in J:
        i = job_assignment[j]
        weighted_completion_time += w[j] * machine_completion[i]

    return weighted_completion_time
