import random

# 设置随机数种子，以便结果可复现
# random.seed(0)

# 机器和作业的集合
M = list(range(1, 8))  # 7台机器
J = list(range(1, 41))  # 40个作业

# 随机生成作业权重和处理时间
w = {j: random.randint(1, 5) for j in J}
p = {j: {i: random.randint(1, 10) for i in M} for j in J}

# 随机分配作业到机器上
random.shuffle(J)  # 随机打乱作业顺序
job_assignment = {j: random.choice(M) for j in J}  # 随机分配作业到机器

# 计算加权完成时间
# 对于每台机器，计算该机器上所有作业的完成时间，并乘以它们的权重
weighted_completion_time = 0
for i in M:
    completion_time = 0
    for j in J:
        if job_assignment[j] == i:
            completion_time += p[j][i]  # 累加机器上作业的处理时间
            weighted_completion_time += w[j] * completion_time  # 加权完成时间

print()
print(f"随机分配算法得到的加权完成时间为{weighted_completion_time}")

