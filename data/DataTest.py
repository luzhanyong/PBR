import numpy as np

# 给定任务的执行时间字典 p
p = {
    1: [10, 15, 20, 25, 30],
    2: [8, 12, 16, 20, 24],
    # ... 可以根据实际情况添加更多机器和对应的任务执行时间
}

# 设置泊松分布的参数（平均任务数）
lambda_param = 2

# 生成泊松分布样本
poisson_samples = np.random.poisson(lambda_param, size=30)

# 随机选择4台不重复的机器
selected_machines = np.random.choice(list(p.keys()), size=4, replace=False)

# 计算每台机器应该选择的任务数量
tasks_per_machine = len(p) * [30 // len(p)]
remainder = 30 % len(p)
for i in range(remainder):
    tasks_per_machine[i] += 1

# 随机选择任务执行时间，可以重复选择
selected_execution_times = [
    (machine, np.random.choice(p[machine])) for machine, num_tasks in zip(selected_machines, tasks_per_machine) for _ in range(num_tasks)
]

# 打印结果
print("泊松分布样本:", poisson_samples)
print("随机选择的机器:", selected_machines)
print("随机选择的任务执行时间:", selected_execution_times)
