# import random
# import math
#
# def generate_grid_points(ρ, β, num_points):
#     grid_points = []
#     for k in range(1, num_points + 1):  # 从 k=1 开始，生成指定数量的网格点
#         point = ρ * (1 + β) ** k
#         grid_points.append(point)
#     return grid_points
#
# # 定义参数
# β = 12.1
#
# # 随机选择 ρ，使得 ln ρ 均匀分布在 [0, ln(1+β))
# ρ = math.exp(random.uniform(0, math.log(1 + β)))
#
# # 生成网格点
# num_points = 10  # 作为示例生成 10 个网格点
# result = generate_grid_points(ρ, β, num_points)
#
# # 打印结果
# print("随机选择的 ρ:", ρ)
# print("生成的网格点:", result)


a = [1,11,111,1,11,1]
for i,_ in enumerate(a):
    print(a[i])

a[3]