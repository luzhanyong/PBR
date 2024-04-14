import random
from data import MyData
# 设置随机数种子，以便结果可复现
# random.seed(0)

def random_allo(M,J,w,p):

    # 随机分配作业到机器上

    # 计算加权完成时间
    # 对于每台机器，计算该机器上所有作业的完成时间，并乘以它们的权重
    weighted_completion_time = 0
    completion_time = 0

    print(p)

    n = len(J)
    # 遍历所有数组元素
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if w[J[j]] < w[J[j+1] ]:
                J[j], J[j + 1] = J[j + 1], J[j]

    print(J)
    for j in J:
        completion_time += p[j][1]  # 累加机器上作业的处理时间
        weighted_completion_time += w[j] * completion_time  # 加权完成时间

    return weighted_completion_time, completion_time


if __name__ == '__main__':
    M0, J0, w0, p0, JL, ML = MyData.data()
    random_allo(M0, J0, w0, p0)