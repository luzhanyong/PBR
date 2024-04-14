import random
import time
# 设置随机数种子


# # 定义机器和作业集合
# M = list(range(1, 8))  # 7台机器
# J = list(range(1, 41))  # 40个作业
#
# # 随机生成作业权重和处理时间
# w = {j: random.randint(1, 5) for j in J}
# p = {j: {i: random.randint(1, 10) for i in M} for j in J}

def data():
    # M_Type = list(range(1, 13))  # 12台机器
    J_Type = list(range(1, 8))  # 7种作业

    # 权重直接设置为任务的优先级 1 2 3  幸存者检测：3    路径规划：2   灾难检测:1
    # w_J_Type = {1:3,2:3,3:3,4:2,5:2,6:1,7:1}
    w_J_Type = {1: 0.8, 2: 0.7, 3: 0.6, 4: 0.6, 5: 0.5, 6: 0.3, 7: 0.4}

    #
    # p_J_Type_By_M_Type = {1:[12.04,6.20,4.54,9.49,4.75,2.54,2.26,1.88,1.21,2.39,39.65,37.6],2:[476.69,245.47,179.74,375.73,187.865,134.53,89.55,74.47,48.02,94.50,1569.71,1488.66],
    #                       3:[117.95,60.73,44.47,92.97,46.485,24.91,22.15,18.42,11.88,23.38,388.40,368.35],4:[182.60,94.03,68.85,79.71,39.855,21.35,19.00,15.70,10.19,20.05,333.01,315.82],
    #                       5:[182.59,94.03,68.85,143.92,71.96,38.57,34.30,28.53,18.39,36.20,601.26,570.22],6:[186.03,95.79,70.15,146.63,73.32,39.29,34.95,29.06,18.74,36.88,612.60,580.96],
    #                       7:[487.98,251.28,184.00,384.625,192.31,103.07,91.67,76.24,49.16,96.74,96.74,1523.90]}
    #
    #
    # p_J_Type_By_M_Type_int = {1:[12,6,5,9,5,3,2,2,1,2,40,38],2:[477,245,180,376,188,135,90,74,48,95,1570,1489],
    #                       3:[118,61,44,93,46,25,22,18,12,23,388,368],4:[183,94,69,80,40,21,19,16,10,20,333,316],
    #                       5:[183,94,69,144,72,39,34,29,18,36,601,570],6:[186,96,70,147,73,39,35,29,19,37,612,581],
    #                       7:[488,251,184,385,192,103,92,76,49,97,97,1524]}

    p_J_Type_By_M_Type_int_tiny = {1:[2,2,1,1],2:[3,3,2,1],
                          3:[2,2,1,1],4:[3,2,1,1],
                          5:[3,3,2,2],6:[3,2,1,1],
                          7:[3,3,3,2]}



    #从M_J_Type中随机选择m台机器
    # m = 7

    #从p_J_Type_By_M_Type中随机选择n个作业
    # n = 41


    mnum = 6
    # 随机抽取m台机器
    # m = 5  # 你可以根据需要设置m的值
    # M_T = random.sample(M_Type, m)
    M_Type = [1,2,3,4]
    M_T = random.choices(M_Type, k=mnum)


    # M = list(range(1,len(M_T)+1))

    # M= random.sample(M_Type, m)


    # 抽取n次任务，每次可以重复抽取c
    n = 5  # 你可以根据需要设置n的值


    # 抽取任务
    J_T = random.choices(J_Type, k=n)


    # 构建字典w，表示抽取任务的权重
    w = {i+1: w_J_Type[J_T[i]] for i in range(len(J_T))}
    print(M_T)
    # 构建字典p，表示任务在机器上的执行时间
    p = {i+1: {m+1: p_J_Type_By_M_Type_int_tiny[J_T[i]][M_T[m]-1] for m in range(len(M_T))} for i in range(len(J_T))}

    # p = {i+1: {m_: p_J_Type_By_M_Type_int[J_T[i]][m_- 1] for m_ in M} for i in range(len(J_T))}

    print(w)
    print(p)
    print(M_T)
    print(J_T)
    # 重新编号列表J，表示抽取的任务编号
    J = list(range(1, len(J_T)+1))
    M = list(range(1, len(M_T)+1))


    return M,J,w,p


# 计算任务权重


# 计算任务在机器上的执行时间
if __name__ == '__main__':
    data()

