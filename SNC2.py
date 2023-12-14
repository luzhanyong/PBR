import networkx as nx
import numpy as np
import random
import RoundAlgorithm

# 假设 f_yuj 函数已经定义
def f_yuj(group,job,group_set_rec):
    yuj = 0
    for group_rec in group_set_rec:
        if group.machine == group_rec.machine and group.base_window == group_rec.base_window and group.job == group_rec.job :
            #遍历这个组下面的所有矩形
            for rectangle in group_rec.members:
                if rectangle.job == job:
                    yuj += rectangle.height
    return yuj

# 步骤 1 - 选择和标记候选边
def choose_and_mark_edges(U, J, f_yuj):
    Hcand = nx.MultiGraph()
    Hmark = nx.MultiGraph()

    for u in U:
        for j in J:
            # 将所有的小组和工作添加到图中
            Hcand.add_node(u)
            Hcand.add_node(j)
            Hmark.add_node(u)
            Hmark.add_node(j)

        # 获取每个组对应所有工作 j 的概率并排序
        probabilities = [(j, f_yuj(u, j, U)) for j in J]
        # 选择最大的两个概率
        probabilities.sort(key=lambda x: x[1], reverse=True)
        jobs = [p[0] for p in probabilities[:2]]  # 取最大的两个工作

        # 为每个小组添加两个工作为候选边
        for job in jobs:
            Hcand.add_edge(u, job)
            # 计算标记概率并决定是否标记该边
            prob_mark = 1 - np.exp(-2 * f_yuj(u, job, U)) / (2 * f_yuj(u, job, U))
            if random.random() < prob_mark:
                Hmark.add_edge(u, job)

    return Hcand, Hmark

# 步骤 2 - 配对边以获得 Hsplit
def pair_edges(Hcand, Hmark):
    Hsplit = nx.Graph()  # 对于 Hsplit 使用普通图是因为它不包含多重边

    for u in Hcand.nodes():
        # 获取与 u 相连的所有工作（边）
        edges = list(Hmark.edges(u))
        random.shuffle(edges)  # 打乱顺序以随机配对

        # 配对工作
        for i in range(0, len(edges), 2):
            if i + 1 < len(edges):
                # 将两个工作作为一个路径添加到 Hsplit
                job1 = edges[i][1]
                job2 = edges[i+1][1]
                Hsplit.add_node(u)
                Hsplit.add_edge(u, job1)
                Hsplit.add_edge(u, job2)

    return Hsplit

# 步骤 3 - 将 Hsplit 中的路径和循环分解成段
def break_into_segments(Hsplit):
    # 这个步骤需要知道如何识别 Hsplit 中的每个连通组件，并将其分解成段
    # 这里我们将简单地将每个连通组件作为一个段
    segments = []
    for component in nx.connected_components(Hsplit):
        segments.append(component)
    return segments

# 步骤 4 - 沿着段进行四舍五入
def rounding_along_segments(segments, Hcand):
    sigma = {}  # 结果映射

    for segment in segments:
        for job in segment:
            if Hcand.has_node(job):
                # 选择与 job 相连的一个小组
                groups = list(Hcand.adj[job])
                chosen_group = random.choice(groups)
                sigma[job] = chosen_group

    return sigma

# 示例数据
U = ['u1', 'u2', 'u3', 'u4']
J = ['j1', 'j2', 'j3', 'j4']
# Hcand 和 Hmark 的创建依赖于 f_yuj 函数的实现
Hcand, Hmark = choose_and_mark_edges(U, J, f_yuj)
Hsplit = pair_edges(Hcand, Hmark)
segments = break_into_segments(Hsplit)
sigma = rounding_along_segments(segments, Hcand)
