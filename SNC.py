import random
import numpy as np

# 初始化集合 U, J, M
U = {'u1', 'u2', 'u3'}  # 假设有三个组
J = {'j1', 'j2', 'j3'}  # 假设有三个工作
M = {'m1', 'm2'}  # 假设有两台机器

# g 函数将 U 中的每个组映射到 M 中的机器
g = {'u1': 'm1', 'u2': 'm2', 'u3': 'm1'}

# 定义 yuj，这里简单地给每个 uj 分配一个固定的概率值，实际中可能需要根据问题具体定义
yuj = {('u1', 'j1'): 0.2, ('u1', 'j2'): 0.2, ('u1', 'j3'): 0.2,
       ('u2', 'j1'): 0.3, ('u2', 'j2'): 0.3, ('u2', 'j3'): 0.3,
       ('u3', 'j1'): 0.4, ('u3', 'j2'): 0.4, ('u3', 'j3'): 0.4}

# Hcand 和 Hmark 初始化为空的多重图
Hcand = {u: set() for u in U}
Hmark = {u: set() for u in U}
print(Hcand)

# 步骤 1 - 选择和标记候选边
def choose_and_mark_edges(U, J, yuj, g):
    for j in J:
        # 随机选择两个候选组
        u1, u2 = random.sample(U, 2)

        # 添加到 Hcand
        Hcand[u1].add(j)
        Hcand[u2].add(j)

        # 标记边
        for u in [u1, u2]:
            # 计算标记概率
            probability = 1 - np.exp(-2 * yuj[(u, j)]) / (2 * yuj[(u, j)])
            if random.random() <= probability:
                # 如果 u 不支配 j 或者 j 未被标记，则标记 uj
                if g[u] != j and j not in Hmark[u]:
                    Hmark[u].add(j)

    return Hcand, Hmark


# 运行步骤 1
Hcand, Hmark = choose_and_mark_edges(U, J, yuj, g)


# 步骤 2 - 配对边以获得 Hsplit
# Hsplit 初始化为包含所有工作的图
Hsplit = {j: set() for j in J}


# 辅助函数，用于找到所有配对的组合
def find_pairs(edge_list):
    if len(edge_list) % 2 == 1:
        # 如果边的数量是奇数，留下一个未配对
        unpaired = [edge_list.pop()]
    else:
        unpaired = []
    # 将边列表分成两两一对
    paired = list(zip(edge_list[::2], edge_list[1::2]))
    return paired, unpaired


# 实现配对过程
def pair_edges(Hcand, Hmark):
    for u in U:
        # 获取当前组在 Hmark 中的所有边
        marked_edges = list(Hmark[u])
        # 找到所有可能的配对和未配对的边
        pairs, unpaired = find_pairs(marked_edges)

        # 为每一对边添加组的副本到 Hsplit
        for j1, j2 in pairs:
            Hsplit[j1].add(u)
            Hsplit[j2].add(u)

        # 对于未配对的边，也添加组的副本到 Hsplit
        for j in unpaired:
            Hsplit[j].add(u)

    return Hsplit


# 运行步骤 2
Hsplit = pair_edges(Hcand, Hmark)


# 步骤 3 - 将 Hsplit 中的路径和循环分解成段
# 初始化段集合 S
S = set()


# 辅助函数，用于创建长度为 4 或 6 的段
def create_segments(path, is_cycle=False):
    segments = set()
    # 如果是循环并且长度为 4 或 6，整个循环就是一个段
    if is_cycle and len(path) in [4, 6]:
        segments.add(tuple(path))
    else:
        # 否则，从路径中创建长度为 4 或 6 的段
        for i in range(0, len(path) - 3, 2):  # 每次跳过一个顶点创建段
            segment = tuple(path[i:i + 4])
            if len(segment) == 4 and segment not in segments:
                segments.add(segment)
            if i + 5 < len(path):  # 如果可能，创建长度为 6 的段
                segment = tuple(path[i:i + 6])
                if len(segment) == 6 and segment not in segments:
                    segments.add(segment)
    return segments


# 实现分解过程
def break_into_segments(Hsplit):
    # 对 Hsplit 中的每个连通组件处理
    for j, connected_u in Hsplit.items():
        # 这里简单地假设连通组件是路径，实际可能需要检测循环
        path = list(connected_u)
        # 随机决定是使用奇数索引还是偶数索引的组来创建段
        if random.random() < 0.5:
            path = path[::2]  # 奇数索引
        else:
            path = path[1::2]  # 偶数索引

        # 创建段并添加到集合 S
        segments = create_segments(path)
        S.update(segments)

    return S


# 运行步骤 3
S = break_into_segments(Hsplit)

