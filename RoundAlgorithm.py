import numpy as np
import random
import math
import networkx as nx
import time
start_time1 = time.time()
import LP



#一个任务集合J，一个机器集合M。用j∈J  i∈M
J = LP.J
M = LP.M
#矩阵pji表示任务j在机器i上的完成时间
Pji = LP.p  #字典
#任务J的权重
W = LP.w    #字典
#问题的解  三维矩阵
x_matrix = LP.x_matrix
#总任务的最大完成时间
T = LP.T

#定义矩形
class Rectangle:
    def __init__(self, i, j, s, x, pji):
        self.start_time = s
        self.pji = pji
        self.height = x
        self.end_time = s+pji
        self.machine = i
        self.job = j
        self._baseWindows = None

    def __str__(self):
        return f"[作业：{self.job},机器：{self.machine},水平跨度：[{self.start_time},{self.start_time}+{self.pji}],高度：{self.height},基本窗口：{self._baseWindows}]"

    @property
    def baseWindows(self):
        return self._baseWindows

    @baseWindows.setter
    def baseWindows(self,new_baseWindows):
        self._baseWindows = new_baseWindows

#定义小组
#一个机器上的一个基本窗口是一个小组
#不在基本窗口的ij对是一个组
class Group:
    def __init__(self, base_window,machine,job):
        self.base_window = base_window
        self.machine = machine
        self.job = job
        self.members = set()

    def __str__(self):
        if self.base_window != 0:
            return f"[小组u的机器为：{self.machine},基本窗口为:{self.base_window}]"
        if self.base_window == 0:
            return f"[小组u的机器为：{self.machine},作业：{self.job}]"

    def __eq__(self, other):
        if isinstance(other, Group):
            # 判断所有属性是否相等
            return (
                    self.base_window == other.base_window and
                    self.machine == other.machine and
                    self.job == other.job
            )
        return False

    def __hash__(self):
        # 使用属性的散列值来生成对象的哈希值
        return hash((self.base_window, self.machine, self.job))

#定义作业节点，作业需要选择两个候选组 也有可能只有一个候选组的情况




    #矩形就是线
    def add_member(self, rectangle):
        self.members.add(rectangle)

    def display_members(self):
        print("该小组中的矩形包括:")
        for member in self.members:
            print(member)



#打印矩形对象集合
def print_obj(obj_set):
    for obj in obj_set:
        print(obj)

#随机选择ρ ∈ [1, 1+β) 并且 lnρ 均匀分布在 [0, ln(1+β)) ，用 a b c 代表 α β ρ
def generate_c(b):
    while True:
        # 生成均匀分布的随机数 lnc ∈ [0, ln(1+b))
        lnc = random.uniform(0, math.log(1 + b))
        # 计算 c 对应的下界和上界
        # 生成满足 lnc 均匀分布的参数 c
        c = math.exp(lnc)
        if c >=1 and c <(1+b):
            break

    print('参数ρ：',c)
    return c

#计算T时间内的网格点  将网格点代表毫秒 任务在机器上的时间是s
def generate_grid_points(c,b,T):
    # 计算网格点
    grid_points = []
    k = 1
    while True:  # T+1 以包含 T 在内的所有时间步
        point = c * (1 + b) ** k
        if point > T*1000:
            break
        grid_points.append(point)
        k += 1
    print('打印所有网格点:')
    print(grid_points)
    return grid_points

#生成位移参数
def generate_shifting_parameter(Pji):
    SPji = Pji
    for row_key,row_value in Pji.items():
        for col_key,col_value in row_value.items():
            tji = random.uniform(0,col_value)
            SPji[row_key][col_key] = tji

    print('打印位移参数：')
    print(SPji)
    return SPji

#构建矩阵
def generate_rectangle(x_matrix):
    # 定义一个矩形集合
    rectangle_obj_set = set()
    # 遍历三维矩阵，得到找到大于0的元素
    for i in range(x_matrix.shape[0]):
        for j in range(x_matrix.shape[1]):
            for s in range(x_matrix.shape[2]):
                if x_matrix[i,j,s] > 0:
                    #创建矩形对象
                    rectangle_obj = Rectangle(i,j,s,x_matrix[i,j,s],Pji[j][i])
                    rectangle_obj_set.add(rectangle_obj)
    print('打印所有矩形：')
    print_obj(rectangle_obj_set)
    return rectangle_obj_set

#判断一个矩形属于哪些时间间隔，若不属于返回0
def find_grid_point(rectangle,grid_points,SPji):
    rectangle_start_time = rectangle.start_time
    rectangle_end_time = rectangle.end_time
    rectangle_job = rectangle.job
    rectangle_machine = rectangle.machine
    rectangle.baseWindows = 0
    #rectangle_start_time ~ rectangle_end_time 是否存在网格点
    for i, _ in enumerate(grid_points):
        if rectangle_start_time*1000 <= grid_points[i] < rectangle_start_time*1000 + SPji[rectangle_job][rectangle_machine]*1000 <= grid_points[i+1]:
            rectangle.baseWindows = i+1+1


#构建所有组的集合
def generate_group(rectangles):
    #定义一个空集合组
    group_set = set()
    group_set_rec = set()
    #一个机器里的一个基本窗口
    for i in M:
        for rectangle in rectangles:
            # 有基本窗口的矩形
            if rectangle.machine == i and rectangle.baseWindows != 0:
                is_exist = 0
                for group in group_set_rec:
                    if group.machine == i and group.base_window == rectangle.baseWindows:
                        is_exist = 1
                        #存在，则将矩形放入group中
                        group.add_member(rectangle)
                if is_exist == 0:
                    group_set.add(Group(rectangle.baseWindows,i,0))
                    #创建组，并将矩形放入组中
                    group = Group(rectangle.baseWindows, i, 0)
                    group.add_member(rectangle)
                    group_set_rec.add(group)
            # 没有基本窗口的矩形
            if rectangle.machine == i and rectangle.baseWindows == 0:
                is_exist = 0
                for group in group_set_rec:
                    if group.machine == i and group.job == rectangle.job:
                        is_exist = 1
                        # 存在，则将矩形放入group中
                        group.add_member(rectangle)
                if is_exist == 0:
                    group_set.add(Group(0,i,rectangle.job))
                    # 创建组，并将矩形放入组中
                    group = Group(0,i,rectangle.job)
                    group.add_member(rectangle)
                    group_set_rec.add(group)
    return group_set,group_set_rec


#构建yuj映射，给出组和作业，得到相应的概率
def f_yuj(group,job,group_set_rec):
    yuj = 0
    for group_rec in group_set_rec:
        if group.machine == group_rec.machine and group.base_window == group_rec.base_window and group.job == group_rec.job :
            #遍历这个组下面的所有矩形
            for rectangle in group_rec.members:
                if rectangle.job == job:
                    yuj += rectangle.height
    return yuj





#判断两个矩阵是否相同
def is_equal_rectangle(rectangleA,rectangleB):
    if rectangleA.job == rectangleB.job and rectangleA.machine == rectangleB.machine and rectangleA.start_time == rectangleB.start_time and rectangleA.height == rectangleB.height and rectangleA.end_time == rectangleB.end_time:
        return True
    else:
        return False

#判断作业是否在矩形集合中
def not_in_rectangle_set(job,rectangle_set):
    flag = True
    max_rectangle = None
    for rectangle in rectangle_set:
        if rectangle.job == job:
            flag = False
            max_rectangle = rectangle
    return flag, max_rectangle

#找到作业属于哪个组
def find_job_in_group(job,group_set):
    group_list_result = []
    for group in group_set:
        flag,_ = not_in_rectangle_set(job,group.members)
        if flag == False:
            group_list_result.append(group)
    return group_list_result




#在矩形集合，删除一个矩形，再添加一个矩形
def reomve_add_rectangle_set(remove_rectangle,add_rectangle,rectangle_set):
    for target_rectangle in rectangle_set:
        if is_equal_rectangle(target_rectangle,remove_rectangle):
            rectangle_set.pop(target_rectangle)
            rectangle_set.add(add_rectangle)
            break



#为每个组u的作业j选择一个锚点矩形
def generate_grouop_Rijs(group_set_rec):
    group_set_Rijs = set()
    for group in group_set_rec:
        max_rectangle_map = {}
        for rectangle in group.members:
            max_rectangle = max_rectangle_map.get(rectangle.job)
            if max_rectangle == None or (max_rectangle != None and rectangle.height > max_rectangle.height):
                max_rectangle_map[rectangle.job] = rectangle
        #将矩形集合封装成组
        max_group = Group(group.base_window,group.machine,group.job)
        for job, rectangle in max_rectangle_map.items():
            max_group.add_member(rectangle)
        group_set_Rijs.add(max_group)
    return group_set_Rijs

#确定只有属于一个组的作业
def calssfly_job(group_set_Rijs):
    single_job = []
    mult_job = []
    for job in J:
        group_list_result = find_job_in_group(job,group_set_Rijs)
        if len(group_list_result) == 1:
            single_job.append(job)
        else:
            mult_job.append(job)
    return single_job,mult_job

#得到小组 在 group_list_Rijs中的下标
def get_group_list_Rijs_index(target_group,group_list_Rijs):
    for i,group in enumerate(group_list_Rijs):
        if group == target_group:
            return i


#确定小组节点 小组节点使用：group_list_Rijs中的下标
def define_group(mult_job,group_list_Rijs):
    group_node_list = []
    for job in mult_job:
        group_list_result = find_job_in_group(job,group_list_Rijs)
        for group in group_list_result:
            if group not in group_node_list:
                group_node_list.append(get_group_list_Rijs_index(group,group_list_Rijs))
    return group_node_list


#根据概率对小组进行排序
def sort_by_yuj(job,group_list,group_set_Rijs):
    n = len(group_list)
    for i in range(n - 1):
        for j in range(0,n-i-1):
            if f_yuj(group_list[j],job,group_set_Rijs) < f_yuj(group_list[j+1],job,group_set_Rijs):
                temp = group_list[j]
                group_list[j] = group_list[j+1]
                group_list[j+1] = temp
    return group_list



#只属于一个组的作业调度映射
def g_single_job(single_job,group_set_Rijs):
    g_single_job_map = {}
    for job in single_job:
        group_list_result = find_job_in_group(job,group_set_Rijs)
        g_single_job_map[job] = group_list_result[0]
    return g_single_job_map


#配对
def pair_edges(Hcand):
    Hsplit = nx.Graph()  # 对于 Hsplit 使用普通图是因为它不包含多重边

    for u in Hcand.nodes():
        # 获取与 u 相连的所有工作（边）
        edges = list(Hcand.edges(u))
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


#生成段
def break_into_segments(Hsplit):
    # 这个步骤需要知道如何识别 Hsplit 中的每个连通组件，并将其分解成段
    # 这里我们将简单地将每个连通组件作为一个段
    segments = []
    for component in nx.connected_components(Hsplit):
        segments.append(component)
    return segments

#分组
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

#找到作业在组中的锚点矩形
def find_Rijs_job_in_group(job,g):
    group = g[job]
    Rijs = None
    for xijs in group.members:
        if xijs.job == job:
            Rijs = xijs
    return Rijs

#判断机器是否支配作业j
def is_domainate(job,g):
    #找到属于机器i的所有组
    group = g[job]
    yuj = 0
    for Rijs in group.members:
        if Rijs.job == job:
            yuj = Rijs.height
    if yuj > 1/2:
        return True
    else:
        return False


#计算θj 构建调度表  θj使用cj表示
def compute_cj(job,g):
    if is_domainate(job,g):
        cj = (1+a) * find_Rijs_job_in_group(job,g).start_time + SPji[job][g[job].machine] + 0.2 * find_Rijs_job_in_group(job,g).pji
    else:
        cj = (1 + a) * find_Rijs_job_in_group(job, g).start_time + SPji[job][g[job].machine]
    return cj

#构建调度表 机器i上的调度任务，并根据cj进行降序排列
def generate_jobs_in_machine(g):
    jobs_in_machine_map = {}
    for machine in M:
        Rijs_list = []
        for job , group in g.items():
            if group.machine == machine:
                Rijs_list.append(find_Rijs_job_in_group(job,g))
        jobs_in_machine_map[machine] = Rijs_list
    return jobs_in_machine_map

#根据cj进行升序序排列jobs_in_machine
def sort_ac_Rijs_list_by_cj(Rijs_list,g):
    n = len(Rijs_list)

    for i in range(n-1):
        for j in range(0,n - i - 1):
            if(compute_cj(Rijs_list[j].job,g) > compute_cj(Rijs_list[j + 1].job,g)):
                temp = Rijs_list[j]
                Rijs_list[j + 1] = Rijs_list[j]
                Rijs_list[j] = temp

def sort_ac_job_in_machine_map_by_cj(jobs_in_machine_map,g):
    for Rijs_list in jobs_in_machine_map.values():
        sort_ac_Rijs_list_by_cj(Rijs_list,g)
    return jobs_in_machine_map





#基于LP矩形的调度算法
#input ：通过线性规划得到的一个实数解xijs
#output：将实数解舍入为整数解的Rijs 并给出机器i调度作业j的序列

xijs = []

#步骤一：定义参数α β ρ  定义基本窗口  定义位移参数tij，为了定义每个矩形属于哪个基本窗口  定义矩形不属于基本窗口的存在
#定义 α=0.3  β=12.1 随机选择ρ ∈ [1, 1+β) 并且 lnρ 均匀分布在 [0, ln(1+β)) ，用 a b c 代表 α β ρ
a = 0.3
b = 12.1
c = generate_c(b)

#定义网格点为一个真实数gp=ρ(1+β)^k k属于整数 因此基本窗口K为(ρ(1 + β)^k−1, ρ(1 + β)^k]
grid_points = generate_grid_points(c,b,T)

#得到矩形集合
rectangles = generate_rectangle(x_matrix)

#对于每一对ij，定义一个位移参数tij∈[0-pij)，矩形属于基本窗口k的定义：s≤ ρ(1 + β)^k−1 ＜ s+tij ≤ ρ(1 + β)^k   #矩形不属于基本窗口的情况：在s 和 s+tij中没有网格点
SPji = generate_shifting_parameter(Pji)

#为每个矩形找到所在基本窗口
for rectangle in rectangles:
    find_grid_point(rectangle,grid_points,SPji)
print_obj(rectangles)


#步骤二：应用SNC：定义组群U  ，定义U->M的映射 ，定义分数概率y
#定义组群 对于每台机器i的基本窗口k为·   一个组，对于不在基本窗口的ij为一组    得到一个组群U
group_set,group_set_rec = generate_group(rectangles)
# print_obj(group_set)
for group in group_set_rec:
    print(group)
    group.display_members()


#定义分数分配：yuj = 一个组中所有xijs的总高度
# yuj = 一个组中所有关于作业j的矩阵高度的和
#而对于映射U->M能直接得到
#定义支配作业的条件：一个组内作业j的xijs总高度（xij）大于1/2，表示机器对作业有足够的支配。

#应用SNC：
# 分组： 一个基本窗口是一组，定义的位移参数tij可以进行分组
#步骤一 对于每个作业随机选择两个侯选组 vj1 vj2 且 vj1 不等于 vj2 且每个组u被选择的概率为2yju 选择候选组要选择yuj大的候选组，尽量避免vj1 和 vj2 指向不同的机器
#为一个组u中的一个作业选择一个锚点矩阵
print("选择锚点矩阵之后的组0----------------------------------------------------------------------")
group_set_Rijs = generate_grouop_Rijs(group_set_rec)
for group in group_set_Rijs:
    print(group)
    group.display_members()

#将集合转化为列表 应用SNC时，就使用下标来代替组
group_list_Rijs = list(group_set_Rijs)
#访问第1的元素
single_job , mult_job = calssfly_job(group_set_Rijs)
g_single_job_map=g_single_job(single_job,group_set_Rijs)

print("只属于一个组的作业")
print(single_job)
print("属于多个组的作业")
print(mult_job)
print("得到组节点")
group_node_list = define_group(mult_job, group_list_Rijs)
print(group_node_list)
print("共有几组：",len(group_list_Rijs))

#为属于多个小组的作业选择两个候选边
Hcand = nx.MultiGraph()
Hmark = nx.MultiGraph()

for u in group_node_list:
    # 将所有的小组和工作添加到图中
    Hcand.add_node(u)
for j in mult_job:
    # 所有的小组和工作添加到图中
    Hcand.add_node(j)


for job in mult_job:
    #找到作业属于哪组，并根据概率对小组进行排序
    group_list_result = find_job_in_group(job,group_set_Rijs)
    sort_group_list_result = sort_by_yuj(job, group_list_result, group_set_Rijs)
    #选择概率高的前两组为候选边
    # 为每个小组添加两个工作为候选边
    Hcand.add_edge(sort_group_list_result[0], job)
    Hcand.add_edge(sort_group_list_result[1], job)




#步骤二：对选择组u的边进行配对，配对是随机均匀选择的，创建多个u的副本，u和u的副本分别拥有一个配对 最终得到一个度为2的图，该图是多个循环的并集。对于未成功配对的边（因为边可能是奇数），我们新加一个组u，将边连接u并配对
Hsplit = pair_edges(Hcand)

#步骤三：假设每个循环是4的倍数，我们取路径长度4为一段，段的结尾是组u，结构为：u-j-u‘-j’-u‘’，其中 u u' u‘’ 属于U  j j' 属于J
segmenrts = break_into_segments(Hsplit)

#步骤四：以1/2的概率   j选择u   j‘选择u’   以剩余1/2的概率     j选择u'    j‘选择u’‘

sigma = rounding_along_segments(segmenrts,Hcand)
for job in mult_job:
    print(f"作业{job}被分配到{sigma[job]}组中")
print("111111111111111111111111111111111111111111111111111")
for job in single_job:
    print(f"作业{job}被分配到{g_single_job_map[job]}组中")

# 使用字典解包合并两个字典
g_temp = {**g_single_job_map, **sigma}
g = {}
for job in J:
    g[job] = g_temp[job]

print("合并后的-----------------------------------------------------------------------------")
for job in J:
    print(f"作业{job}被分配到{g[job]}组中")

group = g[1]
print(group)
print_obj(group.members)

#步骤三：
#为每个作业定义θj
#在作业j上按照θj递增的顺序进行调度
cj = compute_cj(1,g)
print(cj)

print("机器i上需要调度的任务---------------------")
jobs_in_machine_map = generate_jobs_in_machine(g)

for i in M:
    print(f"机器{i}上的矩形有")
    print_obj(jobs_in_machine_map[i])


jobs_in_machine_map_ac = sort_ac_job_in_machine_map_by_cj(jobs_in_machine_map,g)
print("排序后的调度任务-----------------------------------------")
for i in M:
    print(f"机器{i}上的矩形有")
    print_obj(jobs_in_machine_map_ac[i])





weight_sum_time = 0
#
# ## 根据调度表，计算加权完成时间
for machine in M:
    Rijs_ac = jobs_in_machine_map_ac[machine]
    start_time = 0
    for Rijs in Rijs_ac:
        weight_sum_time += W[Rijs.job] * (start_time + Rijs.pji)
        start_time += Rijs.pji


print(f"近似算法得到的加权完成时间为{weight_sum_time}")





# 记录结束时间
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time1

# print(f"程序执行时间：{execution_time} 秒")




