import numpy as np
import random
import math
import networkx as nx


from branchAndBound import bb

from LP import LPold

from data import MyData





J = []
M = []

Pji = {}

W = {}

T = 0

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
        return f"[task：{self.job},machine：{self.machine},line：[{self.start_time},{self.start_time}+{self.pji}],hight：{self.height},base windows：{self._baseWindows}]"

    @property
    def baseWindows(self):
        return self._baseWindows

    @baseWindows.setter
    def baseWindows(self,new_baseWindows):
        self._baseWindows = new_baseWindows


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
            return (
                    self.base_window == other.base_window and
                    self.machine == other.machine and
                    self.job == other.job
            )
        return False

    def __hash__(self):
        return hash((self.base_window, self.machine, self.job))


    def add_member(self, rectangle):
        self.members.add(rectangle)

    def display_members(self):
        print("该小组中的矩形包括:")
        for member in self.members:
            print(member)



def print_obj(obj_set):
    for obj in obj_set:
        print(obj)


def generate_c(b):
    while True:
        lnc = random.uniform(0, math.log(1 + b))
        c = math.exp(lnc)
        if c >=1 and c <(1+b):
            break


    return c

def generate_grid_points(c,b,T):
    grid_points = []
    k = 1
    while True:
        point = c * (1 + b) ** k
        if point > T*1000:
            break
        grid_points.append(point)
        k += 1
    return grid_points

def generate_shifting_parameter(Pji):
    SPji = Pji
    for row_key,row_value in Pji.items():
        for col_key,col_value in row_value.items():
            tji = random.uniform(0,col_value)
            SPji[row_key][col_key] = tji
    return SPji

def generate_rectangle(x_matrix):
    rectangle_obj_set = set()
    for i in range(x_matrix.shape[0]):
        for j in range(x_matrix.shape[1]):
            for s in range(x_matrix.shape[2]):
                if x_matrix[i,j,s] > 0:
                    rectangle_obj = Rectangle(i,j,s,x_matrix[i,j,s],Pji[j][i])
                    rectangle_obj_set.add(rectangle_obj)
    return rectangle_obj_set


def find_grid_point(rectangle,grid_points,SPji):
    rectangle_start_time = rectangle.start_time
    rectangle_end_time = rectangle.end_time
    rectangle_job = rectangle.job
    rectangle_machine = rectangle.machine
    rectangle.baseWindows = 0
    for i, _ in enumerate(grid_points):
        if rectangle_start_time*10 <= grid_points[i] < rectangle_start_time*10 + SPji[rectangle_job][rectangle_machine]*10 <= grid_points[i+1]:
            rectangle.baseWindows = i+1+1


def generate_group(rectangles):
    group_set = set()
    group_set_rec = set()

    for i in M:
        for rectangle in rectangles:
            if rectangle.machine == i and rectangle.baseWindows != 0:
                is_exist = 0
                for group in group_set_rec:
                    if group.machine == i and group.base_window == rectangle.baseWindows:
                        is_exist = 1
                        group.add_member(rectangle)
                if is_exist == 0:
                    group_set.add(Group(rectangle.baseWindows,i,0))

                    group = Group(rectangle.baseWindows, i, 0)
                    group.add_member(rectangle)
                    group_set_rec.add(group)
            if rectangle.machine == i and rectangle.baseWindows == 0:
                is_exist = 0
                for group in group_set_rec:
                    if group.machine == i and group.job == rectangle.job:
                        is_exist = 1

                        group.add_member(rectangle)
                if is_exist == 0:
                    group_set.add(Group(0,i,rectangle.job))

                    group = Group(0,i,rectangle.job)
                    group.add_member(rectangle)
                    group_set_rec.add(group)
    return group_set,group_set_rec


def f_yuj(group,job,group_set_rec):
    yuj = 0
    for group_rec in group_set_rec:
        if group.machine == group_rec.machine and group.base_window == group_rec.base_window and group.job == group_rec.job :

            for rectangle in group_rec.members:
                if rectangle.job == job:
                    yuj += rectangle.height
    return yuj






def is_equal_rectangle(rectangleA,rectangleB):
    if rectangleA.job == rectangleB.job and rectangleA.machine == rectangleB.machine and rectangleA.start_time == rectangleB.start_time and rectangleA.height == rectangleB.height and rectangleA.end_time == rectangleB.end_time:
        return True
    else:
        return False

def not_in_rectangle_set(job,rectangle_set):
    flag = True
    max_rectangle = None
    for rectangle in rectangle_set:
        if rectangle.job == job:
            flag = False
            max_rectangle = rectangle
    return flag, max_rectangle


def find_job_in_group(job,group_set):
    group_list_result = []
    for group in group_set:
        flag,_ = not_in_rectangle_set(job,group.members)
        if flag == False:
            group_list_result.append(group)
    return group_list_result




def reomve_add_rectangle_set(remove_rectangle,add_rectangle,rectangle_set):
    for target_rectangle in rectangle_set:
        if is_equal_rectangle(target_rectangle,remove_rectangle):
            rectangle_set.pop(target_rectangle)
            rectangle_set.add(add_rectangle)
            break




def generate_grouop_Rijs(group_set_rec):
    group_set_Rijs = set()
    for group in group_set_rec:
        max_rectangle_map = {}
        for rectangle in group.members:
            max_rectangle = max_rectangle_map.get(rectangle.job)
            if max_rectangle == None or (max_rectangle != None and rectangle.height > max_rectangle.height):
                max_rectangle_map[rectangle.job] = rectangle
        max_group = Group(group.base_window,group.machine,group.job)
        for job, rectangle in max_rectangle_map.items():
            max_group.add_member(rectangle)
        group_set_Rijs.add(max_group)
    return group_set_Rijs

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

def get_group_list_Rijs_index(target_group,group_list_Rijs):
    for i,group in enumerate(group_list_Rijs):
        if group == target_group:
            return i



def define_group(mult_job,group_list_Rijs):
    group_node_list = []
    for job in mult_job:
        group_list_result = find_job_in_group(job,group_list_Rijs)
        for group in group_list_result:
            if group not in group_node_list:
                group_node_list.append(get_group_list_Rijs_index(group,group_list_Rijs))
    return group_node_list



def sort_by_yuj(job,group_list,group_set_Rijs):
    n = len(group_list)
    for i in range(n - 1):
        for j in range(0,n-i-1):
            if f_yuj(group_list[j],job,group_set_Rijs) < f_yuj(group_list[j+1],job,group_set_Rijs):
                temp = group_list[j]
                group_list[j] = group_list[j+1]
                group_list[j+1] = temp
    return group_list



def g_single_job(single_job,group_set_Rijs):
    g_single_job_map = {}
    for job in single_job:
        group_list_result = find_job_in_group(job,group_set_Rijs)
        g_single_job_map[job] = group_list_result[0]
    return g_single_job_map



def pair_edges(Hcand):
    Hsplit = nx.Graph()

    for u in Hcand.nodes():

        edges = list(Hcand.edges(u))
        random.shuffle(edges)

        for i in range(0, len(edges), 2):
            if i + 1 < len(edges):

                job1 = edges[i][1]
                job2 = edges[i+1][1]
                Hsplit.add_node(u)
                Hsplit.add_edge(u, job1)
                Hsplit.add_edge(u, job2)

    return Hsplit


def break_into_segments(Hsplit):

    segments = []
    for component in nx.connected_components(Hsplit):
        segments.append(component)
    return segments


def rounding_along_segments(segments, Hcand):
    sigma = {}

    for segment in segments:
        for job in segment:
            if Hcand.has_node(job):
                groups = list(Hcand.adj[job])
                chosen_group = random.choice(groups)
                sigma[job] = chosen_group

    return sigma


def find_Rijs_job_in_group(job,g):
    group = g[job]
    Rijs = None
    for xijs in group.members:
        if xijs.job == job:
            Rijs = xijs
    return Rijs


def is_domainate(job,g):
    group = g[job]
    yuj = 0
    for Rijs in group.members:
        if Rijs.job == job:
            yuj = Rijs.height
    if yuj > 1/2:
        return True
    else:
        return False


def compute_cj(job,g,a,SPji):
    if is_domainate(job,g):
        cj = (1 + a) * find_Rijs_job_in_group(job,g).start_time + SPji[job][g[job].machine] + 0.2 * find_Rijs_job_in_group(job,g).pji
    else:
        cj = (1 + a) * find_Rijs_job_in_group(job, g).start_time + SPji[job][g[job].machine]
    return cj

def generate_jobs_in_machine(g):
    jobs_in_machine_map = {}
    for machine in M:
        Rijs_list = []
        for job , group in g.items():
            if group.machine == machine:
                Rijs_list.append(find_Rijs_job_in_group(job,g))
        jobs_in_machine_map[machine] = Rijs_list
    return jobs_in_machine_map

def sort_ac_Rijs_list_by_cj(Rijs_list,g,a,SPji):
    n = len(Rijs_list)
    for i in range(n):
        for j in range(0, n - i - 1):
            if(compute_cj(Rijs_list[j].job,g,a,SPji) > compute_cj(Rijs_list[j + 1].job,g,a,SPji)):
                Rijs_list[j], Rijs_list[j+1] = Rijs_list[j+1], Rijs_list[j]

def sort_ac_job_in_machine_map_by_cj(jobs_in_machine_map,g,a,SPji):
    for machine in M:
        sort_ac_Rijs_list_by_cj(jobs_in_machine_map[machine],g,a,SPji)
    return jobs_in_machine_map



def round(x_matrix,JL,ML):


    xijs = []


    a = 0.3
    b = 12.1
    c = generate_c(b)


    grid_points = generate_grid_points(c,b,T)


    rectangles = generate_rectangle(x_matrix)

    SPji = generate_shifting_parameter(Pji)


    for rectangle in rectangles:
        find_grid_point(rectangle,grid_points,SPji)

    group_set,group_set_rec = generate_group(rectangles)

    group_set_Rijs = generate_grouop_Rijs(group_set_rec)


    group_list_Rijs = list(group_set_Rijs)

    single_job , mult_job = calssfly_job(group_set_Rijs)
    g_single_job_map=g_single_job(single_job,group_set_Rijs)


    group_node_list = define_group(mult_job, group_list_Rijs)


    Hcand = nx.MultiGraph()
    Hmark = nx.MultiGraph()

    for u in group_node_list:
        Hcand.add_node(u)
    for j in mult_job:
        Hcand.add_node(j)


    for job in mult_job:
        group_list_result = find_job_in_group(job,group_set_Rijs)
        sort_group_list_result = sort_by_yuj(job, group_list_result, group_set_Rijs)
        Hcand.add_edge(sort_group_list_result[0], job)
        Hcand.add_edge(sort_group_list_result[1], job)




    Hsplit = pair_edges(Hcand)

    segmenrts = break_into_segments(Hsplit)

    sigma = rounding_along_segments(segmenrts,Hcand)



    g_temp = {**g_single_job_map, **sigma}
    g = {}
    for job in J:
        g[job] = g_temp[job]


    group = g[1]

    cj = compute_cj(1,g,a,SPji)

    jobs_in_machine_map = generate_jobs_in_machine(g)



    jobs_in_machine_map_ac = sort_ac_job_in_machine_map_by_cj(jobs_in_machine_map,g,a,SPji)




    weight_sum_time = 0

    licp_list = []
    licp_sum = 0
    time_sum = 0
    lcp_up = 0
    lcp = 0
    for machine in M:
        Rijs_list = jobs_in_machine_map_ac[machine]
        start_time = 0
        licp = 0
        for Rijs in Rijs_list:
            weight_sum_time += W[Rijs.job] * (start_time + Rijs.pji)
            start_time += Rijs.pji
            licp += JL[Rijs.job]
            time_sum += Rijs.pji
        licp_sum += licp
        licp_list.append(licp)

    licp_avg = licp_sum / len(M)

    lm_sum = 0
    for i, element in enumerate(licp_list):
        lm = element / ML[i+1]
        lm_sum += lm

    for l in licp_list:
        lcp_up += pow(l - licp_avg, 2)
    lcp = lcp_up / len(M)
    print(f'rounding licp_avg  {licp_avg}')
    return weight_sum_time,lcp,time_sum,lm_sum






def avg_process_time():

    weight_time_bb_sum = 0

    weight_time_access_sum = 0
    for i in range(30):

        M0, J0, w0, p0,JL,ML = MyData.data()

        M1, J1, w1, p1, x_matrix, T1, optimal_value1,solved_time1 = LPold.process_LP(M0, J0, w0, p0)

        global_LB = bb.bb_optimization(M1, J1, w1, p1)

        weight_time_bb_sum += global_LB

        are_all_integers = np.all((x_matrix > 0.99) | (x_matrix == 0))

        if are_all_integers:
            weight_time1 = optimal_value1

            licp_list = []

            time_sum_round = 0
            licp_avg = 0
            licp_sum = 0
            lcp_up = 0
            for i in M0:
                licp = 0
                for j in J0:
                    for t in range(T1+1):
                        if x_matrix[i,j,t] > 0.99:
                            time_sum_round += p0[j][i]
                            licp += JL[j]
                licp_list.append(licp)
                licp_sum += licp
            licp_avg = licp_sum / len(M0)

            lm_sum_rounding = 0
            for i, element in enumerate(licp_list):
                lm = element / ML[i+1]
                lm_sum_rounding += lm

            for l in licp_list:
                lcp_up += pow(l - licp_avg,2)
            lcp_round = lcp_up / len(M0)
            print(licp_list)
            print(f'rounding licp_avg  {licp_avg}')

        else:
            global J, M, Pji, W, T
            J = J1
            M = M1
            Pji = p1
            W = w1
            T = T1
            weight_time1,lcp_round,time_sum_round,lm_sum_rounding = round(x_matrix,JL,ML)


if __name__ == '__main__':
    avg_process_time()










