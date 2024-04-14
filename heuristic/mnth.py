import random
import math
from data import MyData

def mnth_process_time(M,J,w,p,JL,ML):
    def calculate_completion_time(solution):
        completion_times = [0] * len(M)
        for task in solution:
            machine_index = task[1] - 1  # 机器编号从0开始，调整为从1开始
            completion_times[machine_index] += task[0]
        return max(completion_times)

    def calculate_weighted_completion_time(solution):
        completion_time = calculate_completion_time(solution)
        weighted_completion_time = 0
        for task in solution:
            weighted_completion_time += w[task[2]] * completion_time
        return weighted_completion_time

    def generate_neighbor(solution):
        neighbor = solution[:]
        index1 = random.randint(0, len(neighbor)-1)
        index2 = random.randint(0, len(neighbor)-1)
        neighbor[index1], neighbor[index2] = neighbor[index2], neighbor[index1]
        return neighbor

    def simulated_annealing():
        current_solution = [(p[j][1], random.choice(M), j) for j in J]  # 随机初始化一个初始解
        current_cost = calculate_weighted_completion_time(current_solution)
        temperature = 1000
        cooling_rate = 0.9
        iterations = 1000
        while temperature > 0.1:
            for _ in range(iterations):
                neighbor = generate_neighbor(current_solution)
                neighbor_cost = calculate_weighted_completion_time(neighbor)
                delta_cost = neighbor_cost - current_cost
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                    current_solution = neighbor[:]
                    current_cost = neighbor_cost
            temperature *= cooling_rate
        return current_solution, current_cost


    best_solution, best_cost = simulated_annealing()
    print("Best Solution:", best_solution)
    print("Best Weighted Completion Time:", best_cost)



    licp_list = []
    time_sum = 0

    licp_sum = 0
    licp_up = 0
    for i in M:
        licp = 0
        for t in best_solution:
            if i == t[1]:
                # licp += p[t[2]][i]
                time_sum += p[t[2]][i]
                licp += JL[t[2]]
        # licp = licp / ML[i]
        licp_sum += licp
        licp_list.append(licp)
    licp_avg = licp_sum / len(M)

    #计算负载
    lm_sum = 0
    for i, element in enumerate(licp_list):
        lm = element / ML[i+1]
        lm_sum += lm

    for l in licp_list:
        licp_up += pow(l - licp_avg, 2)
    lcp = licp_up / len(M)
    print(f'mnth licp_avg  {licp_avg}')
    print(f"mnth计算负荷的方差为{lcp}")

    print(licp_list)

    # lm_sum = lcp
    return best_cost,lcp,time_sum,lm_sum

if __name__ == '__main__':
    M0, J0, w0, p0,JL,ML = MyData.data()
    mnth_process_time(M0, J0, w0, p0,JL,ML)

