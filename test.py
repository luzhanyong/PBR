



licp_list1 = [210, 780, 120, 150, 280, 100]
licp_list2 = [110, 210, 270, 290, 390, 170]
licp_list3 = [470, 40, 350, 40, 360, 180]

list1_sum = 0
list2_sum = 0
list3_sum = 0

for i in licp_list1:
    list1_sum += i

for i in licp_list2:
    list2_sum += i

for i in licp_list3:
    list3_sum += i

print(f'list1   {list1_sum}')
print(f'list2    {list2_sum}')
print(f'list3    {list3_sum}')


# licp_sum = 0
# licp_up = 0
# for licp in licp_list:
#     licp_sum += licp
# licp_avg = licp_sum / 6
#
# for l in licp_list:
#     licp_up += pow(l - licp_avg, 2)
# lcp = licp_up / 6
# print(f"rounding计算负荷的方差为{lcp}")
#
# licp_sum = 0
# licp_up = 0
# for licp in licp_list2:
#     licp_sum += licp
# licp_avg = licp_sum / 6
#
# for l in licp_list2:
#     licp_up += pow(l - licp_avg, 2)
# lcp = licp_up / 6
# print(f"greedy计算负荷的方差为{lcp}")