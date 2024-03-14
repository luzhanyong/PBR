## 使用IFAHP跑出来的权重为  任务的类型、时延要求、资源需求 0.62  0.28  0.1


#定义参数

c1 = 0.62
c2 = 0.28
c3 = 0.1


# def normal()


def priority(value_c1, value_c2, value_c3):

    return (value_c1 * c1 + value_c2 * c2 + value_c3 *c3)

