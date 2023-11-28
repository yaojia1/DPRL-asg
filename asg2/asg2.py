import csv
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import random
from sympy import *
import scipy
import operator

"""
Consider a system that slowly deteriorates over time:
– When it is new it has a failure probability of 0.1
– the probability of failure increases every time unit linearly with 0.01
– Replacement costs are 1, after replacement the part is new

a) What is an appropriate state space? Argue why this satisfies the Markov property
b) Compute the stationary distribution and use this to find the long-run average replacement costs
c) Simulate the system for a long period and verify that you get (approximately) the same answer
d) Solve the average-cost Poisson equation
e) Preventive replacement is possible at cost 0.6. What is the average optimal policy? Solve it using:
• policy iteration
• value iteration
"""

# state space


# solve stationary distribution & find phi
from scipy.linalg import solve

def plot_result(values, title, ylab, xlab, x_l):
    import numpy as np
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    x_list = []
    if x_l == None:
        for k in range(0, len(values) - 1):
            x_list.append(k)
    else:
        x_list = x_l[:]
    # policy[k] = policy[k] + 1

    ax.bar(x_list, values[1:])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    plt.show()

def stationary_distribution(exp_n, policy=90):
    values = []
    def simulate(exp_n):
        cost = 0
        state = 0.1
        for exp in range(1, exp_n+1):
            if np.random.choice(np.arange(0, 2), p=[state, 1 - state]) == 0:
                state = 0.1
                cost += 1
            else:
                if (state-0.1)/0.01 >= policy:
                    cost += 0.6
                    state = 0.1
                else:
                    state += 0.01
            if exp%100 == 0:
                values.append(cost/(exp*100))
        print(cost, cost / 10000)
        plot_result(values, "Average Cost", "cost", "iteration", None)
    sum = 0
    tmps = [1] + [0] * policy
    for i in range(1, policy+1):
        tmps[i] = (0.9 - 0.01 * i) * tmps[i - 1]
    for i in range(policy+1):
        sum += tmps[i]
    possition = tmps[:]
    print(possition)
    print(sum, sum/(policy+1))
    for i in range(policy+1):
        possition[i] /= sum
    print(possition)
    simulate(exp_n)
stationary_distribution(10000, 90)
stationary_distribution(10000, 17)
def poisson():
    print("poisson equation")
    expected_cost = [0] * 91
    right = [0] * 90 + [1]
    probability = np.zeros((91, 91))
    poisson = np.zeros((91, 91))
    for i in range(91):
        expected_cost[i] = (i * 0.01 + 0.1) * 1
        probability[i, 0] = i * 0.01 + 0.1
        probability[i, (i + 1)%91] += (1 - probability[i, 0])
    for i in range(91):
        poisson[i, :91] = - probability[i, :91]
        poisson[i, i] = poisson[i, i] + 1
        poisson[i, 90] = 1
    print(poisson.T)
    result = np.linalg.solve(poisson.T, right)
    print(np.sum(result),result)
    reward = []
    for i in range(91):
        reward.append(result[i] * expected_cost[i])
    print(np.sum(reward), reward)
poisson()

def policy_iteration():
    # policy iteration
    reward = np.zeros((92, 2))
    probability = np.zeros((92, 92, 2))

    for i in range(91):
        reward[i, 0] = (i * 0.01 + 0.1)
        reward[i, 1] = 0.6
    for i in range(91):
        # action 0
        probability[i, i, 0] = 1
        probability[i, i + 1, 0] = (i * 0.01) - 0.9
        probability[i, 0, 0] = probability[i, 0, 0] - (0.01 * i) - 0.1
        probability[i, 91, 0] = 1
        # action 1
        probability[i, i, 1] = 1
        probability[i, 0, 1] = probability[i, 0, 1] - 1
        probability[i, 91, 1] = 1
    probability[91, 0, 0] = 1
    probability[91, 0, 1] = 1
    print(probability[:, :, 0])
    print(probability[:, :, 1])
    print(reward[:, 0])
    v = np.linalg.solve(probability[:, :, 0], reward[:, 0])
    cmp_policy = [0] * 91
    policy = [0] * 91
    loop = 0
    flag = 0
    values = [np.mean(v)]
    while True:
        for i in range(91):
            action_0 = probability[i, (i + 1) % 91, 0] * (-1) * v[i + 1] + probability[i, 0, 0] * (-1) * v[0] + reward[i, 0]
            action_1 = probability[i, 0, 1] * (-1) * v[0] + reward[i, 1]
            tmp_t = cmp_policy[i]
            if action_0 <= action_1:
                cmp_policy[i] = 0
            else:
                cmp_policy[i] = 1
            if tmp_t != cmp_policy[i]:
                print(cmp_policy.count(0), i)
                loop += 1
                test = np.zeros((92, 92))
                re = [0] * 92
                for i in range(91):
                    for j in range(92):
                        test[i, j] = probability[i, j, cmp_policy[i]]
                    re[i] = reward[i, cmp_policy[i]]
                for i in range(91, 92):
                    for j in range(92):
                        test[i, j] = probability[i, j, 1]
                    re[i] = reward[i, 1]
                v = np.linalg.solve(test, re)
                values.append(np.mean(v))
            # else:
            #   values.append(values[-1])
            # policy = cmp_policy
        flag += 1

        if operator.eq(policy, cmp_policy) == True:
            print(policy)
            print(v)
            print(policy.count(0))
            print(loop)
            print(np.mean(v))
            print(flag)
            break

        policy = cmp_policy
        print(np.mean(v), policy.count(0))

def value_iteration():
    # value iteration
    from scipy.linalg import solve
    import operator
    reward = np.zeros((91, 2))
    p = np.zeros((91, 91, 2))

    for i in range(91):
        reward[i, 0] = (i * 0.01 + 0.1)
        reward[i, 1] = 0.6
    for i in range(91):
        p[i, (i + 1) % 91, 0] = 0.9 - (i * 0.01)
        p[i, 0, 0] = (0.01 * i) + 0.1 - p[i, 0, 0]
        # p[i, 91, 0] = 1

        p[i, 0, 1] = 1 - p[i, 0, 1]
    v = [0] * 91
    # print(v)
    v_now = [0] * 91
    v_next = [0] * 91
    policy = [0] * 91
    flag = 0
    values = [0]
    v_map = [0] * 91
    policy_old = [0] * 91
    while True:
        v = np.minimum(
            reward[:, 0] + (p[:, :, 0] @ v_now)
            , reward[:, 1] + (p[:, :, 1] @ v_now)
        )
        # print(v)
        for i in range(91):
            policy[i] = np.argmin([reward[i, 0] + p[i, :, 0] @ v_now, reward[i, 1] + p[i, :, 1] @ v_now])
        v_next = np.minimum(reward[:, 0] + p[:, :, 0] @ v_now, reward[:, 1] + p[:, :, 1] @ v_now)
        for i in range(91):
            if abs(v_next[i]) - abs(v_now[i]) != v_map[i]:
                flag = 1
                v_map[i] = abs(v_next[i]) - abs(v_now[i])
        print(v_map, np.max(v_map))
        values.append(np.mean(v_map))
        if flag == 0 or values[-1] - values[-2] == 0 or len(values) > 1000:
            print(len(values) - 1, values)
            print(policy.count(0))
            break
        flag = 0
        print(policy, policy.count(0))
        v_now = v_next
        policy_old = policy
    print(policy)
    # plot_result(values, "value iteration", values)


def simulate(exp_n, policy):
    values=[]
    cost = 0
    state = 0.1
    for exp in range(1, exp_n + 1):
        if np.random.choice(np.arange(0, 2), p=[state, 1 - state]) == 0:
            state = 0.1
            cost += 1
        else:
            if state >= 0.1 + 0.01 * policy:
                cost += 0.6
                state = 0.1
            else:
                state += 0.01
        if exp % 100 == 0:
            values.append(cost / (exp * 100))
    print(cost, cost / 10000)
    plot_result(values, "Average Cost", "cost", "iteration", None)

simulate(10000,17)
stationary_distribution(10000)