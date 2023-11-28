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
        print("Simulation test for", exp_n, "iterations......")
        cost = 0
        state = 0.1
        for exp in range(1, exp_n+1):
            if (state - 0.1) / 0.01 >= policy:
                cost += 0.6
                state = 0.1
            elif np.random.choice(np.arange(0, 2), p=[state, 1 - state]) == 0:
                state = 0.1
                cost += 1
            else:
                state += 0.01
            if exp%100 == 0:
                values.append(cost/(exp*100))
        print("Long-run average cost: ", cost / exp_n)
        plot_result(values, "Average Cost", "cost", "iteration", None)
    sum = 0
    tmps = [1] + [0] * 91
    for i in range(1, policy+1):
        tmps[i] = (0.9 - 0.01 * i) * tmps[i - 1]
    for i in range(policy+1):
        sum += tmps[i]
    possition = tmps[:]
    #print(possition)
    print("Stationary distribution:")
    for i in range(policy+1):
        possition[i] /= sum
    print(possition)
    plot_result(possition, "Stationary Distribution", "distribution", "state", None)
    simulate(exp_n)

def poisson():
    print("================== Poisson equation =================")
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
    #print(poisson.T)
    result = np.linalg.solve(poisson.T, right)
    #print(np.sum(result), result)
    reward = []
    for i in range(91):
        reward.append(result[i] * expected_cost[i])
    print("Expected reward(cost):", np.sum(reward))
    print(reward)


def policy_iteration():
    print("================== Policy iteration =================")
    # policy iteration
    reward = np.zeros((91, 2))
    probability = np.zeros((91, 91, 2))
    poisson = np.zeros((92, 92, 2))
    right = [0] * 90 + [1]

    for i in range(91):
        reward[i, 0] = (i * 0.01 + 0.1)
        reward[i, 1] = 0.6
    for i in range(91):
        # action 0
        probability[i, 0, 0] = 0.1 + 0.01 * i
        probability[i, (i + 1) % 91, 0] += 1 - probability[i, 0, 0]
        # action 1
        probability[i, 0, 1] = 1
    for i in range(91):
        poisson[i, i+1, 0] = (i * 0.01) - 0.9
        poisson[i, i, 0] = 1
        poisson[i, 0, 0] -= probability[i, 0, 0]
        poisson[i, 91, 0] = 1

        poisson[i, i, 1] = 1
        poisson[i, 0, 1] -= 1
        poisson[i, 91, 1] = 1
    poisson[91, 0, 0] = 1
    poisson[91, 0, 1] = 1
    cost = np.append(reward[:, 0], 0)
    v = np.linalg.solve(poisson[:, :, 0], cost[:])
    cmp_policy = [0] * 91
    policy = [0] * 91
    loop = 0
    flag = 0
    cost = []
    for i in range(91):
        cost.append(v[i] * reward[i, 0])
    values = [np.sum(cost)]
    while True:
        for i in range(91):
            action_1 = probability[i, 0, 1] * v[0] + reward[i, 1]
            if i == 90:
                action_0 = v[0] + reward[i, 0]
            else:
                action_0 = probability[i, (i + 1) % 91, 0] * v[i + 1] + probability[i, 0, 0] * v[0] + reward[i, 0]
            if action_0 <= action_1:
                cmp_policy[i] = 0
            else:
                cmp_policy[i] = 1
            if policy[i] != cmp_policy[i]:
                # print(cmp_policy.count(0), i)
                loop += 1
                test_poi = np.zeros((92, 92))
                cost = [0] * 92 # np.zeros((92, 1))
                for i in range(91):
                    for j in range(92):
                        test_poi[i, j] = poisson[i, j, cmp_policy[i]]
                    cost[i] = reward[i, cmp_policy[i]]
                for j in range(92):
                    test_poi[91, j] = poisson[91, j, 1]
                v = np.linalg.solve(test_poi, cost)
                for i in range(91):
                    cost[i] *= v[i]
                values.append(np.mean(cost))
        flag += 1

        if operator.eq(policy, cmp_policy) == True:
            print("Optimal policy: \n", policy)
            print("Do replacement after iteration: ", policy.count(0))
            print("Loop:", loop)
            #print(values[-1])
            #print(flag)
            #print(values)
            break

        policy = cmp_policy
        #print(np.sum(v), policy.count(0))
    return policy.count(0)

def value_iteration():
    print("================== Value iteration =================")
    # value iteration
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
        # print(v_map, np.max(v_map))
        values.append(np.mean(v_map))
        if flag == 0 or values[-1] - values[-2] == 0 or len(values) > 1000:
            print("Terminate after iteration: ", len(values) - 1)
            #print(policy.count(0))
            break
        flag = 0
        # print(policy, policy.count(0))
        v_now = v_next
        policy_old = policy
    print("Optimal policy:\n", policy)
    print("Do replacement after iteration: ", policy.count(0))
    return policy.count(0)
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


print("==============Stationary distribution==========")
stationary_distribution(10000, 90)
poisson()
policy = policy_iteration()
value = value_iteration()
if policy == value:
    print("Value iteration and Policy iteration generate same policy")

print("================== Applying optimal policy =================")
stationary_distribution(10000, policy)