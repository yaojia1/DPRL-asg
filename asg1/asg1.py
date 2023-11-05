"""
10 items are presented to you one by one.
Upfront you do not know their sizes,
but when an item is presented you are allowed to decide to take it based on its size
• Every item you take gives a reward of 1, 0 if you don’t
• The size of an item is random on all with probability 0.1
• The size of your knapsack is 10
• What is the optimal policy and maximal expected reward?

To find the optimal policy answer the following questions:
a) Prove that when it is optimal to take an item of size at time then it is
also optimal to take items smaller than s. (Hint: use induction in )
b) Use the answer to a) to find appropriate X and A
c) Implement the problem as a finite-horizon dp. Plot the optimal policy and
give the expected maximal reward. Interpret the results
d) Simulate the process 1000 times under the optimal policy, make a
histogram and compute the average reward. Interpret the results
"""

import random
import numpy as np

knapsack_size = 10
T = 10
weight_max = 10
size_possibility = 0.1
reward = 1  # per item
print("hello world")
bound_s = 0

"""
V_t(x) = max_a{r_t(x,a)+V_{t+1}( transition(x,a) )}
"""

alpha = np.zeros([T, knapsack_size + 1])  # optimal policy s for action at state x
V = np.zeros([knapsack_size + 1, T + 1])  # value set of optimal policy, Value[time][state], last raw for ending edge

Q = [-1, -1]  # reward for an action

for t in range(T)[::-1]:  # each time step
    for x in range(weight_max + 1):  # each possible states, [0 - 10]
        # expected reward for each action
        Q = []
        for s in range(x + 1):  # each possible policy s (max taken weight for x and t) [0 - 10]
            expd_weight = 0
            for i in range(1, s + 1):  # each possible transition path for s at state x and t, the expected value
                expd_weight += (size_possibility * V[t + 1][x - i])
            expd_weight += (size_possibility * (weight_max - s) * V[t + 1][x])
            expd_weight += (s * size_possibility)
            Q.append(expd_weight)
        print(Q)
        print("time:", t, "state", x, "best gained value:", np.max(Q), "at s =", np.argmax(Q))
        V[t][x] = np.max(Q)
        alpha[t][x] = np.argmax(Q)
    print("optimal values: ", V[t])
    print("optimal policy: ", alpha[t])

print("optimal values: ", V)
print("optimal policy: ", alpha)


def run_experiment(n=1000, batch=100):
    all_value = 0
    all_weight = 0
    batch_value = 0
    batch_weight = 0
    generations = 0
    for epoch in range(1, n + 1):
        if epoch % batch == 0:
            all_value += batch_value / batch
            batch_value = 0
            all_weight += batch_weight / batch
            batch_weight = 0
            generations += 1
            print("generation", generations, "value: ", all_value, "weight:", all_weight)
        weights = np.random.randint(low=1, high=10, size=10,
                                    dtype=int)  # [random.randint(1, knapsack_size) for x in range(T)]
        # print(weights)
        weight = weight_max
        value = 0
        for t in range(T):
            if weights[t] <= alpha[t][weight]:
                # print("take item ", t + 1, "with weight", weights[t])
                weight -= weights[t]
                value += 1
        batch_value += value
        batch_weight += weight

    print("optimal average value: ", all_value / generations)
    print("average remain weight: ", all_weight / generations)


np.random.seed(42)
run_experiment()