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
import csv
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
    data = {'reward': np.ndarray([]), 'weight': np.ndarray([]), 'best_reward': np.ndarray([]), 'best_weight': np.ndarray([])}

    def determin_solution(weights):
        alpha = np.zeros([T, knapsack_size + 1])  # optimal policy s for action at state x
        V = np.zeros(
            [knapsack_size + 1, T + 1])  # value set of optimal policy, Value[time][state], last raw for ending edge
        for t in range(T)[::-1]:  # each time step
            for x in range(weight_max + 1):  # each possible states, [0 - 10]
                if x - weights[t] >= 0 and 1+V[t+1][x-weights[t]] >= V[t+1][x]:
                    alpha[t][x] = 1
                    V[t][x] = 1+V[t+1][x-weights[t]]
                else:
                    alpha[t][x] = 0
                    V[t][x] = V[t+1][x]
        weight = weight_max
        reward = 0
        for t in range(T):
            if alpha[t][weight]:
                reward += 1
                # print("take item ", t + 1, "with weight", weights[t])
                weight -= weights[t]
        #print("determine best reward:", V[0][weight_max], "weight:", weight)
        if reward != V[0][weight_max]:
            print("error!!!!!", reward, V[0][weight_max])
        #print(V)
        #print(alpha)
        data['best_reward'] = np.append(data['best_reward'], V[0][weight_max])
        data['best_weight'] = np.append(data['best_weight'], weight)

    def prepare_file(dir_name, clear_file=True):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if clear_file:
            if os.path.exists(dir_name + '/results.csv'):
                os.remove(dir_name + '/results.csv')

    def print_2_csv(eponum=None, epo_start=0, expname=None,):
        print("SAVE RESULTS TO CSV")
        if eponum <= batch:
            prepare_file('data/' + str(expname), clear_file=True)
        with open('data/' + str(expname) + '/results.csv', 'a+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            if eponum <= batch:
                filewriter.writerow(
                    ["generation", "rewards", "weights", "best_reward", "weight"])
            else:
                epo_start = eponum - batch
            for i in range(epo_start, eponum):
                filewriter = csv.writer(csvfile, delimiter=',')
                filewriter.writerow([i + 1, data['reward'][i], data['weight'][i],
                                     data['best_reward'][i], data['best_weight'][i]])
            if eponum == n:
                best_optimal_index = np.argmax(data['reward'])
                best_deter_index = np.argmax(data['best_reward'])
                filewriter.writerow(["max", np.max(data['reward']), data['weight'][best_optimal_index],
                                     np.max(data['best_reward']), data['best_weight'][best_deter_index]])
                filewriter.writerow(["mean", np.mean(data['reward']), np.mean(data['weight']),
                                     np.mean(data['best_reward']), np.mean(data['best_weight'])])
                filewriter.writerow(["std", np.std(data['reward']), np.std(data['weight']),
                                     np.std(data['best_reward']), np.std(data['best_weight'])])
                print("optimal best", best_optimal_index, ":", data['reward'][best_optimal_index],
                      data['weight'][best_optimal_index])
                print("determine best", best_deter_index, ":", data['best_reward'][best_deter_index],
                      data['best_weight'][best_deter_index])

                # Creating histogram
                #fig, ax = plt.subplots(1, 1)
                plt.hist(data['reward'], align='left', bins=range(weight_max+1))

                # Set title
                plt.title("Rewards Histogram")

                # adding labels
                plt.xlabel('reward')
                plt.ylabel('count')

                # Make some labels.
                #rects = plt.patches
                labels = [i for i in range(weight_max)]

                #for rect, label in zip(rects, labels):
                    #height = rect.get_height()
                    #ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, int(height),
                            #ha='center', va='bottom')
                #plt.xticks(labels)
                plt.xticks(labels)
                #plt.grid(axis='y', alpha=0.75)
                #plt.xlabel('Value')
                #plt.ylabel('Frequency')
                #range_h = np.arange(0, 11, 1)
                #print(range_h)
                #plt.hist(data['reward'][:1000], align='left', bins=range_h, edgecolor = 'black', histtype='stepfilled')

                plt.show()
                plt.savefig('data/'+expname+"/result")
    for epoch in range(1, n + 1):
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
        data['reward'] = np.append(data['reward'], value)
        data['weight'] = np.append(data['weight'], weight)
        if value >= 8:
            print("9!!!!!", epoch)
        determin_solution(weights)
        if epoch % batch == 0:
            all_value += batch_value / batch
            batch_value = 0
            all_weight += batch_weight / batch
            batch_weight = 0
            generations += 1
            print("generation", generations, "value: ", all_value, "weight:", all_weight)
            print_2_csv(epoch, expname=str(n)+'_runs')
    print("optimal average value: ", all_value / generations)
    print("average remain weight: ", all_weight / generations)


np.random.seed(42)
run_experiment()
