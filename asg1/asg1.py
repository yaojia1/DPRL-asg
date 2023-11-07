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
import numpy as np

knapsack_size = 10
T = 10
weight_max = 10
size_possibility = 0.1
num_runs = 1000


def optimal_policy():
    """
    V_t(x) = max_a{r_t(x,a)+V_{t+1}( transition(x,a) )}
    """
    print("===================== Calculating policy ============================")
    alpha = np.zeros([T, knapsack_size + 1])  # optimal policy s for action at state x
    V = np.zeros(
        [knapsack_size + 1, T + 1])  # value set of optimal policy, Value[time][state], last raw for ending edge

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
    return alpha, V


def run_experiment(n=1000, batch=100):
    all_value = 0
    all_weight = 0
    batch_value = 0
    batch_weight = 0
    generations = 0
    data = {'reward': np.array([]), 'weight': np.array([]), 'best_reward': np.array([]), 'best_weight': np.array([])}
    exp_diff = {'reward_diff': np.array([]), 'weight_diff': np.array([]),
                'reward': np.array([]), 'weight': np.array([]), 'best_reward': np.array([]), 'best_weight': np.array([])}
    all_weights = np.array([])
    uppertake = []
    lowerntake = []
    def determin_solution(weights, value):
        action = np.zeros([T, knapsack_size + 1])  # optimal policy s for action at state x
        V = np.zeros(
            [knapsack_size + 1, T + 1])  # value set of optimal policy, Value[time][state], last raw for ending edge
        for t in range(T)[::-1]:  # each time step
            for x in range(weight_max + 1):  # each possible states, [0 - 10]
                if x - weights[t] >= 0 and 1 + V[t + 1][x - weights[t]] >= V[t + 1][x]:
                    action[t][x] = 1
                    V[t][x] = 1 + V[t + 1][x - weights[t]]
                else:
                    action[t][x] = 0
                    V[t][x] = V[t + 1][x]
        weight = weight_max
        reward = 0
        record = V[0][weight_max] > value
        for t in range(T):
            if action[t][weight]:
                reward += 1
                # print("take item ", t + 1, "with weight", weights[t])
                weight -= weights[t]
                if record and alpha[t][weight] < weights[t]:
                    uppertake.append(weights[t])
                    record = 0
            elif record and alpha[t][weight] > weights[t]:
                lowerntake.append(weight)
                record = 0
        # print("determine best reward:", V[0][weight_max], "weight:", weight)
        if reward != V[0][weight_max]:
            print("error!!!!!", reward, V[0][weight_max])
        # print(V)
        # print(alpha)
        data['best_reward'] = np.append(data['best_reward'], V[0][weight_max])
        data['best_weight'] = np.append(data['best_weight'], int(weight))
        return V[0][weight_max], int(weight)

    def prepare_file(dir_name, clear_file=True):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if clear_file:
            if os.path.exists(dir_name + '/results.csv'):
                os.remove(dir_name + '/results.csv')

    def print_2_csv(eponum=None, epo_start=0, expname=None, ):
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
                print("optimal best at epoch", best_optimal_index, ":", data['reward'][best_optimal_index],
                      data['weight'][best_optimal_index])
                print("determine best at epoch", best_deter_index, ":", data['best_reward'][best_deter_index],
                      data['best_weight'][best_deter_index])

                # Creating histogram
                plt.hist(data['reward'], align='left', bins=range(weight_max + 2))

                # Set title
                plt.title("Rewards Histogram")

                # adding labels
                plt.xlabel('reward')
                plt.ylabel('count')

                # Make some labels.
                labels = [i for i in range(weight_max+2)]

                plt.xticks(labels)

                plt.show()
                plt.savefig('data/' + expname + "/result")

                # Compare with determine solutions
                lens = np.max(exp_diff['reward_diff']).astype(int)
                mins = np.min(exp_diff['reward_diff']).astype(int)
                plt.hist(exp_diff['reward_diff'].astype(int), align='left', bins=range(mins - 1, lens + 2))

                # Set title
                plt.title("Rewards Difference")

                # adding labels
                plt.xlabel('reward')
                plt.ylabel('count')
                plt.xticks([i for i in range(mins - 1, lens + 2)])
                plt.show()

                # the difference of weights
                # Compare with determine solutions
                lens = np.max(exp_diff['weight_diff']).astype(int)
                mins = np.min(exp_diff['weight_diff']).astype(int)
                plt.hist(exp_diff['weight_diff'], align='left', bins=range(mins - 1, lens + 2))
                print(exp_diff['weight_diff'])

                # Set title
                plt.title("Weight Difference")

                # adding labels
                plt.xlabel('weight')
                plt.ylabel('count')
                plt.xticks([i for i in range(mins - 1, lens + 2)])
                plt.show()

                # the difference of weights
                # Compare with determine solutions
                plt.hist(all_weights.astype(int), align='left', bins=range(weight_max + 2))
                #print(all_weights)

                # Set title
                plt.title("Weight Distribution")

                # adding labels
                plt.xlabel('weight')
                plt.ylabel('count')
                plt.xticks([i for i in range(weight_max + 2)])
                plt.show()

                # the difference of weights
                # Compare with determine solutions
                sxdiff = []
                alphadiff = []
                tmp = np.zeros([11, 10])

                for t in range(T):
                    for x in range(1, weight_max+1):
                        tmp[x][t] = alpha[t][x]
                        sxdiff.append(x-alpha[t][x])
                        if x > alpha[t][x]:
                            alphadiff.append(alpha[t][x])
                lens = np.max(uppertake).astype(int)
                mins = np.min(uppertake).astype(int)
                plt.hist(uppertake, align='left', bins=range(mins - 1, lens + 2))
                print(uppertake)

                # Set title
                plt.title("Decision Difference")

                # adding labels
                plt.xlabel('weight')
                plt.ylabel('count')
                plt.xticks([i for i in range(mins - 1, lens + 2)])
                plt.show()

                lens = np.max(lowerntake).astype(int)
                mins = np.min(lowerntake).astype(int)
                plt.hist(lowerntake, align='left', bins=range(mins - 1, lens + 2))
                print(lowerntake)

                # Set title
                plt.title("Decision Difference")

                # adding labels
                plt.xlabel('weight')
                plt.ylabel('count')
                plt.xticks([i for i in range(mins - 1, lens + 2)])
                plt.show()
                print(tmp)
                plt.title("Optimal policy heat map")
                plt.xlabel('time')
                plt.ylabel('state')
                for y in range(tmp.shape[0]):
                    for x in range(tmp.shape[1]):
                        plt.text(x , y , '%d' % tmp[y, x],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 )
                plt.xticks([i for i in range(11)])
                plt.yticks([i for i in range(11)])
                plt.imshow(tmp, cmap='hot', interpolation='nearest')
                plt.show()

    for epoch in range(1, n + 1):
        weights = np.random.randint(low=1, high=11, size=10,
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
        d_reward, d_weight = determin_solution(weights, value)
        if d_reward > value:
            exp_diff['reward'] = np.append(exp_diff['reward'], value)
            exp_diff['weight'] = np.append(exp_diff['weight'], weight)
            exp_diff['best_reward'] = np.append(exp_diff['best_reward'], d_reward)
            exp_diff['best_weight'] = np.append(exp_diff['best_weight'], d_weight)
            exp_diff['reward_diff'] = np.append(exp_diff['reward_diff'], value - d_reward)
            exp_diff['weight_diff'] = np.append(exp_diff['weight_diff'], weight - d_weight)
            # print("diff solution!", exp_diff['weight_diff'], weight, value, d_weight, d_reward)
        if epoch % batch == 0:
            all_value += batch_value / batch
            batch_value = 0
            all_weight += batch_weight / batch
            batch_weight = 0
            generations += 1
            print("generation", generations, "value: ", all_value, "weight:", all_weight)
            print_2_csv(epoch, expname=str(n) + '_runs')
        all_weights = np.append(all_weights, weights)
    print("optimal average value: ", all_value / generations)
    print("average remain weight: ", all_weight / generations)
    print("average item weights: ", np.mean(all_weights))


alpha, V = optimal_policy()
print("optimal values: ", V)
print("optimal policy: ", alpha)

np.random.seed(42)
run_experiment(num_runs)
print("Expect reward:", V[0][weight_max])
