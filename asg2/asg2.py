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

reward = [0] * 91
flag = [0] * 90
flag.append(1)

probability = np.zeros((91, 91))
poisson = np.zeros((91, 91))
p = 1
sum = 0
tmps = [1]+[ 0 for i in range(90)]
for i in range(1,91):
    tmps[i] = (0.91-0.01*i)*tmps[i-1]
for i in range(91):
    sum += tmps[i]
possition = tmps[:]
print(possition)
print(sum)

for i in range(91):
    possition[i] /= sum
print(possition)


for i in range(91):
    reward[i] = (i * 0.01 + 0.1)
    probability[i, 0] = i * 0.01 + 0.1
    if i < 90:
        probability[i, i + 1] = 1 - probability[i, 0]
for i in range(91):
    for j in range(91):
        poisson[i, j] = - probability[i, j]
        # poisson[91, j] = 1
    poisson[i, i] = poisson[i, i] + 1
    poisson[i, 90] = 1

print(poisson.T)

# print(len(poisson))

result = np.linalg.solve(poisson.T, flag)

print(probability)
print(poisson.T)
print(result)
re = []
for i in range(91):
    re.append(result[i] * reward[i])

print(sum(result))
print(sum(re))