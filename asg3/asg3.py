import csv
import os
import random

import numpy as np

debug = 0
start_state = np.zeros([2, 6, 7])
wins_u = {'max': np.array([]), 'min': np.array([]), 'mean': np.array([])}
data = {'max': np.array([]), 'min': np.array([]), 'mean': np.array([])}

error_num = 0

def check_row(state, agent, rows):
    win = 0
    for i in rows:
        for j in range(7):
            if state[agent][i[0]][j] == 1:
                win += 1
            else:
                win = 0
            if win >= 4:
                return agent
    return -1


def check_col(state, agent, cols):
    win = 0
    for i in cols:
        for j in range(6):
            if state[agent][j][i[0]] == 1:
                win += 1
            else:
                win = 0
            if win >= 4:
                return agent
    return -1


def check_diag(state, agent):
    win = 0
    for i in range(-2, 4):
        if np.sum(np.diagonal(state[agent], offset=i)) >= 4:
            diags = np.diagonal(state[agent], offset=i)
            for k in diags:
                if k == 1:
                    win += 1
                else:
                    win = 0
                if win >= 4:
                    return agent
        if np.sum(np.diagonal(np.fliplr(state[agent]), offset=i)) >= 4:
            diags = np.diagonal(np.fliplr(state[agent]), offset=i)
            for k in diags:
                if k == 1:
                    win += 1
                else:
                    win = 0
                if win >= 4:
                    return agent
    return -1


def check_win(state):
    row_1 = np.argwhere(np.sum(state[0], axis=1) >= 4)
    row_2 = np.argwhere(np.sum(state[1], axis=1) >= 4)
    if check_row(state, 0, row_1) != -1:
        return True, 0
    if check_row(state, 1, row_2) != -1:
        return True, 1
    column = np.argwhere(np.sum(state[0], axis=0) >= 4)
    if check_col(state, 0, column) != -1:
        return True, 0
    column = np.argwhere(np.sum(state[1], axis=0) >= 4)
    if check_col(state, 1, column) != -1:
        return True, 1
    if check_diag(state, 0) != -1:
        return True, 0
    if check_diag(state, 1) != -1:
        return True, 1
    return False, None


def show_board(state):
    print("--------------")
    for i in range(6):
        for j in range(7):
            if state[0, i, j] == 1:
                print("x", end=' ')
            elif state[1, i, j] == 1:
                print("o", end=' ')
            else:
                print(" ", end=' ')
        print()
    print("--------------")


def available_move(state):
    actions = []
    for i in range(7):
        if np.sum(state[0][:, i]) + np.sum(state[1][:, i]) < 6:
            actions.append(i)
    return actions


def update_state(state, agent, action):
    global error_num
    current_state = state.copy()
    action_y = np.sum(current_state[0][:, action]).astype(int) + np.sum(current_state[1][:, action]).astype(int)
    if action_y >= 6:
        print("error!")
        error_num += 1
        show_board(current_state)
        print(current_state[0], action, action_y)
        #  print(current_state[1])
        return current_state
    current_state[agent][action_y][action] = 1
    return current_state


def purely_random(state):
    state_list = available_move(state)
    if state_list:
        random_index = np.random.randint(len(state_list))
        return state_list[random_index]
    else:
        return None


def UCB(wins, n_i, N):
    c = np.sqrt(2)
    if np.any(n_i == 0):
        return np.argwhere(n_i == 0)[0][0]
    else:
        a = np.argmax(wins / n_i + c * np.sqrt(np.log(N) / n_i))
    return a


def UCT(start_state, k, record, actions):
    T = 1000
    n_i = np.zeros(k)
    wins = np.zeros(k)
    score = np.zeros([1000, 7])
    for t in range(T):
        a_t = UCB(wins, n_i, t)
        n_i[a_t] += 1
        s = start_state.copy()
        if debug:
            print(a_t)
            print(actions)
            print(n_i)
        s = update_state(start_state.copy(), 0, actions[a_t])
        win = False
        agent2 = True
        while not win:
            win, agent = check_win(s)
            if agent == 0:
                wins[a_t] += 1
            # roll-out
            a = purely_random(s)
            if a != None:
                if agent2:
                    s = update_state(s.copy(), 1, a)
                    agent2 = False
                else:
                    s = update_state(s.copy(), 0, a)
                    agent2 = True
            else:
                print("Fair!")
                show_board(s)
                break
        #  show_board(s)
        if record:
            result = wins / n_i
            wins_u['max'] = np.append(wins_u['max'], np.max(result))
            wins_u['min'] = np.append(wins_u['max'], np.min(result))
            wins_u['mean'] = np.append(wins_u['max'], np.mean(result))
            score[t][actions[:]] = result[:]
            #print(score[t], "\n", result)
    print(wins / n_i)
    print(n_i)
    result = wins / n_i
    if not record:
        data['max'] = np.append(data['max'], np.max(result))
        data['min'] = np.append(data['max'], np.min(result))
        data['mean'] = np.append(data['max'], np.mean(result))
    else:
        print_2_csv(0, 1000, 'result3.csv', True, score.copy())
    return np.argmax(wins / n_i)

def print_2_csv(episode, runs, filename, record, score):
    print("SAVE RESULTS TO CSV")
    if not os.path.exists('data'):
        os.makedirs('data')
    if os.path.exists('data/'+filename):
        os.remove('data/'+filename)
    with open('data/'+filename, 'a+', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(
                ["generation"] + [k for k in range(1, 8)])
        if record:
            for i in range(len(score)):
                filewriter = csv.writer(csvfile, delimiter=',')
                #  filewriter.writerow([i + 1, wins_u['max'][i], wins_u['mean'][i], wins_u['min'][i]])
                filewriter.writerow([i + 1, score[i][0], score[i][1], score[i][2], score[i][3], score[i][4], score[i][5], score[i][6]])
            score = np.zeros([1000, 7])
        else:
            for i in range(len(data['max'])):
                filewriter = csv.writer(csvfile, delimiter=',')
                filewriter.writerow([i + 1, data['max'][i], data['mean'][i], data['min'][i]])


# simulation
random.seed(42)
agent2 = True
win = False
state = start_state.copy()
record = True
while not win:
    if agent2:
        a_2 = purely_random(state)
        state = update_state(state, 1, a_2)
        win, agent = check_win(state)
        show_board(state)
        #print(a_2)
        agent2 = False
    else:
        actions = available_move(state)
        #print(actions)
        a_2 = UCT(state.copy(), len(actions), record, actions)
        record = False
        a_2 = actions[a_2]
        #print(actions)
        state = update_state(state, 0, a_2)
        win, agent = check_win(state)
        show_board(state)
        agent2 = True
    #     print(agent)
show_board(state)

def simulation(episode):
    action_count = np.zeros((20))
    win_count = np.zeros((20))
    lose_count = np.zeros((20))
    for i in range(episode):
        state = start_state.copy()
        print("====== Generation ", i + 1, "=============")
        agent2 = True
        win = False
        state = start_state.copy()
        a_count = 0
        while not win:
            if agent2:
                a_2 = purely_random(state)
                state = update_state(state, 1, a_2)
                win, agent = check_win(state)
                show_board(state)
                agent2 = False
            else:
                actions = available_move(state)
                a_2 = UCT(state.copy(), len(actions), True, actions)
                a_2 = actions[a_2]
                print(actions)
                state = update_state(state, 0, a_2)
                win, agent = check_win(state)
                show_board(state)
                agent2 = True
                a_count += 1
        #     print(agent)
        win, agent = check_win(state)
        action_count[a_count] += 1
        show_board(state)
        if agent == 0:
            win_count[a_count] += 1
        elif agent == 1:
            lose_count[a_count] += 1
    print(action_count)
    print(win_count)
    print(lose_count)
    print(error_num)
    print_2_csv(0, len(data['max']), 'result_2.csv', False)
# simulation 1000
episode = 1000
#  simulation(episode)