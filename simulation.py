import numpy as np
from bash import Formula

class SimpleMarkovChain:
    def __init__(self, list_states, transition_matrix):
        self.states = list_states
        self.num_states = len(self.states)
        self.indexed_states = dict()

        for i in range(self.num_states):
            self.indexed_states[self.states[i]] = i
        self.transition_matrix = []
        sums = []
        for row in transition_matrix:
            sum = 0
            for entry in row:
                sum += entry
            sums.append(sum)
        for row_num in range(len(transition_matrix)):
            norm_row = []
            for entry in transition_matrix[row_num]:
                norm_row.append(entry/sums[row_num])
            self.transition_matrix.append(norm_row)

    def step(self, initial_state):
        prob_row = self.transition_matrix[initial_state]
        x = np.random.choice(self.states, 1, p=prob_row)[0]
        return self.states[x]

    def time_travel(self, iterations, initial_state):
        curr_state = initial_state
        for i in range(iterations):
            print(curr_state)
            curr_state = self.step(curr_state)


def find_hitting_time(schoning):
    num_steps = 0

    f = Formula([[1, 2, 3], [-1, 2, 3], [1, -2, 3], [1, 2, -3], [1, -2, -3], [-1, -2, 3], [-1, 2, -3]], 3)
    if schoning:
        tm = f.find_probabilities_schoning()
    else:
        tm = f.find_probabilities_one_step_lookahead()
    mc = SimpleMarkovChain([i for i in range(pow(2, f.num_variables))], tm)

    #print(tm)
    curr_state = 0
    found = 0
    while not found:
        if curr_state == pow(2, f.num_variables) - 1:
            found = 1
        else:
            curr_state = mc.step(curr_state)
            num_steps += 1
        #print(curr_state)
    return num_steps
#print()
#print("Hitting Time: ", find_hitting_time(True))
#f = Formula([[1, 2, 3], [-1, 2, 3], [1, -2, 3], [1, 2, -3], [1, -2, -3], [-1, -2, 3], [-1, 2, -3]], 3)
#print(f.find_probabilities_schoning())
#print(f.find_probabilities_one_step_lookahead())

sch_av = []
osla_av = []

for i in range(100):
    sch_av.append(find_hitting_time(True))
    osla_av.append(find_hitting_time(False))

print("Schoning", sum(sch_av)/100)
print("One Step Look Ahead", sum(osla_av)/100)



