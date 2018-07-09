import numpy
from math import gcd
import functools
import copy
from itertools import combinations
import matplotlib.pyplot as plt
from random import shuffle


class Formula:
    def __init__(self, formula, num_variables):
        self.formula = formula  # "formula" is a list of list of numbers, with negative meaning NOT.
        self.num_variables = num_variables

    def evaluate_clause_by_clause(self, x):
        # "x" is the assignment of values to variables. TIME TO REVAMP: X IS NOW JUST A NUMBER
        assignment = [int(j) for j in bin(x)[2:]]
        if len(assignment) < self.num_variables:
            for i in range(0, self.num_variables - len(assignment)):
                assignment.insert(0, 0)  # To fix the length of the assignments

        clause_evaluation = []
        for clause in self.formula:
            output = 0
            for literal in clause:
                if literal < 0:
                    true_variable = assignment[-1 * literal - 1]
                    if true_variable == 0:
                        true_variable = 1
                    else:
                        true_variable = 0
                else:
                    true_variable = assignment[literal - 1]
                output = output + true_variable
            clause_evaluation.append(output)  # You could add in an extra line to evaluate the whole formula.

        return clause_evaluation

    def sum(self, vector):
        sum = 0
        for entry in vector:
            sum += entry
        return sum

    def find_probabilities_schoning(self):
        # Return a list of lists, where the first list is indexed by the binary assignment (turned into a number) and
        # the second list is the probabilities of moving to a, b, c, etc.
        transition_matrix = []
        assignments = [i for i in range(pow(2, self.num_variables))]

        for x in assignments:
            probability_distribution = [0 for i in range(pow(2, self.num_variables))]
            c_e = self.evaluate_clause_by_clause(x)  # First, evaluate the formula.
            if 0 in c_e:  # If it's not the SAT assignment.
                for clause_number in range(len(c_e)):
                    if not c_e[clause_number]:  # If it's false.
                        for literal in self.formula[clause_number]:
                            variable = literal
                            if variable < 0:
                                variable = -1 * variable
                            variable = variable - 1
                            # Given that we have a variable, find out the neighbor corresponding to it.
                            if int(x / pow(2, variable)) % 2 == 1:
                                place_in_matrix = x - pow(2, variable)
                            else:
                                place_in_matrix = x + pow(2, variable)
                            probability_distribution[place_in_matrix] += 1
            else:
                probability_distribution[x] = 1
            transition_matrix.append(probability_distribution)

        return transition_matrix

    def find_probabilities_one_step_lookahead(self):
        schoning_tm = self.find_probabilities_schoning()
        one_step_ahead = []

        for row in schoning_tm:
            multiples = []
            for entry_num in range(len(row)):
                multiple_of_dist = [row[entry_num] * x for x in schoning_tm[entry_num]]
                multiples.append(multiple_of_dist)
            one_step_ahead.append([sum(x) for x in zip(*multiples)])

        return one_step_ahead

    def lcm(self, numbers):
        product = 1
        complementary = [1 for i in range(len(numbers))]
        for num in range(len(numbers)):
            product = product * numbers[num]
            for comp in range(len(numbers)):
                if num - comp is not 0:
                    complementary[comp] = complementary[comp] * numbers[num]
        gcd_comp = functools.reduce(gcd, complementary)

        return int(product/gcd_comp)

    def find_probabilities_break(self):
        # Return a list of lists, where the first list is indexed by the binary assignment (turned into a number) and
        # the second list is the probabilities of moving to a, b, c, etc.
        transition_matrix = []
        assignments = [i for i in range(pow(2, self.num_variables))]

        for x in assignments:
            probability_distribution = [0 for i in range(pow(2, self.num_variables))]
            c_e = self.evaluate_clause_by_clause(x)  # First, evaluate
            if 0 in c_e:
                bad_variables = [-1 for i in range(self.num_variables)]  # BIG: Don't forget variables are off by 1.
                for clause_number in range(len(c_e)):
                    if not c_e[clause_number]:
                        for literal in self.formula[clause_number]:
                            if literal < 0:
                                true_variable = -1 * literal - 1
                            else:
                                true_variable = literal - 1
                            bad_variables[true_variable] += 1  # This is the part where we count the bad variables
                for var in range(len(bad_variables)):
                    if bad_variables[var] >= 0:
                        # So here, we flip each bad variable and find the number of bad clauses. First, we flip.
                        if int(x / pow(2, var)) % 2 == 1:
                            one_away = x - pow(2, var)
                        else:
                            one_away = x + pow(2, var)
                        one_away_evaluated = self.evaluate_clause_by_clause(one_away)  # HERE
                        # Count the number of bad clauses. Ideally, reusing bad_variables won't...
                        number_of_bad_clauses = 0
                        for output in one_away_evaluated:
                            if output == 0:
                                number_of_bad_clauses += 1
                        bad_variables[var] = number_of_bad_clauses  # So, successful ones get to be 0, not -1.

                # Next step:

                number_zero = 0
                for var in range(len(bad_variables)):
                    if bad_variables[var] == 0:
                        number_zero += 1

                find_lcm = []
                if number_zero == 0:
                    for var in range(len(bad_variables)):
                        if bad_variables[var] > 0:  # so this is b_i
                            find_lcm.append(bad_variables[var])
                    lcm = self.lcm(find_lcm)
                    for var in range(len(bad_variables)):
                        if bad_variables[var] > 0:
                            if int(x / pow(2, var)) % 2 == 1:
                                place_in_matrix = x - pow(2, var)
                            else:
                                place_in_matrix = x + pow(2, var)
                            probability_distribution[place_in_matrix] = lcm/bad_variables[var]  # maybe this works....?
                else:
                    #print("waoh", x)
                    for var in range(len(bad_variables)):
                        if bad_variables[var] == 0:
                            if int(x / pow(2, var)) % 2 == 1:
                                place_in_matrix = x - pow(2, var)
                            else:
                                place_in_matrix = x + pow(2, var)
                            probability_distribution[place_in_matrix] = 1  # sum it up

            else:
                probability_distribution[x] = 1
            transition_matrix.append(probability_distribution)

        return transition_matrix

    def find_neighbors(self, number):
        powers_of_two = [pow(2, i) for i in range(0, self.num_variables)]
        neighbors = []
        # So if it has a 1 in the last place, you subtract the one. If it as a 1 in the second to last, subtract 2.
        for two in powers_of_two:
            if int(number/two) % 2 == 1:
                neighbors.append(number - two)
            else:
                neighbors.append(number + two)
        return neighbors

    def find_hitting_times(self, sat, schoning):
        if schoning:
            transition_matrix = self.find_probabilities_schoning()
        else:
            transition_matrix = self.find_probabilities_break()

        matrix = []
        b = []
        for i in range(pow(2, self.num_variables)):
            probability_row = transition_matrix[i]
            matrix_row = [0 for i in range(len(probability_row))]
            sum = self.sum(probability_row)
            #print(sum)
            matrix_row[i] = sum
            if i is not sat:
                neighbors = self.find_neighbors(i)
                for neigh in neighbors:
                    matrix_row[neigh] = -1 * probability_row[neigh]
                b.append(sum)
            else:
                b.append(0)
            matrix.append(matrix_row)

        #print(transition_matrix)
        #print(matrix)
        #print(b)
        #print("Condition Number : ", numpy.linalg.cond(matrix))
        hitting_times = numpy.linalg.solve(matrix, b)

        return hitting_times, numpy.linalg.cond(matrix)  # UGH

########### GENERATION OF COUNTEREXAMPLES ###########

    # This method takes a clause list and then finds the assignments that are still SAT.
    def still_sat(self, clause_list, num_variables, k):
        assignments = [[int(j) for j in bin(i)[2:]] for i in range(pow(2, num_variables))]
        for assignment in assignments:
            if len(assignment) < num_variables:
                for i in range(0, num_variables - len(assignment)):
                    assignment.insert(0, 0)  # To fix the length of the assignments

        binary_digits = [[int(j) for j in bin(x)[2:]] for x in range(pow(2, num_variables - k))]
        for word in binary_digits:
            if len(word) < num_variables - k:
                for i in range(0, num_variables - k - len(word)):
                    word.insert(0, 0)

        finished_assignments = []
        for clause in clause_list:
            true_variable = [0 for q in range(len(clause))]
            digit = [0 for q in range(len(clause))]

            for q in range(len(clause)):
                if clause[q] < 0:
                    true_variable[q] = -1 * clause[q] - 1
                    digit[q] = 1
                else:
                    true_variable[q] = clause[q] - 1
                    digit[q] = 0

            for rep in binary_digits:
                modified_rep = copy.deepcopy(rep)
                for q in range(len(clause)):
                    modified_rep.insert(true_variable[q], digit[q])

                if modified_rep not in finished_assignments:
                    finished_assignments.append(modified_rep)

        pow_of_two = [pow(2, i) for i in range(6)]
        num_assignments = [i for i in range(pow(2, 6))]
        num_finished_assignments = []
        for binary_rep in finished_assignments:
            number = 0
            for d in range(6):
                number += binary_rep[d] * pow_of_two[5 - d]
            num_finished_assignments.append(number)

        #print(num_assignments)
        #print(num_finished_assignments)
        return numpy.setdiff1d(num_assignments, num_finished_assignments)

    def complete_good_clauses(self, assignment_list, num_variables, k):
        all_literals = [i for i in range(1, num_variables + 1)]
        for i in range(1, num_variables + 1):
            all_literals.append(-i)
        all_clauses = list(combinations(all_literals, k))
        for i in range(1, num_variables + 1):
            if (i, -i) in all_clauses:
                #print("here")
                all_clauses.remove((i, -i))
            if (-i, i) in all_clauses:
                #print("also here")
                all_clauses.remove((i, -i))

        allowed = []
        for clause in all_clauses:
            success = 1
            for x in assignment_list:
                assignment = [int(j) for j in bin(x)[2:]]
                if len(assignment) < self.num_variables:
                    for i in range(0, self.num_variables - len(assignment)):
                        assignment.insert(0, 0)
                output = 0
                for literal in clause:
                    if literal < 0:
                        true_variable = assignment[-1 * literal - 1]
                        if true_variable == 0:
                            true_variable = 1
                        else:
                            true_variable = 0
                    else:
                        true_variable = assignment[literal - 1]
                    output = output + true_variable
                if output == 0:
                    success = 0
            if success:
                allowed.append(clause)

        return allowed


def hitting_times():
    num_var = 6
    k = 3

    formula = Formula([], 6)

    good_clauses = formula.complete_good_clauses([pow(2, num_var) - 1], num_var, k)
    bad_clauses = []
    total = good_clauses + bad_clauses
    print("Formula: ", total, len(total))
    print("Assignments Still SAT: ", formula.still_sat(total, num_var, k))

    next_formula = Formula(total, num_var)

    ht_schoning, s_cond_number = next_formula.find_hitting_times(pow(2, num_var) - 1, True)
    ht_break, b_cond_number = next_formula.find_hitting_times(pow(2, num_var) - 1, False)
    total_sum_s = 0
    total_sum_b = 0
    for entry in range(len(ht_schoning)):
        total_sum_s += ht_schoning[entry]
        total_sum_b += ht_break[entry]
    print("Schöning Expected Time : ", total_sum_s / pow(2, num_var))
    print("Break Expected Time : ", total_sum_b / pow(2, num_var))

def base_list_generator(num_var, k):
    powers_of_two = [i for i in range(pow(2, k) - 1)]
    binary_powers = []
    for num in powers_of_two:
        assignment = [int(j) for j in bin(num)[2:]]
        if len(assignment) < k:
            for i in range(0, k - len(assignment)):
                assignment.insert(0, 0)
        binary_powers.append(assignment)

    clause_list = []
    i = 1
    while i < num_var:
        for assignment in binary_powers:
            clause = []
            for literal in assignment:
                if literal == 0:
                    clause.append(i)
                else:
                    clause.append(-1 * i)
                i += 1
            #print(clause)
            clause_list.append(clause)
            i = i - k
            print(i)
        if i + k > num_var + 1 and i < num_var:
            remainder = i + k - num_var
            i = i + remainder
        else:
            i = i + k

    #print(len(clause_list))
    return clause_list

def num_clauses_experiment(num_var, k):

    base_clauses = base_list_generator(num_var, k)
    formula = Formula([], num_var)
    good_clauses = formula.complete_good_clauses([pow(2, num_var) - 1], num_var, k)

    shuffle(good_clauses)

    num_clauses = []
    break_performance = []
    schoning_performance = []

    m = len(base_clauses)
    i = 0

    #print(len(good_clauses))
    for clause in good_clauses:
        if i < 100:
            if clause not in base_clauses:
                base_clauses.append(clause)
                if len(formula.still_sat(base_clauses, num_var, k)) > 1:
                    raise ("NOT A UNIQUE SAT")

                m += 1

                nf = Formula(base_clauses, num_var)
                ht_schoning, s_cond = nf.find_hitting_times(pow(2, num_var) - 1, True)
                ht_break, b_cond = nf.find_hitting_times(pow(2, num_var) - 1, False)
                total_sum_s = 0
                total_sum_b = 0
                for entry in range(len(ht_schoning)):
                    total_sum_s += ht_schoning[entry]
                    total_sum_b += ht_break[entry]
                # print("Schöning Expected Time : ", total_sum_s / pow(2, num_var))
                # print("Break Expected Time : ", total_sum_b / pow(2, num_var))
                print("Condition Numbers: ", s_cond, b_cond)
                if s_cond < 1e+14 and b_cond < 1e14:
                    num_clauses.append(m)
                    break_performance.append(total_sum_b / pow(2, num_var))
                    schoning_performance.append(total_sum_s / pow(2, num_var))
                    print(i)
                    i += 1

    plt.scatter(num_clauses, break_performance, c="g", label="Break")
    plt.scatter(num_clauses, schoning_performance, c="r", label="Schoning")
    plt.xlabel("Number of Clauses")
    plt.ylabel("Expected Number of Steps")
    plt.legend()
    plt.title("{}-SAT on {} variables".format(k, num_var))
    plt.show()

#num_clauses_experiment(9, 3)
#print(base_list_generator(7, 3))
# Comments : Right now, this only works for formulas with one SAT assignment.
# Update : This just doesn't work.
