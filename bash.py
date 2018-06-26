import numpy
from math import gcd
import functools
import copy
from itertools import combinations


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
            c_e = self.evaluate_clause_by_clause(x)  # First, evaluate
            if 0 in c_e:
                for clause_number in range(len(c_e)):
                    if not c_e[clause_number]:
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
        print("Condition Number : ", numpy.linalg.cond(matrix))
        hitting_times = numpy.linalg.solve(matrix, b)

        return hitting_times  # UGH

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

num_var = 6
k = 4

formula = Formula([], 6)

good_clauses = formula.complete_good_clauses([pow(2, num_var) - 1], num_var, k)
bad_clauses = []
total = good_clauses + bad_clauses
print("Formula: ", total)
print("Assignments Still SAT: ", formula.still_sat(total, num_var, k))

next_formula = Formula(total, num_var)

ht_schoning = next_formula.find_hitting_times(pow(2, num_var) - 1, True)
ht_break = next_formula.find_hitting_times(pow(2, num_var) - 1, False)
total_sum_s = 0
total_sum_b = 0
for entry in range(len(ht_schoning)):
    total_sum_s += ht_schoning[entry]
    total_sum_b += ht_break[entry]
print("Schöning Expected Time : ", total_sum_s/pow(2, num_var))
print("Break Expected Time : ", total_sum_b/pow(2, num_var))

# Comments : Right now, this only works for formulas with one SAT assignment.
# Update : This just doesn't work.
