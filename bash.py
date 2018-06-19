import numpy
from math import gcd
import functools
import copy

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

    def find_hitting_times(self, sat):
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
        print(numpy.linalg.cond(matrix))
        hitting_times = numpy.linalg.solve(matrix, b)

        return hitting_times  # UGH

    def random_subroutine(self):
        assignments = [[int(j) for j in bin(i)[2:]] for i in range(pow(2, 6))]
        for assignment in assignments:
            if len(assignment) < 6:
                for i in range(0, 6 - len(assignment)):
                    assignment.insert(0, 0)  # To fix the length of the assignments

        binary_four_digits = [[int(j) for j in bin(x)[2:]] for x in range(16)]
        for word in binary_four_digits:
            if len(word) < 4:
                for i in range(0, 4 - len(word)):
                    word.insert(0, 0)

        clause_list = [[-4, 5], [-3, 5], [-2, 5], [-1, 5], [5, 6], [1, -6], [2, -3],[2, -4], [3, -4], [4, -5], [-5, 6]]  # from [1, -2] : good clauses
        #clause_list = [[1, -2]]
        finished_assignments = []
        for clause in clause_list:
            literal_1 = clause[0]
            literal_2 = clause[1]
            if literal_1 < 0:
                true_variable_1 = -1 * literal_1 - 1
                digit_1 = 1
            else:
                true_variable_1 = literal_1 - 1  # _ _ 1 _ 0 _  add 1, 4, 16, 32 are free.
                digit_1 = 0

            if literal_2 < 0:
                true_variable_2 = -1 * literal_2 - 1
                digit_2 = 1
            else:
                true_variable_2 = literal_2 - 1  # _ _ 1 _ 0 _  add 1, 4, 16, 32 are free.
                digit_2 = 0

            for rep in binary_four_digits:
                modified_rep = copy.deepcopy(rep)
                modified_rep.insert(true_variable_1, digit_1)
                modified_rep.insert(true_variable_2, digit_2)
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
            #print(binary_rep, number)
        #print(finished_assignments)
        return numpy.setdiff1d(num_assignments, num_finished_assignments)

formula = Formula([[-4, 5], [-3, 5], [-2, 5], [-1, 5], [5, 6], [1, -6], [2, -3],[2, -4], [3, -4], [4, -5], [-5, 6]],6)  # [-3, 4], [-2, 4], [1, 2], [-1, 2], [1, -2], [3, -4]
# [1, 2, 3], [-1, 2, 3], [1, -2, 3], [1, 2, -3], [-1, -2, 3], [-1, 2, -3], [1, -2, -3]
print(formula.find_hitting_times(63))
#print(len(formula.random_subroutine()), formula.random_subroutine())

# Comments : Right now, this only works for formulas with one SAT assignment.
# Update : This just doesn't work.
