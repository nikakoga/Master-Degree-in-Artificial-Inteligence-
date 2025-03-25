import tsplib95
import numpy as np

def load_problem(problem_name):
    return tsplib95.load(f'{problem_name}.tsp')

def calculate_distance_matrix(problem):
    nodes = list(problem.get_nodes())
    n = len(nodes)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = problem.get_weight(nodes[i], nodes[j])
    return distance_matrix

# Load the problems
kroa200 = load_problem('kroA200')
krob200 = load_problem('kroB200')

# Calculate distance matrices
distance_matrix_kroa200 = calculate_distance_matrix(kroa200)
distance_matrix_krob200 = calculate_distance_matrix(krob200)

# Print the distance matrices
print("Distance Matrix for kroA:")
print(distance_matrix_kroa200)

print("\nDistance Matrix for kroB:")
print(distance_matrix_krob200)