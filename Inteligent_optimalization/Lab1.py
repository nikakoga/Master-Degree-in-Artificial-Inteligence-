import tsplib95
import numpy as np
import random
import matplotlib.pyplot as plt

class CycleOptimizer:
    def __init__(self, problem_name):
        self.problem = self.load_problem(problem_name)
        self.distance_matrix = self.calculate_distance_matrix(self.problem)

    def load_problem(self, problem_name):
        return tsplib95.load(f'{problem_name}.tsp')

    def calculate_distance_matrix(self, problem):
        nodes = list(problem.get_nodes())
        n = len(nodes)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i, j] = problem.get_weight(nodes[i], nodes[j])
        return distance_matrix

    def create_two_cycles(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def calculate_cycle_length(self, cycle):
        length = 0
        for i in range(len(cycle)):
            length += self.distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]]
        return length

    def plot_cycles(self, cycle1, cycle2, title, subplot_position):
        nodes = list(self.problem.get_nodes())
        coordinates = np.array([self.problem.node_coords[node] for node in nodes])

        plt.subplot(subplot_position)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c='black')

        def plot_cycle(cycle, color):
            cycle_coords = np.array([coordinates[node] for node in cycle])
            cycle_coords = np.vstack([cycle_coords, cycle_coords[0]])
            plt.plot(cycle_coords[:, 0], cycle_coords[:, 1], color=color)

        plot_cycle(cycle1, 'blue')
        plot_cycle(cycle2, 'red')

        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')


class BalancedCycleOptimizer(CycleOptimizer):
    def insert_best(self, cycle, unvisited, strategy):
        best_value = float('inf')
        best_node = None
        best_pos = None

        for node in unvisited:
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                increase = self.distance_matrix[a, node] + self.distance_matrix[node, b] - self.distance_matrix[a, b]
                value = strategy(increase, a, node, b)
                if value < best_value:
                    best_value = value
                    best_node = node
                    best_pos = i + 1

        return best_node, best_pos

    def build_balanced_cycles(self, strategy):
        n = len(self.distance_matrix)
        target1 = n // 2 + n % 2
        target2 = n // 2

        start1 = random.randint(0, n - 1)
        start2 = max(range(n), key=lambda i: self.distance_matrix[start1][i])

        cycle1 = [start1, start1]
        cycle2 = [start2, start2]

        unvisited = set(range(n)) - {start1, start2}

        while len(cycle1) - 1 < target1 or len(cycle2) - 1 < target2:
            if len(cycle1) - 1 < target1:
                node, pos = self.insert_best(cycle1, unvisited, strategy)
                cycle1.insert(pos, node)
                unvisited.remove(node)
            if len(cycle2) - 1 < target2 and unvisited:
                node, pos = self.insert_best(cycle2, unvisited, strategy)
                cycle2.insert(pos, node)
                unvisited.remove(node)

        return cycle1[:-1], cycle2[:-1]


class GreedyCycleOptimizer(BalancedCycleOptimizer):
    def create_two_cycles(self):
        return self.build_balanced_cycles(lambda inc, a, n, b: inc)


class NearestNeighborCycleOptimizer(BalancedCycleOptimizer):
    def create_two_cycles(self):
        def nearest_insertion_cost(inc, a, n, b):
            return min(self.distance_matrix[n][a], self.distance_matrix[n][b])
        return self.build_balanced_cycles(nearest_insertion_cost)


class RegretCycleOptimizer(BalancedCycleOptimizer):
    def create_two_cycles(self):
        def regret_strategy(inc, a, n, b):
            return inc
        return self.build_balanced_cycles(regret_strategy)


class WeightedRegretCycleOptimizer(BalancedCycleOptimizer):
    def create_two_cycles(self, weight1=1, weight2=-1):
        def weighted_regret(inc, a, n, b):
            return weight1 * inc + weight2 * (self.distance_matrix[a][n] + self.distance_matrix[n][b]) / 2
        return self.build_balanced_cycles(weighted_regret)


# === EXPERIMENTAL RESULTS TABLE ===
def run_experiments(optimizer_class, problem_name, label, runs=100):
    total_lengths = []

    for i in range(runs):
        random.seed(i)
        optimizer = optimizer_class(problem_name)
        if isinstance(optimizer, WeightedRegretCycleOptimizer):
            cycle1, cycle2 = optimizer.create_two_cycles(weight1=1, weight2=-1)
        else:
            cycle1, cycle2 = optimizer.create_two_cycles()
        length1 = optimizer.calculate_cycle_length(cycle1)
        length2 = optimizer.calculate_cycle_length(cycle2)
        total_lengths.append(length1 + length2)

    avg = round(np.mean(total_lengths), 2)
    min_val = int(np.min(total_lengths))
    max_val = int(np.max(total_lengths))

    return f"{avg} ({min_val} – {max_val})"

methods = [
    (GreedyCycleOptimizer, "Greedy"),
    (NearestNeighborCycleOptimizer, "Nearest Neighbor"),
    (RegretCycleOptimizer, "2-Regret"),
    (WeightedRegretCycleOptimizer, "Weighted 2-Regret")
]

# === OPTIONAL: PLOT EXAMPLES ===
def plot_example_cycles():
    fig = plt.figure(figsize=(12, 10))
    for j, instance in enumerate(["kroA200", "kroB200"]):
        for i, (optimizer_class, label) in enumerate(methods):
            optimizer = optimizer_class(instance)
            if isinstance(optimizer, WeightedRegretCycleOptimizer):
                cycle1, cycle2 = optimizer.create_two_cycles(weight1=1, weight2=-1)
            else:
                cycle1, cycle2 = optimizer.create_two_cycles()
            pos = 4 * j + i + 1  # 4 columns layout: 2x4 (instance x methods)
            optimizer.plot_cycles(cycle1, cycle2, f"{label} - {instance}", 240 + pos)
    plt.tight_layout()
    plt.show()

print("\n=== EXPERIMENTAL RESULTS (100 RUNS) ===")

instances = ["kroA200", "kroB200"]

header = "| Method \t\t\t| kroA200 \t\t\t\t| kroB200 \t\t\t\t|"
separator = "|-----------------------|-------------------------------|-------------------------------|"
print(header)
print(separator)

for optimizer_class, label in methods:
    results = [run_experiments(optimizer_class, inst, label) for inst in instances]
    print(f"| {label:<21} | {results[0]:<29} | {results[1]:<29} |")

plot_example_cycles()
