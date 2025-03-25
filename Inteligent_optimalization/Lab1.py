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

class GreedyCycleOptimizer(CycleOptimizer):
    def create_two_cycles(self):
        n = len(self.distance_matrix)
        half_n = n // 2 + n % 2

        start_node1 = random.randint(0, n - 1)
        start_node2 = max(range(n), key=lambda node: self.distance_matrix[start_node1, node])

        unvisited = set(range(n))

        cycle1 = self.greedy_cycle(start_node1, half_n, unvisited)
        cycle2 = self.greedy_cycle(start_node2, n - half_n, unvisited)

        return cycle1, cycle2

    def greedy_cycle(self, start_node, cycle_length, unvisited):
        n = len(self.distance_matrix)
        cycle = [start_node]
        unvisited.remove(start_node)

        while len(cycle) < cycle_length:
            next_node = min(unvisited, key=lambda node: min(self.distance_matrix[node, cycle[i]] for i in range(len(cycle))))
            best_position = min(range(len(cycle)), key=lambda i: self.distance_matrix[cycle[i], next_node] + self.distance_matrix[next_node, cycle[(i + 1) % len(cycle)]] - self.distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]])
            cycle.insert(best_position + 1, next_node)
            unvisited.remove(next_node)

        return cycle

class NearestNeighborCycleOptimizer(CycleOptimizer):
    def create_two_cycles(self):
        n = len(self.distance_matrix)
        half_n = n // 2 + n % 2

        start_node1 = random.randint(0, n - 1)
        start_node2 = max(range(n), key=lambda node: self.distance_matrix[start_node1, node])

        unvisited = set(range(n))
        unvisited.remove(start_node1)
        unvisited.remove(start_node2)

        cycle1 = self.nearest_neighbor_cycle(start_node1, half_n, unvisited)
        cycle2 = self.nearest_neighbor_cycle(start_node2, n - half_n, unvisited)

        return cycle1, cycle2

    def nearest_neighbor_cycle(self, start_node, cycle_length, unvisited):
        cycle = [start_node]

        while len(cycle) < cycle_length:
            last_node = cycle[-1]
            next_node = min(unvisited, key=lambda node: self.distance_matrix[last_node, node])
            cycle.append(next_node)
            unvisited.remove(next_node)

        return cycle

class RegretCycleOptimizer(CycleOptimizer):
    def create_two_cycles(self):
        n = len(self.distance_matrix)
        half_n = n // 2 + n % 2

        start_node1 = random.randint(0, n - 1)
        start_node2 = max(range(n), key=lambda node: self.distance_matrix[start_node1, node])

        unvisited = set(range(n))
        unvisited.remove(start_node1)
        unvisited.remove(start_node2)

        cycle1 = self.regret_cycle(start_node1, half_n, unvisited)
        cycle2 = self.regret_cycle(start_node2, n - half_n, unvisited)

        return cycle1, cycle2

    def regret_cycle(self, start_node, cycle_length, unvisited):
        n = len(self.distance_matrix)
        cycle = [start_node]

        while len(cycle) < cycle_length:
            regrets = []
            for node in unvisited:
                best_increase = float('inf')
                second_best_increase = float('inf')
                for i in range(len(cycle)):
                    increase = self.distance_matrix[cycle[i], node] + self.distance_matrix[node, cycle[(i + 1) % len(cycle)]] - self.distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]]
                    if increase < best_increase:
                        second_best_increase = best_increase
                        best_increase = increase
                    elif increase < second_best_increase:
                        second_best_increase = increase
                regret = second_best_increase - best_increase
                regrets.append((regret, node))
            regrets.sort(reverse=True)
            next_node = regrets[0][1]
            best_position = min(range(len(cycle)), key=lambda i: self.distance_matrix[cycle[i], next_node] + self.distance_matrix[next_node, cycle[(i + 1) % len(cycle)]] - self.distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]])
            cycle.insert(best_position + 1, next_node)
            unvisited.remove(next_node)

        return cycle

class WeightedRegretCycleOptimizer(CycleOptimizer):
    def create_two_cycles(self, weight1=1, weight2=-1):
        n = len(self.distance_matrix)
        half_n = n // 2 + n % 2

        start_node1 = random.randint(0, n - 1)
        start_node2 = max(range(n), key=lambda node: self.distance_matrix[start_node1, node])

        unvisited = set(range(n))
        unvisited.remove(start_node1)
        unvisited.remove(start_node2)

        cycle1 = self.weighted_regret_cycle(start_node1, half_n, unvisited, weight1, weight2)
        cycle2 = self.weighted_regret_cycle(start_node2, n - half_n, unvisited, weight1, weight2)

        return cycle1, cycle2

    def weighted_regret_cycle(self, start_node, cycle_length, unvisited, weight1, weight2):
        n = len(self.distance_matrix)
        cycle = [start_node]

        while len(cycle) < cycle_length:
            regrets = []
            for node in unvisited:
                best_increase = float('inf')
                second_best_increase = float('inf')
                for i in range(len(cycle)):
                    increase = self.distance_matrix[cycle[i], node] + self.distance_matrix[node, cycle[(i + 1) % len(cycle)]] - self.distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]]
                    if increase < best_increase:
                        second_best_increase = best_increase
                        best_increase = increase
                    elif increase < second_best_increase:
                        second_best_increase = increase
                regret = weight1 * best_increase + weight2 * second_best_increase
                regrets.append((regret, node))
            regrets.sort(reverse=True)
            next_node = regrets[0][1]
            best_position = min(range(len(cycle)), key=lambda i: self.distance_matrix[cycle[i], next_node] + self.distance_matrix[next_node, cycle[(i + 1) % len(cycle)]] - self.distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]])
            cycle.insert(best_position + 1, next_node)
            unvisited.remove(next_node)

        return cycle
    
# Załaduj problemy z plików .tsp
optimizer_kroa200_greedy = GreedyCycleOptimizer('kroA200')
optimizer_krob200_greedy = GreedyCycleOptimizer('kroB200')

optimizer_kroa200_nn = NearestNeighborCycleOptimizer('kroA200')
optimizer_krob200_nn = NearestNeighborCycleOptimizer('kroB200')

optimizer_kroa200_regret = RegretCycleOptimizer('kroA200')
optimizer_krob200_regret = RegretCycleOptimizer('kroB200')

optimizer_kroa200_weighted_regret = WeightedRegretCycleOptimizer('kroA200')
optimizer_krob200_weighted_regret = WeightedRegretCycleOptimizer('kroB200')

# Utwórz dwa cykle dla kroA200 za pomocą różnych metod
cycle1_kroa200_greedy, cycle2_kroa200_greedy = optimizer_kroa200_greedy.create_two_cycles()
length1_kroa200_greedy = optimizer_kroa200_greedy.calculate_cycle_length(cycle1_kroa200_greedy)
length2_kroa200_greedy = optimizer_kroa200_greedy.calculate_cycle_length(cycle2_kroa200_greedy)

cycle1_kroa200_nn, cycle2_kroa200_nn = optimizer_kroa200_nn.create_two_cycles()
length1_kroa200_nn = optimizer_kroa200_nn.calculate_cycle_length(cycle1_kroa200_nn)
length2_kroa200_nn = optimizer_kroa200_nn.calculate_cycle_length(cycle2_kroa200_nn)

cycle1_kroa200_regret, cycle2_kroa200_regret = optimizer_kroa200_regret.create_two_cycles()
length1_kroa200_regret = optimizer_kroa200_regret.calculate_cycle_length(cycle1_kroa200_regret)
length2_kroa200_regret = optimizer_kroa200_regret.calculate_cycle_length(cycle2_kroa200_regret)

cycle1_kroa200_weighted_regret, cycle2_kroa200_weighted_regret = optimizer_kroa200_weighted_regret.create_two_cycles()
length1_kroa200_weighted_regret = optimizer_kroa200_weighted_regret.calculate_cycle_length(cycle1_kroa200_weighted_regret)
length2_kroa200_weighted_regret = optimizer_kroa200_weighted_regret.calculate_cycle_length(cycle2_kroa200_weighted_regret)

# Utwórz dwa cykle dla kroB200 za pomocą różnych metod
cycle1_krob200_greedy, cycle2_krob200_greedy = optimizer_krob200_greedy.create_two_cycles()
length1_krob200_greedy = optimizer_krob200_greedy.calculate_cycle_length(cycle1_krob200_greedy)
length2_krob200_greedy = optimizer_krob200_greedy.calculate_cycle_length(cycle2_krob200_greedy)

cycle1_krob200_nn, cycle2_krob200_nn = optimizer_krob200_nn.create_two_cycles()
length1_krob200_nn = optimizer_krob200_nn.calculate_cycle_length(cycle1_krob200_nn)
length2_krob200_nn = optimizer_krob200_nn.calculate_cycle_length(cycle2_krob200_nn)

cycle1_krob200_regret, cycle2_krob200_regret = optimizer_krob200_regret.create_two_cycles()
length1_krob200_regret = optimizer_krob200_regret.calculate_cycle_length(cycle1_krob200_regret)
length2_krob200_regret = optimizer_krob200_regret.calculate_cycle_length(cycle2_krob200_regret)

cycle1_krob200_weighted_regret, cycle2_krob200_weighted_regret = optimizer_krob200_weighted_regret.create_two_cycles()
length1_krob200_weighted_regret = optimizer_krob200_weighted_regret.calculate_cycle_length(cycle1_krob200_weighted_regret)
length2_krob200_weighted_regret = optimizer_krob200_weighted_regret.calculate_cycle_length(cycle2_krob200_weighted_regret)

# Wyświetl wyniki
print("Cycles for kroA200 using Greedy Cycle:")
print("Cycle 1:", cycle1_kroa200_greedy, "Length:", length1_kroa200_greedy)
print("Cycle 2:", cycle2_kroa200_greedy, "Length:", length2_kroa200_greedy)

print("\nCycles for kroB200 using Greedy Cycle:")
print("Cycle 1:", cycle1_krob200_greedy, "Length:", length1_krob200_greedy)
print("Cycle 2:", cycle2_krob200_greedy, "Length:", length2_krob200_greedy)

print("\nCycles for kroA200 using Nearest Neighbor:")
print("Cycle 1:", cycle1_kroa200_nn, "Length:", length1_kroa200_nn)
print("Cycle 2:", cycle2_kroa200_nn, "Length:", length2_kroa200_nn)

print("\nCycles for kroB200 using Nearest Neighbor:")
print("Cycle 1:", cycle1_krob200_nn, "Length:", length1_krob200_nn)
print("Cycle 2:", cycle2_krob200_nn, "Length:", length2_krob200_nn)

print("\nCycles for kroA200 using 2-Regret:")
print("Cycle 1:", cycle1_kroa200_regret, "Length:", length1_kroa200_regret)
print("Cycle 2:", cycle2_kroa200_regret, "Length:", length2_kroa200_regret)

print("\nCycles for kroB200 using 2-Regret:")
print("Cycle 1:", cycle1_krob200_regret, "Length:", length1_krob200_regret)
print("Cycle 2:", cycle2_krob200_regret, "Length:", length2_krob200_regret)

print("\nCycles for kroA200 using Weighted 2-Regret:")
print("Cycle 1:", cycle1_kroa200_weighted_regret, "Length:", length1_kroa200_weighted_regret)
print("Cycle 2:", cycle2_kroa200_weighted_regret, "Length:", length2_kroa200_weighted_regret)

print("\nCycles for kroB200 using Weighted 2-Regret:")
print("Cycle 1:", cycle1_krob200_weighted_regret, "Length:", length1_krob200_weighted_regret)
print("Cycle 2:", cycle2_krob200_weighted_regret, "Length:", length2_krob200_weighted_regret)

plt.figure(figsize=(16, 12))

# kroA200
optimizer_kroa200_greedy.plot_cycles(cycle1_kroa200_greedy, cycle2_kroa200_greedy, "Greedy Cycle - kroA200", 241)
optimizer_kroa200_nn.plot_cycles(cycle1_kroa200_nn, cycle2_kroa200_nn, "Nearest Neighbor - kroA200", 242)
optimizer_kroa200_regret.plot_cycles(cycle1_kroa200_regret, cycle2_kroa200_regret, "2-Regret - kroA200", 243)
optimizer_kroa200_weighted_regret.plot_cycles(cycle1_kroa200_weighted_regret, cycle2_kroa200_weighted_regret, "Weighted 2-Regret - kroA200", 244)

# kroB200
optimizer_krob200_greedy.plot_cycles(cycle1_krob200_greedy, cycle2_krob200_greedy, "Greedy Cycle - kroB200", 245)
optimizer_krob200_nn.plot_cycles(cycle1_krob200_nn, cycle2_krob200_nn, "Nearest Neighbor - kroB200", 246)
optimizer_krob200_regret.plot_cycles(cycle1_krob200_regret, cycle2_krob200_regret, "2-Regret - kroB200", 247)
optimizer_krob200_weighted_regret.plot_cycles(cycle1_krob200_weighted_regret, cycle2_krob200_weighted_regret, "Weighted 2-Regret - kroB200", 248)

plt.tight_layout()
plt.show()
