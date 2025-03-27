import tsplib95
import numpy as np
import random
import matplotlib.pyplot as plt

class CycleOptimizer:
    def __init__(self, problem_name):
        self.problem = tsplib95.load(f"{problem_name}.tsp")
        self.nodes = list(self.problem.get_nodes())
        self.distance_matrix = self._compute_distance_matrix()
        self.coords = np.array([self.problem.node_coords[i] for i in self.nodes])

    def _compute_distance_matrix(self):
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.problem.get_weight(self.nodes[i], self.nodes[j])
        return matrix

    def cycle_length(self, cycle):
        return sum(
            self.distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]]
            for i in range(len(cycle))
        )
        
    def plot_cycles(self, cycle1, cycle2, title=""):
        plt.figure(figsize=(8, 6))
        coords = self.coords

        # Draw cycle 1
        for i in range(len(cycle1)):
            a = coords[cycle1[i]]
            b = coords[cycle1[(i + 1) % len(cycle1)]]
            plt.plot([a[0], b[0]], [a[1], b[1]], 'b-', linewidth=1.5)
        plt.scatter(coords[cycle1, 0], coords[cycle1, 1], c='blue', label='Cycle 1')

        # Draw cycle 2
        for i in range(len(cycle2)):
            a = coords[cycle2[i]]
            b = coords[cycle2[(i + 1) % len(cycle2)]]
            plt.plot([a[0], b[0]], [a[1], b[1]], 'orange', linewidth=1.5)
        plt.scatter(coords[cycle2, 0], coords[cycle2, 1], c='orange', label='Cycle 2')

        # Draw labels
        for i, (x, y) in enumerate(coords):
            plt.text(x, y, str(i), fontsize=6, ha='center', va='center')

        plt.title(title)
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()

class BalancedCycleOptimizer(CycleOptimizer):
    def insert_best(self, cycle, unvisited, strategy, regret_mode=False):
        best_score = float('inf')
        best_node = None
        best_pos = None

        for node in unvisited:
            insertions = []
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                score = strategy(a, node, b)
                insertions.append((score, i + 1))
            insertions.sort()

            if len(insertions) >= 2:
                regret = insertions[1][0] - insertions[0][0]
            else:
                regret = 0

            if regret_mode:
                total_score = -regret
            else:
                total_score = insertions[0][0]

            if total_score < best_score:
                best_score = total_score
                best_node = node
                best_pos = insertions[0][1]

        return best_node, best_pos

    def insert_best_nn(self, cycle, unvisited, strategy):
        best_score = float('inf')
        best_node = None
        last_node = cycle[-1]

        for node in unvisited:
            score = strategy(last_node, node)

            if score < best_score:
                best_score = score
                best_node = node

        return best_node
    
    def insert_best_weighted(self, cycle, unvisited, strategy, w=-0.5):
        best_score = float('inf')
        best_node = None
        best_pos = None

        for node in unvisited:
            insertions = []
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                score = strategy(a, node, b)
                insertions.append((score, i + 1))
            insertions.sort()

            if len(insertions) >= 2:
                regret = insertions[1][0] - insertions[0][0]
            else:
                regret = 0
            
            total_score = ((1 + w) * insertions[0][0]) + (w * regret)

            if total_score < best_score:
                best_score = total_score
                best_node = node
                best_pos = insertions[0][1]

        return best_node, best_pos

    def build_cycles_weighted_regret(self, strategy, w=-0.5):
        n = len(self.nodes)
        t1, t2 = n // 2 + n % 2, n // 2
        s1 = random.randint(0, n - 1)
        s2 = max(range(n), key=lambda i: self.distance_matrix[s1][i])

        c1 = [s1, min(range(n), key=lambda i: self.distance_matrix[s1][i] if i != s1 else float('inf'))]
        c2 = [s2, min(range(n), key=lambda i: self.distance_matrix[s2][i] if i != s2 else float('inf'))]

        unvisited = set(range(n)) - {s1, s2, c1[1], c2[1]}

        while len(c1) < t1 or len(c2) < t2:
            if len(c1) < t1:
                node, pos = self.insert_best_weighted(c1, unvisited, strategy, w)
                c1.insert(pos, node)
                unvisited.remove(node)
            if len(c2) < t2 and unvisited:
                node, pos = self.insert_best_weighted(c2, unvisited, strategy, w)
                c2.insert(pos, node)
                unvisited.remove(node)

        return c1, c2
    
    def build_cycles(self, strategy, regret_mode=False):
        n = len(self.nodes)
        t1, t2 = n // 2 + n % 2, n // 2
        s1 = random.randint(0, n - 1)
        s2 = max(range(n), key=lambda i: self.distance_matrix[s1][i])

        c1 = [s1, min(range(n), key=lambda i: self.distance_matrix[s1][i] if i != s1 else float('inf'))]
        c2 = [s2, min(range(n), key=lambda i: self.distance_matrix[s2][i] if i != s2 else float('inf'))]

        unvisited = set(range(n)) - {s1, s2, c1[1], c2[1]}

        while len(c1) < t1 or len(c2) < t2:
            if len(c1) < t1:
                node, pos = self.insert_best(c1, unvisited, strategy, regret_mode)
                c1.insert(pos, node)
                unvisited.remove(node)
            if len(c2) < t2 and unvisited:
                node, pos = self.insert_best(c2, unvisited, strategy, regret_mode)
                c2.insert(pos, node)
                unvisited.remove(node)

        return c1, c2

    def build_cycles_nn(self, strategy, regret_mode=False):
        n = len(self.nodes)
        t1, t2 = n // 2 + n % 2, n // 2
        s1 = random.randint(0, n - 1)
        s2 = max(range(n), key=lambda i: self.distance_matrix[s1][i])

        c1 = [s1] 
        c2 = [s2]

        unvisited = set(range(n)) - {s1, s2}

        while len(c1) < t1 or len(c2) < t2:
            if len(c1) < t1:
                node = self.insert_best_nn(c1, unvisited, strategy)
                c1.append(node)
                unvisited.remove(node)
            if len(c2) < t2 and unvisited:
                node = self.insert_best_nn(c2, unvisited, strategy)
                c2.append(node)
                unvisited.remove(node)

        return c1, c2

# === ALGORYTMY ===

class Greedy(BalancedCycleOptimizer):
    def create_two_cycles(self):
        def strategy(a, n, b):
            return self.distance_matrix[a][n] + self.distance_matrix[n][b] - self.distance_matrix[a][b]
        return self.build_cycles(strategy, regret_mode=False)

class NearestNeighbor(BalancedCycleOptimizer):
    def create_two_cycles(self):
        def strategy(a, n):
            return self.distance_matrix[a][n]
        return self.build_cycles_nn(strategy, regret_mode=False)

class TwoRegret(BalancedCycleOptimizer):
    def create_two_cycles(self):
        def strategy(a, n, b):
            return self.distance_matrix[a][n] + self.distance_matrix[n][b] - self.distance_matrix[a][b]
        return self.build_cycles(strategy, regret_mode=True)

class Weighted2Regret(BalancedCycleOptimizer):
    def create_two_cycles(self, w=-0.5):
        def strategy(a, n, b):
            return self.distance_matrix[a][n] + self.distance_matrix[n][b] - self.distance_matrix[a][b]
        return self.build_cycles_weighted_regret(strategy, w=w)

# === EKSPERYMENTY ===

def run_experiment(optimizer_class, instance, label):
    results = []
    for i in range(100):
        optimizer = optimizer_class(instance)
        if label == "Weighted 2-Regret":
            c1, c2 = optimizer.create_two_cycles(w=-0.5)
        else:
            c1, c2 = optimizer.create_two_cycles()
        total = optimizer.cycle_length(c1) + optimizer.cycle_length(c2)
        results.append([total, c1, c2])
        
    return results

def full_table():
    instances = ["kroA200", "kroB200"]
    methods = [
        (Weighted2Regret, "Weighted 2-Regret"),
        (NearestNeighbor, "Nearest Neighbor"),
        (TwoRegret, "2-Regret"),
        (Greedy, "Greedy"),
    ]

    # Save the results to a file
    with open("results.txt", "w") as f:
        f.write(f"method,instance,total,cycle1,cycle2\n")

    print(f"{'':<20} {'kroA200':<30} {'kroB200'}")
    print("=" * 80)
    for cls, label in methods:
        res_a = run_experiment(cls, "kroA200", label)
        res_b = run_experiment(cls, "kroB200", label)
        
        mean_res_a = round(sum(x[0] for x in res_a) / len(res_a), 2)
        mean_res_b = round(sum(x[0] for x in res_b) / len(res_b), 2)

        print(f"{label:<20} {mean_res_a:<30} {mean_res_b}")

        # Save the results to a file
        with open("results.txt", "a") as f:
            for total, c1, c2 in res_a:
                f.write(f"{label},kroA200,{total},{c1},{c2}\n")
            for total, c1, c2 in res_b:
                f.write(f"{label},kroB200,{total},{c1},{c2}\n")


def visualize_all_cycles():
    instances = ["kroA200", "kroB200"]
    methods = [
        (Greedy, "Greedy"),
        (NearestNeighbor, "Nearest Neighbor"),
        (TwoRegret, "2-Regret"),
        (Weighted2Regret, "Weighted 2-Regret")
    ]

    fig, axes = plt.subplots(len(instances), len(methods), figsize=(16, 8))
    fig.suptitle("Cycle Visualizations", fontsize=18)

    for i, instance in enumerate(instances):
        for j, (cls, label) in enumerate(methods):
            random.seed(42) # czemu?
            optimizer = cls(instance)
            if label == "Weighted 2-Regret":
                c1, c2 = optimizer.create_two_cycles(w=-0.5)
            else:
                c1, c2 = optimizer.create_two_cycles()

            ax = axes[i][j] if len(instances) > 1 else axes[j]
            coords = optimizer.coords

            # CYCLE 1 - blue
            for k in range(len(c1)):
                a = coords[c1[k]]
                b = coords[c1[(k + 1) % len(c1)]]
                ax.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=1)
            ax.scatter(coords[c1, 0], coords[c1, 1], color='blue', edgecolors='black', s=10, zorder=3)

            # CYCLE 2 - red
            for k in range(len(c2)):
                a = coords[c2[k]]
                b = coords[c2[(k + 1) % len(c2)]]
                ax.plot([a[0], b[0]], [a[1], b[1]], color='red', linewidth=1)
            ax.scatter(coords[c2, 0], coords[c2, 1], color='red', edgecolors='black', s=10, zorder=3)

            # Node labels
            for idx, (x, y) in enumerate(coords):
                ax.text(x, y, str(idx), fontsize=4, ha='center', va='center')

            ax.set_title(f"{label} - {instance}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(coords[:, 0].min() - 50, coords[:, 0].max() + 50)
            ax.set_ylim(coords[:, 1].min() - 50, coords[:, 1].max() + 50)
            ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


full_table()
visualize_all_cycles()
