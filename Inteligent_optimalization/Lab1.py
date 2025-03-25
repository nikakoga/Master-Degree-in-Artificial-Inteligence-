import tsplib95
import numpy as np
import random
import matplotlib.pyplot as plt

# Funkcja ładująca problem z pliku .tsp
def load_problem(problem_name):
    return tsplib95.load(f'{problem_name}.tsp')

# Funkcja obliczająca macierz odległości dla danego problemu
def calculate_distance_matrix(problem):
    nodes = list(problem.get_nodes())  # Pobierz listę węzłów
    n = len(nodes)  # Liczba węzłów
    distance_matrix = np.zeros((n, n))  # Inicjalizacja macierzy odległości zerami
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = problem.get_weight(nodes[i], nodes[j])  # Oblicz odległość między węzłami
    return distance_matrix

# Funkcja tworząca cykl metodą rozbudowy cyklu
def greedy_cycle(distance_matrix, start_node, cycle_length, unvisited):
    n = len(distance_matrix)  # Liczba węzłów
    cycle = [start_node]  # Inicjalizacja cyklu węzłem startowym
    unvisited.remove(start_node)  # Usuń węzeł startowy ze zbioru nieodwiedzonych

    while len(cycle) < cycle_length:
        # Znajdź najbliższy węzeł do dowolnego węzła w cyklu
        next_node = min(unvisited, key=lambda node: min(distance_matrix[node, cycle[i]] for i in range(len(cycle))))
        # Znajdź najlepsze miejsce do wstawienia nowego węzła w cyklu
        best_position = min(range(len(cycle)), key=lambda i: distance_matrix[cycle[i], next_node] + distance_matrix[next_node, cycle[(i + 1) % len(cycle)]] - distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]])
        cycle.insert(best_position + 1, next_node)  # Wstaw nowy węzeł do cyklu
        unvisited.remove(next_node)  # Usuń nowy węzeł ze zbioru nieodwiedzonych

    return cycle

# Funkcja tworząca cykl metodą najbliższego sąsiada
def nearest_neighbor_cycle(distance_matrix, start_node, cycle_length, unvisited):
    cycle = [start_node]
    unvisited.remove(start_node)
    
    while len(cycle) < cycle_length:
        last_node = cycle[-1]
        next_node = min(unvisited, key=lambda node: distance_matrix[last_node, node])
        cycle.append(next_node)
        unvisited.remove(next_node)
    
    return cycle

# Funkcja tworząca dwa rozłączne cykle
def create_two_cycles(distance_matrix, method):
    n = len(distance_matrix)  # Liczba węzłów
    half_n = n // 2 + n % 2  # Liczba węzłów w pierwszym cyklu (jeśli n jest nieparzyste, pierwszy cykl ma jeden węzeł więcej)

    # Wybierz dwa węzły startowe
    start_node1 = random.randint(0, n - 1)  # Losowy węzeł startowy dla pierwszego cyklu
    start_node2 = max(range(n), key=lambda node: distance_matrix[start_node1, node])  # Najodleglejszy węzeł od pierwszego węzła startowego

    # Zbiór nieodwiedzonych węzłów
    unvisited = set(range(n))

    # Utwórz dwa cykle
    if method == 'greedy_cycle':
        cycle1 = greedy_cycle(distance_matrix, start_node1, half_n, unvisited)  # Pierwszy cykl
        cycle2 = greedy_cycle(distance_matrix, start_node2, n - half_n, unvisited)  # Drugi cykl
    elif method == 'nearest_neighbor':
        cycle1 = nearest_neighbor_cycle(distance_matrix, start_node1, half_n, unvisited)  # Pierwszy cykl
        cycle2 = nearest_neighbor_cycle(distance_matrix, start_node2, n - half_n, unvisited)  # Drugi cykl

    return cycle1, cycle2

# Funkcja obliczająca długość cyklu
def calculate_cycle_length(distance_matrix, cycle):
    length = 0
    for i in range(len(cycle)):
        length += distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]]  # Dodaj odległość między kolejnymi węzłami w cyklu
    return length

# Funkcja rysująca cykle na wykresie
def plot_cycles(problem, cycle1, cycle2, title, subplot_position):
    nodes = list(problem.get_nodes())
    coordinates = np.array([problem.node_coords[node] for node in nodes])

    plt.subplot(subplot_position)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='black')

    def plot_cycle(cycle, color):
        cycle_coords = np.array([coordinates[node] for node in cycle])
        cycle_coords = np.vstack([cycle_coords, cycle_coords[0]])  # Zamknij cykl
        plt.plot(cycle_coords[:, 0], cycle_coords[:, 1], color=color)

    plot_cycle(cycle1, 'blue')
    plot_cycle(cycle2, 'red')

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

# Załaduj problemy z plików .tsp
kroa200 = load_problem('kroA200')
krob200 = load_problem('kroB200')

# Oblicz macierze odległości
distance_matrix_kroa200 = calculate_distance_matrix(kroa200)
distance_matrix_krob200 = calculate_distance_matrix(krob200)

# Utwórz dwa cykle dla kroA200 za pomocą metody rozbudowy cyklu
cycle1_kroa200_greedy, cycle2_kroa200_greedy = create_two_cycles(distance_matrix_kroa200, 'greedy_cycle')
length1_kroa200_greedy = calculate_cycle_length(distance_matrix_kroa200, cycle1_kroa200_greedy)
length2_kroa200_greedy = calculate_cycle_length(distance_matrix_kroa200, cycle2_kroa200_greedy)

# Utwórz dwa cykle dla kroB200 za pomocą metody rozbudowy cyklu
cycle1_krob200_greedy, cycle2_krob200_greedy = create_two_cycles(distance_matrix_krob200, 'greedy_cycle')
length1_krob200_greedy = calculate_cycle_length(distance_matrix_krob200, cycle1_krob200_greedy)
length2_krob200_greedy = calculate_cycle_length(distance_matrix_krob200, cycle2_krob200_greedy)

# Utwórz dwa cykle dla kroA200 za pomocą metody najbliższego sąsiada
cycle1_kroa200_nn, cycle2_kroa200_nn = create_two_cycles(distance_matrix_kroa200, 'nearest_neighbor')
length1_kroa200_nn = calculate_cycle_length(distance_matrix_kroa200, cycle1_kroa200_nn)
length2_kroa200_nn = calculate_cycle_length(distance_matrix_kroa200, cycle2_kroa200_nn)

# Utwórz dwa cykle dla kroB200 za pomocą metody najbliższego sąsiada
cycle1_krob200_nn, cycle2_krob200_nn = create_two_cycles(distance_matrix_krob200, 'nearest_neighbor')
length1_krob200_nn = calculate_cycle_length(distance_matrix_krob200, cycle1_krob200_nn)
length2_krob200_nn = calculate_cycle_length(distance_matrix_krob200, cycle2_krob200_nn)

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

# Rysuj cykle
plt.figure(figsize=(14, 14))
plot_cycles(kroa200, cycle1_kroa200_greedy, cycle2_kroa200_greedy, "Cycles for kroA200 using Greedy Cycle", 221)
plot_cycles(krob200, cycle1_krob200_greedy, cycle2_krob200_greedy, "Cycles for kroB200 using Greedy Cycle", 222)
plot_cycles(kroa200, cycle1_kroa200_nn, cycle2_kroa200_nn, "Cycles for kroA200 using Nearest Neighbor", 223)
plot_cycles(krob200, cycle1_krob200_nn, cycle2_krob200_nn, "Cycles for kroB200 using Nearest Neighbor", 224)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()