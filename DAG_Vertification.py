from z3 import *
import numpy as np
import networkx as nx
from itertools import permutations

def generate_permutation_matrices(_n):
    for p in permutations(range(_n)):
        matrix = np.zeros((_n, _n), dtype=int)
        for i in range(_n):
            matrix[i][p[i]] = 1
        yield matrix



def DAG_Vertification(_n, _DAGs):
    A = np.array([[Int(f"a_{i}_{j}") for j in range(_n)] for i in range(_n)])
    solver = Solver()

    for _i in range(_n):
        for _j in range(_n):
            if _i >= _j:
                solver.add(A[_i][_j] == 0)
            else:
                solver.add(Or(A[_i][_j] == 0, A[_i][_j] == 1))

    for _dag_x in _DAGs:
        adj_matrix = nx.to_numpy_array(_dag_x, dtype=int)
        for _pre_mx in generate_permutation_matrices(_n):
            result = np.dot(np.dot(_pre_mx, adj_matrix), _pre_mx.T)
            solver.add(Or([A[_i][_j] != result[_i][_j] for _i in range(_n) for _j in range(_n)]))

    D_n = np.linalg.matrix_power(A + np.eye(_n), _n) - np.eye(_n)
    X_n = np.dot(A, D_n)
    for _i in range(_n):
        for _j in range(_n):
            solver.add(Or(A[_i][_j] == 0, X_n[_i][_j] == 0))

    if solver.check() == sat:
        model = solver.model()
        for i in range(_n):
            print( [model.evaluate(A[i][j]) for j in range(_n)])
        print("solution")
    else:
        print("no solution")


if __name__ == "__main__":
    _n = 3
    dag_list = []
    for edges in [[],   [(0, 1)],
                  [(0, 1), (0, 2)],
                  # [(0, 2), (1, 2)],
                  [(0, 1), (1, 2)]]:
        temp_dag = nx.DiGraph()
        temp_dag.add_nodes_from(range(_n))
        temp_dag.add_edges_from(edges)
        dag_list.append(temp_dag)
    DAG_Vertification(3, dag_list)
