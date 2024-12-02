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
    solver = Solver()  # A_squared = np.dot(A, A)

    # (1) 基础条件；
    for _i in range(_n):
        for _j in range(_n):
            if _i >= _j:
                solver.add(A[_i][_j] == 0)
            else:
                solver.add(Or(A[_i][_j] == 0, A[_i][_j] == 1))

    # (2) 所有置换矩阵穷举；
    for _dag_x in _DAGs:
        adj_matrix = nx.to_numpy_array(_dag_x, dtype=int)
        for _pre_mx in generate_permutation_matrices(_n):
            result = np.dot(np.dot(_pre_mx, adj_matrix), _pre_mx.T)  # 计算 P * A * P^T
            solver.add(Or([A[_i][_j] != result[_i][_j] for _i in range(_n) for _j in range(_n)]))

    # (3) 去冗余；
    D_n = np.linalg.matrix_power(A + np.eye(_n), _n) - np.eye(_n)
    X_n = np.dot(A, D_n)
    for _i in range(_n):
        for _j in range(_n):
            solver.add(Or(A[_i][_j] == 0, X_n[_i][_j] == 0))
    # solver.add(And(Or(A[_i][_j] == 0, X_n[_i][_j] == 0) for _i in range(_n) for _j in range(_n)))
    # solver.add(Or([A[_i][_j] != result[_i][_j] for _i in range(_n) for _j in range(_n)]))

    if solver.check() == sat:
        model = solver.model()
        for i in range(_n):
            print( [model.evaluate(A[i][j]) for j in range(_n)])
        print("有解")
    else:
        print("无解")


if __name__ == "__main__":
    """ shape 穷举测试 """
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

# n = 8
# tt = 0
# for matrix in generate_permutation_matrices(n):
#     # print(matrix)
#     # print()
#     tt += 1
# print(tt)



# def solve_01_matrix(n):
#     # 创建一个 n x n 的 01 矩阵变量
#     matrix = [[Bool(f"x_{i}_{j}") for j in range(n)] for i in range(n)]    
#     solver = Solver()
#     # 添加 01 的约束条件
#     for i in range(n):
#         for j in range(n):
#             solver.add(Or(matrix[i][j] == True, matrix[i][j] == False))
    
#     # 示例：添加每行至多有 2 个 1 的约束
#     for row in matrix:
#         solver.add(Sum([If(cell, 1, 0) for cell in row]) <= 2)

#     # 求解
#     if solver.check() == sat:
#         model = solver.model()
#         # 提取解
#         solution = [[1 if model[cell] else 0 for cell in row] for row in matrix]
#         return solution
#     else:
#         return None

# # 示例：求解 4 阶 01 矩阵
# n = 4
# result = solve_01_matrix(n)
# if result:
#     for row in result:
#         print(row)
# else:
#     print("No solution found.")