from typing import Tuple, List

import numpy as np


class LocalSearch:
    def __init__(self, dist_mtx: List, flow_mtx: List):
        self.dist_mtx = np.asarray(dist_mtx)
        self.flow_mtx = np.asarray(flow_mtx)
        self.N = self.dist_mtx.shape[0]

    def total_cost(self, solution: np.ndarray) -> float:
        dist_solution = self.dist_mtx[solution][:, solution]
        return np.sum(self.flow_mtx * dist_solution)

    def delta_cost(self, solution: np.ndarray, r: int, s: int) -> float:
        if r == s:
            return 0
        f_r, f_s = solution[r], solution[s]

        all_indices = np.arange(self.N)
        mask = (all_indices != r) & (all_indices != s)
        k = all_indices[mask]
        f_k = solution[k]

        delta = 2 * np.sum(
            (self.flow_mtx[s, k] - self.flow_mtx[r, k]) *
            (self.dist_mtx[f_s, f_k] - self.dist_mtx[f_r, f_k]))
        return delta

    def solve(self, solution: np.array = None) -> Tuple[np.ndarray, float]:
        """Local search - first-improvement + don't look bits"""
        if solution is None:
            solution = np.random.permutation(self.N)
        else:
            solution = np.array(solution)

        total_cost = self.total_cost(solution)
        dont_look_bits = np.zeros(self.N, dtype=bool)

        improved = True
        while improved:
            improved = False
            for s in range(self.N):
                if dont_look_bits[s]:
                    continue
                for r in range(self.N):
                    if s == r:
                        continue
                    delta = self.delta_cost(solution, r, s)
                    if delta < 0:
                        solution[r], solution[s] = solution[s], solution[r]
                        total_cost += delta
                        dont_look_bits[r] = False
                        dont_look_bits[s] = False
                        improved = True
                        break
                else:
                    dont_look_bits[s] = True
        return solution, total_cost

    def save_solution(self, file_path: str, solution: np.array) -> None:
        with open(file_path, 'w') as file:
            file.write(" ".join(map(str, solution)))


def read_qap_data(file_path: str) -> Tuple[List, List]:
    with open(file_path, 'r') as file:
        lines = [line for line in file]

    n = int(lines[0])
    dist = [list(map(int, lines[i].split())) for i in range(1, n + 1)]
    flow = [list(map(int, lines[i].split())) for i in range(n + 2, 2 * n + 2)]

    return dist, flow


if __name__ == '__main__':
    # python -m src.local_search
    dist_list, flow_list = read_qap_data('benchmarks/tai20a')

    ls = LocalSearch(dist_list, flow_list)

    sol, cost = ls.solve()
    ls.save_solution("tai20a.sol", sol)
    print("Solution:", sol)
    print("Cost:", cost)
