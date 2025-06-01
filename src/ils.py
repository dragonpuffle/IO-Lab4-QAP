from typing import List, Tuple

import numpy as np

from src.local_search import LocalSearch, read_qap_data


class IteratedLocalSearch(LocalSearch):
    def __init__(self, dist_mtx: List, flow_mtx: List, max_iters: int = 200):
        super().__init__(dist_mtx, flow_mtx)
        self.max_iters = max_iters

    def perturbation(self, solution: np.ndarray) -> np.ndarray:
        new_solution = solution.copy()
        i, j = sorted(np.random.choice(self.N, size=2, replace=False))
        new_solution[i:j + 1] = new_solution[i:j + 1][::-1]

        return new_solution

    def iterated_solve(self, solution: np.array = None) -> Tuple[np.ndarray, float]:
        if solution is None:
            solution = np.random.permutation(self.N)
        else:
            solution = np.array(solution)

        current_solution, current_cost = self.solve(solution)
        best_solution = current_solution.copy()
        best_cost = current_cost

        no_improve_count = 0
        for iteration in range(self.max_iters):
            perturbation_solution = self.perturbation(current_solution)
            new_solution, new_cost,  = self.solve(perturbation_solution)

            if new_cost < best_cost:
                best_solution = new_solution.copy()
                best_cost = new_cost
                no_improve_count = 0
            else:
                no_improve_count += 1

            current_solution = new_solution

        return best_solution, best_cost

    def save_solution(self, file_path: str, solution: np.array) -> None:
        with open(file_path, 'w') as file:
            file.write(" ".join(map(str, solution)))


if __name__ == "__main__":
    dist_list, flow_list = read_qap_data('benchmarks/tai100a')

    ils = IteratedLocalSearch(dist_list, flow_list, max_iters=200)

    sol, cost = ils.iterated_solve()
    print("Solution:", sol)
    print("Cost:", cost)