from typing import List, Tuple

import numpy as np

from src.local_search import LocalSearch, read_qap_data


class IteratedLocalSearch(LocalSearch):
    def __init__(self, dist_mtx: List, flow_mtx: List, max_iters: int = 100):
        super().__init__(dist_mtx, flow_mtx)
        self.max_iters = max_iters
        self.dont_look_bits = np.zeros(self.N, dtype=int)

    def perturbation(self, solution: np.ndarray, current_iter: int) -> np.ndarray:
        k = max(4, self.N // 5)
        possible_indices = np.argsort(self.dont_look_bits)[:2*k]
        perturbation_indices = np.random.choice(possible_indices, size=k, replace=False)
        self.dont_look_bits[perturbation_indices] = current_iter

        new_solution = solution.copy()
        np.random.shuffle(new_solution[perturbation_indices])

        return new_solution


    def iterated_solve(self, solution: np.array = None) -> Tuple[np.ndarray, float]:
        if solution is None:
            solution = np.random.permutation(self.N)
        else:
            solution = np.array(solution)

        current_solution, current_cost = self.solve(solution)
        best_solution = current_solution.copy()
        best_cost = current_cost

        for iteration in range(self.max_iters):
            perturbation_solution = self.perturbation(current_solution, iteration)
            new_solution, new_cost = self.solve(perturbation_solution)

            if new_cost < best_cost:
                best_solution = new_solution.copy()
                best_cost = new_cost
            current_solution = new_solution

        return best_solution, best_cost


if __name__ == "__main__":
    dist_list, flow_list = read_qap_data('benchmarks/tai20a')

    ils = IteratedLocalSearch(dist_list, flow_list, max_iters=100)

    sol, cost = ils.iterated_solve()
    print("Solution:", sol)
    print("Cost:", cost)