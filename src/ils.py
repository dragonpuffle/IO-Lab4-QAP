from typing import List, Tuple

import numpy as np

from src.local_search import LocalSearch, read_qap_data


class IteratedLocalSearch(LocalSearch):
    def __init__(self, dist_mtx: List, flow_mtx: List, max_iters: int = 100):
        super().__init__(dist_mtx, flow_mtx)
        self.max_iters = max_iters

    def perturbation(self, solution: np.ndarray, dont_look_bits) -> np.ndarray:
        k = max(2, self.N // 10)
        indices = np.arange(self.N)

        if dont_look_bits is not None and not np.all(dont_look_bits):
            mask = ~dont_look_bits
        else:
            mask = np.ones(self.N, dtype=bool)

        candidates = indices[mask]
        if len(candidates) < k:
            others = indices[~mask]
            additional_candidates = np.random.choice(others, size=k - len(candidates), replace=False)
            candidates = np.concatenate((candidates, additional_candidates))

        perturbation_indices = np.random.choice(candidates, size=k, replace=False)

        new_solution = solution.copy()
        np.random.shuffle(new_solution[perturbation_indices])

        return new_solution


    def iterated_solve(self, solution: np.array = None) -> Tuple[np.ndarray, float]:
        if solution is None:
            solution = np.random.permutation(self.N)
        else:
            solution = np.array(solution)

        current_solution, current_cost, current_dont_look_bits = self.solve(solution)
        best_solution = current_solution.copy()
        best_cost = current_cost

        for _ in range(self.max_iters):
            perturbation_solution = self.perturbation(current_solution, current_dont_look_bits)
            new_solution, new_cost, new_dont_look_bits = self.solve(perturbation_solution)

            if new_cost < best_cost:
                best_solution = new_solution.copy()
                best_cost = new_cost

            current_solution = new_solution
            current_dont_look_bits = new_dont_look_bits

        return best_solution, best_cost


if __name__ == "__main__":
    dist_list, flow_list = read_qap_data('benchmarks/tai20a')

    ils = IteratedLocalSearch(dist_list, flow_list, max_iters=100)

    sol, cost = ils.iterated_solve()
    print("Solution:", sol)
    print("Cost:", cost)