import os
import time
import pandas as pd
from src.local_search import *
from src.ils import IteratedLocalSearch


class Benchmark:
    def __init__(self, algorithm_classes, bench_dir: str, runs: int = 3):
        self.algorithms = algorithm_classes
        self.runs = runs
        self.bench_dir = bench_dir
        self.results = []

    def run_all(self):
        files = [f for f in os.listdir(self.bench_dir)]
        for file in sorted(files):
            print(f'Benchmarking {file} ...')
            full_path = os.path.join(self.bench_dir, file)
            self.run_one(full_path, file)

        df = pd.DataFrame(self.results).sort_values(by=['benchmark_id', 'alg'], ascending=True)
        df.to_csv('results.csv', index=False)

    def run_one(self, path: str, benchmark: str):
        dist_list, flow_list = read_qap_data(path)

        for algorithm in self.algorithms:
            best_total_cost = float('inf')
            best_solution = []
            total_time = 0
            for _ in range(self.runs):
                alg = algorithm(dist_list, flow_list)
                start = time.time()

                if isinstance(alg, IteratedLocalSearch):
                    solution, total_cost = alg.iterated_solve()
                else:
                    solution, total_cost = alg.solve()

                total_time += time.time() - start
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_solution = solution.copy()

            avg_time = (total_time) / self.runs
            self.results.append({
                'benchmark_id': int(benchmark[3:-1]),
                'alg': algorithm.__name__,
                'benchmark': benchmark,
                'best_total_cost': round(best_total_cost, 7),
                'avg_time_sec': round(avg_time, 7),
                'solution': " ".join(map(str, best_solution))
            })
            if algorithm is IteratedLocalSearch:
                self.save_solution("results/"+benchmark + ".sol", best_solution)

    def save_solution(self, file_path: str, solution: np.array) -> None:
        with open(file_path, 'w') as file:
            file.write(" ".join(map(str, solution)))


if __name__ == '__main__':
    benchmark = Benchmark((LocalSearch, IteratedLocalSearch), 'benchmarks', 20)
    benchmark.run_all()

