import time
import numpy as np
from tqdm import tqdm 
from scipy.optimize import brentq, minimize, LinearConstraint
from collections import namedtuple


# --- More Efficient Optimization Problem Class ---
class OptimizationProblem:
    """
    Encapsulates the optimization problem with efficient dimensionality reduction.
    Objective and gradient now operate ONLY on the effective dimension.
    """
    def __init__(self, p: np.ndarray, alpha: float, threshold: float = 1e-5):
        if not np.isclose(np.sum(p), 1.0):
            raise ValueError("p vector must be normalized (sum to 1).")
        if not np.all(p >= 0):
            raise ValueError("All elements of p vector must be non-negative.")

        self.p_full = p
        self.K_full = len(p)
        self.alpha = alpha
        self.target_sum = 1 - alpha
        
        if threshold is None:
            self.K = self.K_full
            self.p = self.p_full
            print("Full-dimensional problem instance created.")
        else:
            self.K = get_cutoff_index(self.p_full, threshold) 
            self.p = self.p_full[:self.K]                      
            print(f"Dimensionality reduced from {self.K_full} to {self.K}")

        
    def g(self, gamma: np.ndarray) -> float:
        """Computes g(gamma) on an effective-sized gamma vector."""
        gamma = np.clip(gamma, 1e-10, 1 - 1e-10)
        log_prod_1_minus_gamma = np.sum(np.log(1 - gamma))
        sum_gamma_over_1_minus_gamma = np.sum(gamma / (1 - gamma))
        return np.exp(log_prod_1_minus_gamma) * (1 + sum_gamma_over_1_minus_gamma)

    def objective(self, gamma_effective: np.ndarray) -> float:
        """Computes log(g) efficiently on the effective-sized gamma."""
        val_g = self.g(gamma_effective)
        return np.log(val_g + 1e-100)

    def gradient(self, gamma_effective: np.ndarray) -> np.ndarray:
        """Computes gradient efficiently on the effective-sized gamma."""
        gamma = np.clip(gamma_effective, 1e-10, 1 - 1e-10)
        one_minus_gamma = 1 - gamma
        gamma_over_1_minus_gamma = gamma / one_minus_gamma
        S = np.sum(gamma_over_1_minus_gamma)
        S_minus_term = S - gamma_over_1_minus_gamma
        denominator = 1 + S
        grad = - (1 / one_minus_gamma) * (S_minus_term / (denominator + 1e-100))
        return grad

    def project(self, y_effective: np.ndarray) -> np.ndarray:
        """Projects a point onto the feasible set of the effective problem."""
        p_effective = self.p
        def f(lambda_val: float) -> float:
            x_proj = np.clip(y_effective - lambda_val * p_effective, 0, 1)
            return np.sum(p_effective * x_proj) - self.target_sum

        safe_mask = p_effective > 1e-12
        if not np.any(safe_mask): return y_effective
        safe_p, safe_y = p_effective[safe_mask], y_effective[safe_mask]
        
        lambda_min, lambda_max = np.min((safe_y - 1) / safe_p) - 1, np.max(safe_y / safe_p) + 1
        try:
            optimal_lambda = brentq(f, lambda_min, lambda_max, xtol=1e-12, rtol=1e-12)
            return np.clip(y_effective - optimal_lambda * p_effective, 0, 1)
        except ValueError:
            f_min, f_max = f(lambda_min), f(lambda_max)
            return np.clip(y_effective - (lambda_min if abs(f_min) < abs(f_max) else lambda_max) * p_effective, 0, 1)

    def get_initial_point(self, strategy: str = 'random') -> np.ndarray:
        """Generates an initial point of the effective dimension."""
        if strategy == 'sorted_interior':
            y = 1e-6 + (1 - 2*1e-6) * np.random.rand(self.K)
            return self.project(np.sort(y)[::-1])
        return self.project(np.random.rand(self.K))
  

# --- Solver Classes ---
SolveResult = namedtuple('SolveResult', 
                         ['solution', 'value', 'success', 'message', 'history'])

class PGASolver:
    """
    A Projected Gradient Ascent (PGA) solver with a robust and stable
    backtracking line search that uses the Armijo condition.
    """
    def __init__(self, problem: OptimizationProblem, tol: float = 1e-9, max_iter: int = 5000):
        self.problem = problem
        self.tol = tol
        self.max_iter = max_iter

    def _line_search(self, gamma, grad, initial_eta=1.0, alpha=0.5, beta=1e-4):
        """
        Performs a backtracking line search to find an appropriate step size (eta).
        """
        eta = initial_eta
        current_obj = self.problem.objective(gamma)
        slope_metric = np.dot(grad, grad)

        for _ in range(30): 
            gamma_new = self.problem.project(gamma + eta * grad)
            new_obj = self.problem.objective(gamma_new)
            
            if new_obj >= current_obj + beta * eta * slope_metric:
                return eta
            
            eta *= alpha
            
        return eta

    def solve(self, gamma0: np.ndarray = None, verbose: bool = False) -> 'SolveResult':
        gamma = self.problem.project(gamma0) if gamma0 is not None else self.problem.get_initial_point('sorted_interior')
        result_reduced = None
        for _ in range(self.max_iter):
            grad, eta = self.problem.gradient(gamma), self._line_search(gamma, self.problem.gradient(gamma))
            if eta < 1e-12:
                result_reduced = SolveResult(gamma, self.problem.objective(gamma), True, "Convergence (negligible step size).", [])
                break
            gamma_new = self.problem.project(gamma + eta * grad)
            if np.linalg.norm(gamma_new - gamma) < self.tol:
                result_reduced = SolveResult(gamma_new, self.problem.objective(gamma_new), True, "Convergence tolerance reached.", [])
                break
            gamma = gamma_new
        if result_reduced is None:
            result_reduced = SolveResult(gamma, self.problem.objective(gamma), False, "Maximum iterations reached.", [])
        
        solution_full = np.zeros(self.problem.K_full)
        solution_full[:self.problem.K] = result_reduced.solution
        return result_reduced._replace(solution=solution_full)
   

class SLSQPSolver:
    """
    Wrapper for the Scipy SLSQP solver to provide a consistent interface.
    """
    def __init__(self, problem: OptimizationProblem, max_iter: int = None, tol: float = 1e-9):
        self.problem = problem
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, gamma0: np.ndarray = None, verbose: bool = False) -> SolveResult:
        gamma0 = gamma0 if gamma0 is not None else self.problem.get_initial_point()
        result = minimize(
            fun=lambda g: -self.problem.objective(g), x0=gamma0, method='SLSQP',
            bounds=[(0, 1) for _ in range(self.problem.K)],
            constraints={'type': 'eq', 'fun': lambda g: np.sum(self.problem.p * g) - self.problem.target_sum},
            options={'maxiter': self.max_iter, 'ftol': self.tol, 'disp': verbose}
        )

        solution_full = np.zeros(self.problem.K_full)
        if result.success:
            solution_full[:self.problem.K] = result.x
        return SolveResult(solution_full, -result.fun if result.success else -np.inf, result.success, result.message, [])



# --- Multi-start Optimization ---
def find_best_solution_with_multistart(problem: OptimizationProblem, solver, num_starts: int) -> SolveResult:
    """
    Performs multi-start optimization.
    """
    best_result_overall = None
    best_value_so_far = -np.inf 

    strategies = ['sorted_interior'] * (num_starts // 2 + 1) 
    strategies += ['random'] * (num_starts - len(strategies))
    np.random.shuffle(strategies)
    print(f"\n--- Starting Multi-start Optimization with {num_starts} Runs for {solver.__class__.__name__} ---")
    
    progress_bar = tqdm(range(num_starts), desc="Multi-start Progress")
    for i in progress_bar:
        strategy = strategies[i]
        initial_gamma = problem.get_initial_point(strategy=strategy)

        current_result = solver.solve(initial_gamma, verbose=False)

        if current_result.value > best_value_so_far:
            best_value_so_far = current_result.value
            best_result_overall = current_result
            tqdm.write(f"Run {i+1}/{num_starts}: New best solution found! Objective improved to {current_result.value:.6f}")
            
    if best_result_overall is None:
        return SolveResult(None, -np.inf, False, "All optimization runs failed.", [])
        
    return best_result_overall


# --- Utility Functions ---
def get_cutoff_index(p: np.ndarray, threshold: float = 1e-5) -> int:
    """
    Finds the index k such that the tail sum (from k onwards) is negligible.
    This corrected version finds the point where sum(p[k:]) <= threshold.
    """
    tail_sum = np.cumsum(p[::-1])[::-1]
    effective_k_candidates = np.where(tail_sum <= threshold)[0]
    
    if len(effective_k_candidates) > 0:
        k_eff = effective_k_candidates[0]
        return max(2, k_eff)

    return len(p)


def print_results(title, result, time_taken):
    """
    Utility function to print formatted results.
    """
    print("\n" + "="*50)
    print(f"--- {title} ---")
    if result.solution is not None:
        log_g_val = result.value
        g_val = np.exp(log_g_val)
        non_zero_gamma = result.solution[result.solution > 1e-6]
        print(f"Final Objective Value (log_g): {log_g_val:.6f}")
        print(f"Final Objective Value (g):     {g_val:.6f}")
        print(f"Solution gamma (non-zero part, {len(non_zero_gamma)} elements): {np.round(non_zero_gamma, 4)}")
    else:
        print("No solution found.")
    print(f"Total time taken: {time_taken:.4f} seconds")
    print("="*50)

def generate_p_dirichlet(K: int, scale: float = 0.1, seed: int = None) -> np.ndarray:
    """
    Generate a Dirichlet-distributed random vector and return it sorted in descending order.
    """
    if seed is not None:
        np.random.seed(seed)
    p = np.random.dirichlet([scale]*K)

    return np.sort(p)[::-1]

def conjecture_gamma(p, alpha):
    """
    Conjectured gamma based on the given p and alpha.
    """
    K = len(p)
    prefix_sum = np.cumsum(p)
    target = 1 - alpha
    for m in range(2, K+1):
        numerator = prefix_sum[m-1] - target
        denom = m - 1
        val = numerator / denom if denom > 0 else np.inf
        p_m = p[m-1]
        p_next = p[m] if m < K else 0
        if p_m > val >= p_next:
            gamma = np.zeros(K)
            gamma[0] = 1
            for i in range(1, m):
                gamma[i] = 1 - numerator / ((m-1) * p[i])
            return gamma
    return np.zeros(K)


# --- Main Execution ---
if __name__ == '__main__':
    K = 1000
    p = generate_p_dirichlet(K, scale=0.005)
    alpha = 0.05
    threshold = 0.001
    print(f"p = {np.round(p, 4)}")

    problem = OptimizationProblem(p, alpha, threshold=threshold)
    num_starts = 5

    # --- Solve with Projected Gradient Ascent (Multi-start) ---
    #pga_solver = PGASolver(problem, tol=1e-9, max_iter=1000)
    #start_time_pga = time.time()
    #best_result_pga = find_best_solution_with_multistart(problem, pga_solver, num_starts=num_starts)
    #end_time_pga = time.time()

    # --- Solve with SLSQP (Multi-start) ---
    slsqp_solver = SLSQPSolver(problem, tol=1e-9, max_iter=1000)
    start_time_slsqp = time.time()
    best_result_slsqp = find_best_solution_with_multistart(problem, slsqp_solver, num_starts=num_starts)
    end_time_slsqp = time.time()

    #print_results("Overall Best Result from PGA Multi-start", best_result_pga, end_time_pga - start_time_pga)
    print_results("Overall Best Result from SLSQP Multi-start", best_result_slsqp, end_time_slsqp - start_time_slsqp)


    # --- Conjectured gamma for comparison ---
    print("\n--- Conjectured gamma for comparison ---")
    gamma_conjecture = conjecture_gamma(p, alpha)
    non_zero_gamma = gamma_conjecture[gamma_conjecture > 1e-6]
    problem_full = OptimizationProblem(p, alpha, threshold=None)
    print(f"Conjectured gamma (non-zero part, {len(non_zero_gamma)} elements): {np.round(non_zero_gamma, 4)}")
    g_conjecture = problem_full.g(gamma_conjecture)
    print(f"Conjectured g(gamma) = {g_conjecture:.6f}")
    print(f"Conjectured log_g(gamma) = {np.log(g_conjecture):.6f}")