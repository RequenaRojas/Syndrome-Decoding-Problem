import galois
import numpy as np
from tabulate import tabulate
import itertools

def solve_sdp_solution_set(H, s, GF, verbose=False):
    """
    Solves the Syndrome Decoding Problem (SDP) by searching through the minimum weight solutions of He=s.
    Assumes H is in systematic form H = [I_{n-k} | A].
    """
    r, n = H.shape
    k = n - r
    q = GF.order

    T = GF.Zeros((r, k+1))
    for i in range(r):
        for j in range(k):
            T[i][j] = -H[i][r + j]
        T[i][k] = s[i][0]
        
    if verbose:
        print(f"T:\n{tabulate(T, tablefmt='plain')}")

    solution_space = GF.Zeros((q**k, n))
    weights = np.zeros(q**k, dtype=int)
    prod_q_k = list(itertools.product(list(GF.elements), repeat=k))

    for id in range(q**k):
        solution = GF.Zeros(n)
        for i in range(k):
            solution[r + i] = prod_q_k[id][i]
        for i in range(r):
            for j in range(k):
                solution[i] += prod_q_k[id][j] * T[i][j]
            solution[i] += T[i][k]
        solution_space[id] = solution
        weights[id] = np.count_nonzero(solution)
    
    min_weight_id = np.argmin(weights)
    min_weight_e = solution_space[min_weight_id].reshape(n, 1)
    
    if verbose:
        print(f"e (min weight = {weights[min_weight_id]}):\n{min_weight_e}")
    return min_weight_e

def run_single_test_sdp(n, k, GF):
    r = n - k
    # Force systematic H: H = [I_r | A]
    I_r = GF.Identity(r)
    A = GF.Random((r, k))
    H = np.hstack((I_r, A))
            
    y = GF.Random(n)
    s = (H @ y).reshape(r, 1)

    e_min = solve_sdp_solution_set(H, s, GF)

    if e_min is None:
        return False, H, y, None, False

    # Mathematical validations
    s_computed = H @ e_min.flatten()
    success = np.array_equal(s_computed, s.flatten())
    
    b_computed = y - e_min.flatten()
    verification = np.array_equal(H @ b_computed, GF.Zeros(r))
    
    return success, H, y, e_min, verification

# --- Testing Loop for SDP ---
success_count = 0
error_count = 0
error_details = []
num_tests = 5

n_vals = [15 + 2*i for i in range(num_tests)]
k_vals = [n_vals[i]//2 for i in range(num_tests)]

GF = galois.GF(2)

print(f"\n{'='*50}")
print(f"INITIALIZING TESTS")
print(f"{'='*50}")
print(f"Testing range for n: {n_vals}")
print(f"Testing range for k: {k_vals}")
print(f"Total iterations planned: {num_tests}\n")


for i in range(num_tests):
    n = n_vals[i]
    k = k_vals[i]
    try:
        success, H, y, e, is_verified = run_single_test_sdp(n, k, GF)

        if success and is_verified:
            success_count += 1
        else:    
            error_count += 1
            error_details.append({
                'iteration': i,
                'H': H.copy(),
                'y': y.copy(),
                'e': e.copy() if e is not None else "Not found",
            })
            print(f"Error or non-convergence at iteration {i} with n={n}, k={k}")
            
    except Exception as exc:
        error_count += 1
        print(f"Exception at iteration {i}: {str(exc)}")

print(f"\n{'='*50}")
print(f"FINAL RESULTS - SYNDROME DECODING PROBLEM (SDP)")
print(f"{'='*50}")
print(f"Total iterations: {num_tests}")
print(f"Successes: {success_count}")
print(f"Errors: {error_count}")
print(f"Success rate: {success_count*(100/num_tests)}%")