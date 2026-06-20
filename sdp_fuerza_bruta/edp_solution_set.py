import galois
import numpy as np
import scipy.special
from tabulate import tabulate
import itertools

def compute_gv_bound(n: int, r: int) -> int:
    space_syndromes = 2**r
    sphere_volumen = 1
    for t in range(1, n + 1):
        sphere_volumen += scipy.special.comb(n, t, exact=True)
        if sphere_volumen >= space_syndromes:
            return t
    return n 

def solve_esd_solution_set(H, s, t, GF, verbose=False):
    """
    Solves the Exact Syndrome Decoding (ESD) problem via exhaustive search.
    Assumes H is in systematic form H = [I_{n-k} | A].
    """
    n_minus_k, n = H.shape
    k = n - n_minus_k
    q = GF.order

    T = GF.Zeros((n_minus_k, k+1))
    for i in range(n_minus_k):
        for j in range(k):
            T[i][j] = -H[i][n_minus_k + j]
        T[i][k] = s[i][0]

    prod_q_k = list(itertools.product(list(GF.elements), repeat=k))

    for r in range(q**k):
        solution = GF.Zeros(n)
        for i in range(k):
            solution[n_minus_k + i] = prod_q_k[r][i]
        for i in range(n_minus_k):
            for j in range(k):
                solution[i] += prod_q_k[r][j] * T[i][j]
            solution[i] += T[i][k]
        
        if np.count_nonzero(solution) == t:
            if verbose:
                print(f"Target weight e found (t={t}):\n{solution}")     
            return solution.reshape(n, 1)

    return None

def run_single_test_esd(n, k, t, GF):
    r = n - k
    # Force systematic H: H = [I_r | A]
    I_r = GF.Identity(r)
    A = GF.Random((r, k))
    H = np.hstack((I_r, A))
            
    G = H.null_space()
    
    m = GF.Random(k)
    b = m @ G
    
    # Generate exact weight error t
    e = GF.Zeros(n)
    positions = np.random.choice(n, t, replace=False)
    e[positions] = 1
    
    y = b + e
    s = (H @ y).reshape(r, 1)

    e_weight_t = solve_esd_solution_set(H, s, t, GF)

    if e_weight_t is None:
        return False, H, y, e, False

    # Mathematical validations
    s_computed = H @ e_weight_t.flatten()
    success = np.array_equal(s_computed, s.flatten())
    verification = np.array_equal(y, b + e_weight_t.flatten())
    
    return success, H, y, e_weight_t, verification

# --- Testing Loop ---
success_count = 0
error_count = 0
error_details = []
num_tests = 5

n_vals = [18 + 2*i for i in range(num_tests)]
k_vals = [n_vals[i]//2 for i in range(num_tests)]

print(f"\n{'='*50}")
print(f"INITIALIZING TESTS")
print(f"{'='*50}")
print(f"Testing range for n: {n_vals}")
print(f"Testing range for k: {k_vals}")
print(f"Total iterations planned: {num_tests}\n")

GF = galois.GF(2)

for i in range(num_tests):
    n = n_vals[i]
    k = k_vals[i]
    try:
        D = compute_gv_bound(n,n-k) 
        for t_test in range(D):
            success, H, y, e, is_verified = run_single_test_esd(n, k, t_test+1, GF)
            if success: break

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
print(f"FINAL RESULT - EXACT DECODING PROBLEM (EXD)")
print(f"{'='*50}")
print(f"Total iterations: {num_tests}")
print(f"Successes: {success_count}")
print(f"Errors: {error_count}")
print(f"Success rate: {success_count*(100/num_tests)}%")