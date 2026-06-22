import galois
import numpy as np
import scipy.special
import itertools

def compute_gv_bound(n: int, r: int) -> int:
    space_syndromes = 2**r
    sphere_volumen = 1
    for t in range(1, n + 1):
        sphere_volumen += scipy.special.comb(n, t, exact=True)
        if sphere_volumen >= space_syndromes:
            return t
    return n 


def solve_esd_combinations(H, s, t, GF, verbose=False):
    """
    Solves the Exact Syndrome Decoding (ESD) problem by strictly iterating over the 
    Hamming sphere of radius t.
    """
    r, n = H.shape
    non_zero_elements = list(GF.elements)[1:] 
    
    for positions in itertools.combinations(range(n), t):
        
        for values in itertools.product(non_zero_elements, repeat=t):
            
            e = GF.Zeros(n)
            for id, pos in enumerate(positions):
                e[pos] = values[id]
            
            s_computed = (H @ e).reshape(r, 1)
            
            if np.array_equal(s_computed, s):
                if verbose:
                    print(f"Target weight e found (t={t}):\n{e}")
                return e.reshape(n, 1)
                
    return None

def solve_esd_vectorized(H, s, t, GF, verbose=False):
    """
    Solves the Exact Syndrome Decoding (ESD) problem using fully vectorized 
    matrix operations, eliminating nested Python loops.
    
    WARNING: The space complexity is O( (n choose t) * (q-1)^t * n ). 
    This will trigger an Out-Of-Memory (OOM) error for large values of n and t.
    """
    r, n = H.shape
    non_zero_elements = list(GF.elements)[1:] 
    
    # 1. Generate all positions and values (Array creation)
    pos_array = np.array(list(itertools.combinations(range(n), t)))
    val_array = np.array(list(itertools.product(non_zero_elements, repeat=t)))
    
    N_c = pos_array.shape[0]
    N_v = val_array.shape[0]
    N_total = N_c * N_v
    
    # 2. Expand to create the Cartesian product conceptually
    pos_rep = np.repeat(pos_array, N_v, axis=0)
    val_tile = np.tile(val_array, (N_c, 1))
    
    # 3. Construct the massive error matrix E of shape (N_total, n)
    E = GF.Zeros((N_total, n))
    
    # Advanced NumPy indexing to assign values without loops
    row_idx = np.repeat(np.arange(N_total), t)
    col_idx = pos_rep.flatten()
    E[row_idx, col_idx] = val_tile.flatten()
    
    # 4. Massive Matrix Multiplication: S_computed = E @ H^T
    # This evaluates the syndrome for all N_total vectors simultaneously in C
    S_computed = E @ H.T
    
    # 5. Find the collision with the target syndrome 's'
    # s is expected to be shape (r, 1), so s.T is (1, r) for broadcasting
    matches = np.all(S_computed == s.T, axis=1)
    
    if np.any(matches):
        match_idx = np.argmax(matches)
        e_found = E[match_idx].reshape(n, 1)
        if verbose:
            print(f"Target weight e found (t={t}):\n{e_found}")
        return e_found
        
    return None

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

def run_single_test_esd_solution_set(n, k, t, GF):
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

def run_edp_testing_loop():
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
                success, H, y, e, is_verified = run_single_test_esd_solution_set(n, k, t_test+1, GF)
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

    return 0


if __name__ == "__main__":
    run_edp_testing_loop()