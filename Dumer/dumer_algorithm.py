import numpy as np
import galois
from itertools import combinations
from tabulate import tabulate 
import scipy.special

def random_parity_check_matrix(n, r, GF):
    H = GF.Random((r, n))
    while np.linalg.matrix_rank(H) != r:
        H = GF.Random((r, n))
    return H

def compute_gv_bound(n: int, r: int) -> int:
    syndrome_space = 2**r
    sphere_volume = 0
    for t in range(1, n + 1):
        sphere_volume += scipy.special.comb(n, t, exact=True)
        if sphere_volume >= syndrome_space:
            return t
    return n

def generate_test_instance(n, r, t, GF):
    """
    Generates a full-rank parity-check matrix H, a random codeword c, 
    an error vector e of exact weight t, and the resulting syndrome v.
    """
    while True:
        H = GF.Random((r, n))
        if np.linalg.matrix_rank(H) == r:
            break
            
    G = H.null_space()
    k = G.shape[0] 
    
    m = GF.Random(k)
    c = m @ G
    
    e = GF.Zeros(n)
    positions = np.random.choice(n, t, replace=False)
    e[positions] = 1
    
    y = c + e
    v = H @ y
    return H, c, e, v

def run_dumer_algorithm(H, v, n, p, t_range, GF):
    """
    Executes Dumer's Meet-in-the-Middle algorithm to find the error vector.
    """
    global_solution = None

    for t in t_range:
        print(f"Exploring weight t = {t}...")
        
        # Iterate over weight partitions: t1 + t2 = t
        for t1 in range(max(0, t - (n - p)), min(t, p) + 1):
            t2 = t - t1
            
            Z = [] 
            
            # Z': First half (weight t1)
            for pos1 in combinations(range(p), t1):
                e_prime = GF.Zeros(n)
                e_prime[list(pos1)] = 1
                v0 = H @ e_prime
                Z.append((tuple(v0.tolist()), 0, tuple(e_prime.tolist())))
                
            # Z'': Second half (weight t2)
            for pos2 in combinations(range(p, n), t2):
                e_double = GF.Zeros(n)
                e_double[list(pos2)] = 1
                v1 = H @ e_double
                v1_plus_v = v1 + v  
                Z.append((tuple(v1_plus_v.tolist()), 1, tuple(e_double.tolist())))

            # Sort to find collisions efficiently
            Z.sort()

            # Search for collisions
            for i in range(len(Z) - 1):
                current_synd = Z[i][0]
                next_synd = Z[i+1][0]
                current_flag = Z[i][1]
                next_flag = Z[i+1][1]
                
                if (current_synd == next_synd) and (current_flag == 0 and next_flag == 1):
                    e_left = GF(Z[i][2])
                    e_right = GF(Z[i+1][2])
                    candidate = e_left + e_right
                    
                    if np.array_equal(H @ candidate, v):
                        global_solution = candidate
                        break 
                        
            if global_solution is not None:
                break 
                
        if global_solution is not None:
            return global_solution, t
            
    return None, None


def main():
    GF = galois.GF(2)

    # ------------------------------------ Parameters ---------------------------------
    n, k = 21, 10
    r = n - k
    p = n // 2
    
    D_GV = compute_gv_bound(n, r)
    
    # Mode selection: 0 for fixed t (cryptanalysis), 1 for range (noisy channel)
    search_mode = 0 
    
    if search_mode:
        t_real = np.random.choice(range(1, D_GV + 1))
        t_range = range(D_GV + 1)    
    else:
        t_real = (D_GV - 1) // 2
        t_range = range(t_real, t_real + 1)

    H, c, e, v = generate_test_instance(n, r, t_real, GF)

    print("-" * 60)
    print(f"Code Parameters [{n},{k}]:")
    print(tabulate(H, tablefmt="plain"))
    print(f"Gilbert-Varshamov Bound: t_GV = {D_GV}")
    print(f"Search Limit: t in {list(t_range)}")
    print(f"Injected Error (e): {e}")
    print(f"Target Syndrome (v): {v}")
    print("-" * 60)

    # ------------------------------------ Execution ------------------------------------
    solution, found_at_t = run_dumer_algorithm(H, v, n, p, t_range, GF)

    if solution is not None:
        print(f"\nSolution found at t = {found_at_t}")
        print(f"Error vector recovered: {solution}")
        if np.array_equal(solution, e): 
            print("Verification: SUCCESS (Matches injected error)")
        else: 
            print("Verification: FAILED (Collision found, but does not match original error)")
    else:
        print(f"\nFailure: No solutions found in the interval {list(t_range)}.")
        print("Reason: The original error likely had a weight beyond the expected GV bound.")

if __name__ == "__main__":
    main()