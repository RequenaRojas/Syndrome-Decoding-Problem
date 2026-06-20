import galois
import numpy as np
from tabulate import tabulate
import itertools

def solve_sdp_cosets(H, s, GF, verbose=False):
    """
    Solves the SDP by generating the affine space (coset) of solutions z + C.
    Assumes H is in systematic form H = [I_{n-k} | A].
    """
    r, n = H.shape
    k = n - r

    G = H.null_space()
    
    if verbose:
        print(f"G:\n{tabulate(G, tablefmt='plain')}")

    # Construct the particular solution z = (s, 0)
    z = GF.Zeros(n)
    z[:r] = s.flatten()

    # Generate all messages m in GF^k
    k_list = list(itertools.product(range(GF.order), repeat=k))
    messages = GF(k_list)

    # Compute all codewords: c = mG
    codewords = messages @ G

    # Construct the coset: z + C
    coset = codewords + z

    # Calculate weights (non-zero entries per row)
    weights = np.count_nonzero(coset.view(np.ndarray), axis=1)
    
    if verbose:
        print(f"Coset Space:")
        print(tabulate(coset, tablefmt="plain"))
        print(f"Weights: {weights}")
    
    min_weight_id = np.argmin(weights)
    e_min_weight = coset[min_weight_id]

    if verbose:
        print(f"e (min weight = {weights[min_weight_id]}):\n{e_min_weight}")
    
    return e_min_weight, G

def run_single_test_coset(n, k, GF):
    r = n - k
    # Force systematic H: H = [I_r | A]
    I_r = GF.Identity(r)
    A = GF.Random((r, k))
    H = np.hstack((I_r, A))
    
    y = GF.Random(n)
    s = H @ y
    
    while np.all(s == 0):
        y = GF.Random(n)
        s = H @ y
   
    e_min_weight, G = solve_sdp_cosets(H, s, GF)

    # Verify decoding: b = y - e
    b_computed = y - e_min_weight
    verification = H @ b_computed
    success = np.all(verification == 0)
    
    return success, G, H, y, s, e_min_weight, verification

# --- Testing Loop ---
success_count = 0
error_count = 0
error_details = []
num_tests = 5

n_vals = [18 + 2*i for i in range(num_tests)]
k_vals = [n_vals[i]//2 for i in range(num_tests)]

print(f"\n{'='*50}")
print(f"INITIALIZING TESTS - COSET SEARCH")
print(f"{'='*50}")
print(f"Testing range for n: {n_vals}")
print(f"Testing range for k: {k_vals}")
print(f"Total iterations planned: {num_tests}\n")

GF = galois.GF(2)

for i in range(num_tests):
    n = n_vals[i]
    k = k_vals[i]
    try:
        success, G, H, y, s, e, verification = run_single_test_coset(n, k, GF)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            error_details.append({
                'iteration': i,
                'G': G.copy(),
                'H': H.copy(),
                'y': y.copy(),
                's': s.copy(),
                'e': e.copy(),
                'verification': verification.copy(),
            })
            print(f"Error at iteration {i} with n={n}, k={k}")
            
    except Exception as exc:
        error_count += 1
        print(f"Exception at iteration {i}: {str(exc)}")

print(f"\n{'='*50}")
print(f"FINAL RESULTS")
print(f"{'='*50}")
print(f"Total iterations: {num_tests}")
print(f"Successes: {success_count}")
print(f"Errors: {error_count}")
print(f"Success rate: {success_count*(100/num_tests)}%")

if error_details:
    print(f"\nDetails of the first 3 errors:")
    for i, err in enumerate(error_details[:3]):
        print(f"\nError {i + 1} (Iteration {err['iteration']}):")
        print("Matrix G:\n", tabulate(err['G'], tablefmt="plain"))
        print("Matrix H:\n", tabulate(err['H'], tablefmt="plain"))
        print("y:\n", err['y'])
        print("s:\n", err['s'])
        print("e:\n", err['e'])
        print("Verification (Hb):\n", err['verification'])