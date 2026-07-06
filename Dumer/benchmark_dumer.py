import time
import numpy as np
import pandas as pd
import galois
import scipy.special
import matplotlib.pyplot as plt
from itertools import combinations

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

def dumer_solver(n, r, H, v, t, GF):
    """
    Executes Dumer's algorithm for a fixed target weight t 
    and returns the execution time.
    """
    p = n // 2
    start_time = time.perf_counter()
    
    for t1 in range(max(0, t - (n - p)), min(t, p) + 1):
        t2 = t - t1
        Z = []
        
        # Z': First half
        for pos1 in combinations(range(p), t1):
            e_prime = GF.Zeros(n)
            e_prime[list(pos1)] = 1
            v0 = H @ e_prime
            Z.append((tuple(v0.tolist()), 0, tuple(e_prime.tolist())))
            
        # Z'': Second half
        for pos2 in combinations(range(p, n), t2):
            e_double = GF.Zeros(n)
            e_double[list(pos2)] = 1
            v1 = H @ e_double
            v1_plus_v = v1 + v  
            Z.append((tuple(v1_plus_v.tolist()), 1, tuple(e_double.tolist())))

        Z.sort()

        # Search for collisions
        for i in range(len(Z) - 1):
            if (Z[i][0] == Z[i+1][0]) and (Z[i][1] == 0 and Z[i+1][1] == 1):
                e_left = GF(Z[i][2])
                e_right = GF(Z[i+1][2])
                candidate = e_left + e_right
                
                if np.array_equal(H @ candidate, v):
                    return time.perf_counter() - start_time 
       
    return time.perf_counter() - start_time

def run_benchmark():
    GF = galois.GF(2)
    # Target range for the benchmark. Adjust max limit based on CPU availability.
    n_vals = list(range(14, 46, 2)) 
    trials_per_n = 5 
    
    results = []

    print(f"{'='*60}")
    print("STARTING DUMER BENCHMARK (CRYPTANALYSIS REGIME)")
    print(f"Targeting: R=0.5, t = floor((t_GV - 1) / 2)")
    print(f"{'='*60}\n")

    for n in n_vals:
        r = n // 2
        k = n - r
        total_time = 0.0
        
        D_GV = compute_gv_bound(n, r)
        t = (D_GV - 1) // 2
        
        print(f"Benchmarking n={n} (r={r}, t={t})... ", end="", flush=True)
        
        for _ in range(trials_per_n):
            H, c, e, v = generate_test_instance(n, r, t, GF)
            exec_time = dumer_solver(n, r, H, v, t, GF)
            total_time += exec_time
            
        avg_time = total_time / trials_per_n
        
        results.append({
            'n': n,
            'k': k,
            't': t,
            'Avg_Time_sec': avg_time
        })
        print(f"Done. (Avg Time: {avg_time:.4f}s)")

    # Export results for LaTeX integration
    df = pd.DataFrame(results)
    df.to_csv("dumer_benchmark_results.csv", index=False)

    # Generate visual plot
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    
    # Plotting the main benchmark line
    plt.plot(df['n'], df['Avg_Time_sec'], marker='o', linestyle='-', color='purple', label="Dumer's MitM")
    
    # Annotate the "Phase Jumps" (Combinatorial Steps) dynamically
    for i in range(1, len(df)):
        if df['t'].iloc[i] > df['t'].iloc[i-1]:
            plt.axvline(x=df['n'].iloc[i], color='gray', linestyle=':', alpha=0.7)
            plt.text(df['n'].iloc[i] + 0.5, df['Avg_Time_sec'].iloc[i], f"$t$ = {df['t'].iloc[i]}", 
                     color='gray', fontsize=9, verticalalignment='bottom')

    plt.title("Dumer's Algorithm Benchmark: Combinatorial Step\nFixed Weight $t = \\lfloor(t_{GV}-1)/2\\rfloor$")
    plt.xlabel("Code Length ($n$)")
    plt.ylabel("Execution Time (Seconds) [Log Scale]")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.savefig("plot_dumer_benchmark.pdf", format='pdf', bbox_inches='tight')
    print("\nFiles successfully exported: 'dumer_benchmark_results.csv' and 'plot_dumer_benchmark.pdf'")

if __name__ == "__main__":
    run_benchmark()