import time
import galois
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the fully vectorized strict combinatorial solver
from edp_brute_force import solve_esd_vectorized

def generate_esd_instance(n, k, t, GF):
    """
    Generates a valid instance for Exact Syndrome Decoding (ESD).
    H can be a completely random full-rank matrix (does not need to be systematic).
    """
    r = n - k
    while True:
        H = GF.Random((r, n))
        if np.linalg.matrix_rank(H) == r:
            break
            
    # Inject an error vector 'e' of strict weight 't'
    e = GF.Zeros(n)
    positions = np.random.choice(n, t, replace=False)
    e[positions] = 1
    
    # Compute the actual target syndrome
    s = (H @ e).reshape(r, 1)
    
    return H, s

def run_esd_vectorized_benchmarks():
    GF = galois.GF(2)
    trials = 10
    
    print(f"{'='*60}")
    print(f"STARTING ESD BENCHMARK (FULLY VECTORIZED APPROACH)")
    print(f"WARNING: Keep an eye on your RAM usage for large n and t.")
    print(f"{'='*60}\n")
    
    # --- PLOT 1: Time vs Code Length n (Fixed relative weight delta) ---
    n_vals = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    delta = 0.2  # t will be approximately 20% of n
    results_plot1 = []
    
    print("Running Plot 1 (Time vs n) data collection...")
    for n in n_vals:
        k = n // 2
        t = max(1, int(n * delta)) 
        
        time_total = 0.0
        for _ in range(trials):
            H, s = generate_esd_instance(n, k, t, GF)
            start = time.perf_counter()
            solve_esd_vectorized(H, s, t, GF, verbose=False)
            time_total += (time.perf_counter() - start)
        
        avg_time = time_total / trials
        results_plot1.append({'n': n, 'k': k, 't': t, 'Time_sec': avg_time})
        print(f"  n={n:02d}, k={k:02d}, t={t:02d} -> {avg_time:.6f} seconds")
        
    df2 = pd.DataFrame(results_plot1)
    df2.to_csv("esd_vectorized_plot1_data.csv", index=False)
    
    # --- PLOT 2: Time vs Target Weight t (Fixed n and k) ---
    n_fixed = 30
    k_fixed = 15
    # Iterate from t=1 up to n/2 (maximum combinatorial explosion)
    t_vals = list(range(1, 8)) 
    results_plot2 = []
    
    print("\nRunning Plot 2 (Time vs t) data collection...")
    for t in t_vals:
        time_total = 0.0
        for _ in range(trials):
            H, s = generate_esd_instance(n_fixed, k_fixed, t, GF)
            start = time.perf_counter()
            solve_esd_vectorized(H, s, t, GF, verbose=False)
            time_total += (time.perf_counter() - start)
            
        avg_time = time_total / trials
        results_plot2.append({'t': t, 'Time_sec': avg_time})
        print(f"  n={n_fixed:02d}, t={t:02d} -> {avg_time:.6f} seconds")
        
    df3 = pd.DataFrame(results_plot2)
    df3.to_csv("esd_vectorized_plot2_data.csv", index=False)
    
    # --- Generation of Visuals ---
    
    # Figure 2 Setup (Logarithmic Scale)
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.plot(df2['n'], df2['Time_sec'], marker='o', color='g', label=fr'ESD Vectorized (Fixed $\delta={delta}$)')
    plt.title('ESD Benchmark: Execution Time vs. Code Length ($n$)')
    plt.xlabel('Code Length ($n$)')
    plt.ylabel('Execution Time (Seconds) [Log Scale]')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("plot_esd_vectorized_time_vs_n.pdf", format='pdf', bbox_inches='tight')
    
    # Figure 3 Setup (Linear Scale to show the Binomial bell curve)
    plt.figure(figsize=(8, 6))
    plt.plot(df3['t'], df3['Time_sec'], marker='s', color='purple', label=fr'ESD Vectorized (Fixed $n={n_fixed}$)')
    plt.title('ESD Benchmark: Execution Time vs. Target Weight ($t$)')
    plt.xlabel('Target Weight ($t$)')
    plt.ylabel('Execution Time (Seconds) [Linear Scale]')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("plot_esd_vectorized_time_vs_t.pdf", format='pdf', bbox_inches='tight')
    
    print("\nBenchmarks complete. CSVs and PDFs exported successfully.")

if __name__ == "__main__":
    run_esd_vectorized_benchmarks()