import time
import numpy as np
import pandas as pd
import galois
import scipy.special
import matplotlib.pyplot as plt
from itertools import combinations
from benchmark_dumer import generate_test_instance, compute_gv_bound, dumer_solver
from edp_brute_force import solve_esd_vectorized


def run_benchmark():
    GF = galois.GF(2)
    # Target range: Bounded to 26 because Brute Force will collapse the CPU beyond this point.
    n_vals = list(range(10, 28, 2)) 
    trials_per_n = 3 
    
    results = []

    print(f"{'='*60}")
    print("STARTING COMPARATIVE BENCHMARK: BRUTE FORCE VS DUMER")
    print(f"Targeting: R=0.5, t = floor((t_GV - 1) / 2)")
    print(f"{'='*60}\n")

    for n in n_vals:
        r = n // 2
        k = n - r
        time_bf_total = 0.0
        time_dumer_total = 0.0
        
        D_GV = compute_gv_bound(n, r)
        t = (D_GV - 1) // 2
        
        print(f"Benchmarking n={n} (r={r}, t={t})... ", end="", flush=True)
        
        for _ in range(trials_per_n):
            H, c, e, s = generate_test_instance(n, r, t, GF)
            
            # Benchmark Dumer
            time_dumer_total += dumer_solver(n, r, H, s, t, GF)
            
            # Benchmark Brute Force
            time_bf_total += solve_esd_vectorized(H, s, t, GF)
            
        avg_bf = time_bf_total / trials_per_n
        avg_dumer = time_dumer_total / trials_per_n
        
        results.append({
            'n': n,
            'k': k,
            't': t,
            'Time_BruteForce_sec': avg_bf,
            'Time_Dumer_sec': avg_dumer
        })
        print(f"Done. (BF: {avg_bf:.4f}s | Dumer: {avg_dumer:.4f}s)")

    # Export results for LaTeX integration
    df = pd.DataFrame(results)
    df.to_csv("bf_vs_dumer_results.csv", index=False)

    # Generate visual plot
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    
    # Plotting the benchmark lines
    plt.plot(df['n'], df['Time_BruteForce_sec'], marker='s', linestyle='--', color='red', label="Brute Force (ESD)")
    plt.plot(df['n'], df['Time_Dumer_sec'], marker='o', linestyle='-', color='purple', label="Dumer's MitM")
    
    # Annotate Phase Jumps dynamically
    for i in range(1, len(df)):
        if df['t'].iloc[i] > df['t'].iloc[i-1]:
            plt.axvline(x=df['n'].iloc[i], color='gray', linestyle=':', alpha=0.7)
            plt.text(df['n'].iloc[i] + 0.3, df['Time_BruteForce_sec'].iloc[i], f"$t\\to{df['t'].iloc[i]}$", 
                     color='gray', fontsize=9, verticalalignment='bottom')

    plt.title("Computational Complexity: Brute Force vs Dumer's Algorithm\nFixed Weight $t = \\lfloor(t_{GV}-1)/2\\rfloor$")
    plt.xlabel("Code Length ($n$)")
    plt.ylabel("Execution Time (Seconds) [Log Scale]")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.savefig("plot_bf_vs_dumer.pdf", format='pdf', bbox_inches='tight')
    print("\nFiles successfully exported: 'bf_vs_dumer_results.csv' and 'plot_bf_vs_dumer.pdf'")

if __name__ == "__main__":
    run_benchmark()