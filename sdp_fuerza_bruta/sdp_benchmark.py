import time
import galois
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sdp_coset import solve_sdp_cosets_systematic_form, solve_sdp_cosets

def generate_systematic_instance(n, k, GF):
    """
    Generates a systematic parity-check matrix H so the optimized solver works,
    while the general solver will expose its overhead of computing the RREF.
    """
    r = n - k
    I_r = GF.Identity(r)
    A = GF.Random((r, k))
    H = np.hstack((I_r, A))
    
    y = GF.Random(n)
    s = (H @ y).reshape(r, 1)
    
    return H, s

def run_coset_benchmark():
    n_vals = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    code_rate = 0.5  # R = k/n
    trials_per_n = 10
    GF = galois.GF(2)
    
    results = []

    print(f"{'='*60}")
    print(f"STARTING COSET SDP BENCHMARK: SYSTEMATIC VS GENERAL")
    print(f"Targeting: n = {n_vals}")
    print(f"Code Rate: R = {code_rate}")
    print(f"{'='*60}\n")

    for n in n_vals:
        k = int(n * code_rate)
        
        time_sys_total = 0.0
        time_gen_total = 0.0
        
        print(f"Benchmarking n={n}, k={k}... ", end="", flush=True)
        
        for _ in range(trials_per_n):
            H, s = generate_systematic_instance(n, k, GF)
            
            # --- Benchmark: Systematic Form (Direct Assignment) ---
            start_time = time.perf_counter()
            solve_sdp_cosets_systematic_form(H, s, GF, verbose=False)
            time_sys_total += (time.perf_counter() - start_time)
            
            # --- Benchmark: General Form (RREF Overhead) ---
            start_time = time.perf_counter()
            solve_sdp_cosets(H, s, GF, verbose=False)
            time_gen_total += (time.perf_counter() - start_time)
            
        avg_time_sys = time_sys_total / trials_per_n
        avg_time_gen = time_gen_total / trials_per_n
        
        results.append({
            'n': n,
            'k': k,
            'Time_Sys_sec': avg_time_sys,
            'Time_Gen_sec': avg_time_gen
        })
        print(f"Done. (Sys: {avg_time_sys:.4f}s | Gen: {avg_time_gen:.4f}s)")

    # Export results for LaTeX integration
    df = pd.DataFrame(results[1:])
    df.to_csv("coset_benchmark_results.csv", index=False)

    # Generate visual plot
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    
    plt.plot(df['n'], df['Time_Sys_sec'], marker='o', linestyle='-', color='b', label='Systematic Coset (Direct $z$)')
    plt.plot(df['n'], df['Time_Gen_sec'], marker='s', linestyle='--', color='r', label='General Coset (RREF $z$)')
    
    plt.title('SDP Coset Benchmark: Systematic vs General Matrix\nConstant Rate $R=0.5$')
    plt.xlabel('Code Length ($n$)')
    plt.ylabel('Execution Time (Seconds) [Log Scale]')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.savefig("plot_coset_benchmark.pdf", format='pdf', bbox_inches='tight')
    print("\nFiles successfully exported: 'coset_benchmark_results.csv' and 'plot_coset_benchmark.pdf'")

if __name__ == "__main__":
    run_coset_benchmark()