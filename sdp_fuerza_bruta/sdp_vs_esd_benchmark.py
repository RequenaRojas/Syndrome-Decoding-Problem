import time
import galois
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sdp_coset import solve_sdp_cosets
from edp_brute_force import solve_esd_vectorized

def generate_shared_instance(n, k, t, GF):
    """
    Generates a shared valid instance for both solvers.
    Creates a random full-rank matrix H and a syndrome s derived 
    from an error e of exact weight t.
    """
    r = n - k
    while True:
        H = GF.Random((r, n))
        if np.linalg.matrix_rank(H) == r:
            break
            
    # Inject a known error of strict weight t
    e = GF.Zeros(n)
    positions = np.random.choice(n, t, replace=False)
    e[positions] = 1
    
    s = (H @ e).reshape(r, 1)
    return H, s

def run_comparative_benchmark():
    GF = galois.GF(2)
    trials = 3
    code_rate = 0.5
    delta = 0.2  # Relative target weight t/n
    
    n_vals = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    results = []
    
    print(f"{'='*60}")
    print(f"STARTING COMPARATIVE BENCHMARK: SDP vs ESD")
    print(f"Code Rate: R = {code_rate} | Relative Weight: delta = {delta}")
    print(f"{'='*60}\n")
    
    for n in n_vals:
        k = int(n * code_rate)
        t = max(1, int(n * delta)) 
        
        time_sdp_total = 0.0
        time_esd_total = 0.0
        
        print(f"Benchmarking n={n:02d}, k={k:02d}, t={t:02d}... ", end="", flush=True)
        
        for _ in range(trials):
            H, s = generate_shared_instance(n, k, t, GF)
            
            # --- Measure General SDP (Coset Approach) ---
            start_sdp = time.perf_counter()
            solve_sdp_cosets(H, s, GF, verbose=False)
            time_sdp_total += (time.perf_counter() - start_sdp)
            
            # --- Measure Exact Syndrome Decoding (Vectorized Approach) ---
            start_esd = time.perf_counter()
            solve_esd_vectorized(H, s, t, GF, verbose=False)
            time_esd_total += (time.perf_counter() - start_esd)
            
        avg_time_sdp = time_sdp_total / trials
        avg_time_esd = time_esd_total / trials
        
        results.append({
            'n': n,
            'k': k,
            't': t,
            'Time_SDP_sec': avg_time_sdp,
            'Time_ESD_sec': avg_time_esd
        })
        print(f"Done. (SDP: {avg_time_sdp:.4f}s | ESD: {avg_time_esd:.4f}s)")
        
    # Data Export
    df = pd.DataFrame(results)
    csv_filename = "comparative_sdp_vs_esd_data.csv"
    df.to_csv(csv_filename, index=False)
    
    # Visual Generation
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    
    # Plotting both curves
    plt.plot(df['n'], df['Time_SDP_sec'], marker='o', linestyle='-', color='red', label='General SDP (Coset)')
    plt.plot(df['n'], df['Time_ESD_sec'], marker='s', linestyle='--', color='blue', label=fr'ESD (Target $t \approx 0.2n$)')    
   
    
    plt.title('Brute-Force Complexity: General SDP vs. Exact Syndrome Decoding')
    plt.xlabel('Code Length ($n$)')
    plt.ylabel('Execution Time (Seconds) [Log Scale]')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    pdf_filename = "plot_comparative_sdp_vs_esd.pdf"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"\nBenchmark complete. Files exported: '{csv_filename}' and '{pdf_filename}'")

if __name__ == "__main__":
    run_comparative_benchmark()