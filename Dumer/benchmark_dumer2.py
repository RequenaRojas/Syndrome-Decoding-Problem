import numpy as np
import galois
from itertools import combinations
import scipy.special
import time
import matplotlib.pyplot as plt

GF = galois.GF(2)

def random_parity_check_matrix(n, r, GF):
    H = GF.Random((r, n))
    while np.linalg.matrix_rank(H) != r:
        H = GF.Random((r, n))
    return H

def calcular_cota_gv(n: int, r: int) -> int:
    espacio_sindromes = 2**r
    volumen_esfera = 0
    for t in range(1, n + 1):
        volumen_esfera += scipy.special.comb(n, t, exact=True)
        if volumen_esfera >= espacio_sindromes:
            return t
    return n

def generate_test_instance(n, r, t):
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


def dumer_solver(n, r, H, v, t_range):
    """
    Ejecuta el algoritmo de Dumer y retorna el tiempo tomado.
    """
    p = n // 2
    start_time = time.perf_counter()
    
    for t in t_range:    
        for t1 in range(max(0, t - (n - p)), min(t, p) + 1):
            t2 = t - t1
            
            Z = []
            
            for pos1 in combinations(range(p), t1):
                e_prime = GF.Zeros(n)
                e_prime[list(pos1)] = 1
                v0 = H @ e_prime
                Z.append((tuple(v0.tolist()), 0, tuple(e_prime.tolist())))
                
            for pos2 in combinations(range(p, n), t2):
                e_double = GF.Zeros(n)
                e_double[list(pos2)] = 1
                v1 = H @ e_double
                v1_plus_v = v1 + v  
                Z.append((tuple(v1_plus_v.tolist()), 1, tuple(e_double.tolist())))

            Z.sort()

            for i in range(len(Z) - 1):
                if (Z[i][0] == Z[i+1][0]) and (Z[i][1] == 0 and Z[i+1][1] == 1):
                    e_izq = GF(Z[i][2])
                    e_der = GF(Z[i+1][2])
                    candidato = e_izq + e_der
                    
                    if np.array_equal(H @ candidato, v):
                        end_time = time.perf_counter()
                        return end_time - start_time 
        
       
    end_time = time.perf_counter()
    return end_time - start_time

# --------------------------------------- CONFIGURACIÓN DEL BENCHMARK ---------------------------------------
# R = 0.5
rango_n = range(10, 40, 2) 
iteraciones_por_n = 10
tiempos_promedio = []

print("Iniciando Benchmark de Dumer (R=0.5, t in [1, D_GV] ...")

for n in rango_n:
    r = n // 2
    tiempo_acumulado = 0.0
    
    print(f"\nEvaluando n = {n} (r = {r})")
    
    for iteracion in range(iteraciones_por_n):

        D_GV = calcular_cota_gv(n, r)
        t_range = range(1, D_GV+1) 
        t = np.random.choice(t_range)
        H, b, e, v = generate_test_instance(n, r, t)

        t_ejecucion = dumer_solver(n, r, H, v, t_range)
        tiempo_acumulado += t_ejecucion
        print(f"  Iteración {iteracion+1}: {t_ejecucion:.4f} seg")
        
    tiempos_promedio.append(tiempo_acumulado / iteraciones_por_n)

# ------------------------------------------ GRAFICACIÓN ------------------------------------------
plt.figure(figsize=(10, 6))

# Gráfica en escala Lineal
plt.subplot(1, 2, 1)
plt.plot(rango_n, tiempos_promedio, marker='o', color='b', linestyle='-')
plt.title("Complejidad (Escala Lineal)")
plt.xlabel("Longitud del código (n)")
plt.ylabel("Tiempo Promedio (segundos)")
plt.grid(True)

# Gráfica en escala Semilogarítmica
plt.subplot(1, 2, 2)
plt.plot(rango_n, tiempos_promedio, marker='o', color='r', linestyle='-')
plt.yscale('log') # Eje Y logarítmico
plt.title("Complejidad (Escala Logarítmica)")
plt.xlabel("Longitud del código (n)")
plt.ylabel("Tiempo Promedio (log sec)")
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()