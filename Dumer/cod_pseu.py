import numpy as np
import galois
from itertools import combinations
from tabulate import tabulate 
import math
import scipy.special

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



# ------------------------------------ Parámetros ---------------------------------
n, k = 21, 10

r = n - k
p = n // 2
D_GV = calcular_cota_gv(n,r) # d aprox D_GV
s = 0 # 0: t=D_GV-1//2 fijo (criptoanálisis); 1: t=1,...,D_GV (canal ruidoso)
t_real = 0
if s:
    while t_real == 0:
        t_real = np.random.choice(1, D_GV+1)
        
    t_range = range(D_GV+1)    
else:
    t_real = (D_GV-1)//2
    t_range = range(t_real, t_real+1)


H, b, e, v = generate_test_instance(n, r, t_real)

# Buscar la palabra del peso minímo
""" 
v = GF.Zeros(r)
b = GF.Zeros(n)
#"""

solucion_global = None

print("-" * 50)
print(f"Código [{n},{k}]:   " )
print(tabulate(H, tablefmt="plain"))
print(f"Cota Gilber-Varshamov: D_GV = {D_GV}")
print(f"Límite de búsqueda: t ∈ {t_range}")
print(f"Mensaje recibido b: {b}")
print(f"Error inyectado e: {e}")
print(f"Síndrome H@(b+e) = v: {v}")
print("-" * 50)



# ------------------------------------ Algoritmo de Dumer ------------------------------------
for t in t_range:
    print(f"Explorando peso t = {t}...")
    
    # Iterar sobre las particiones del peso: t1 + t2 = t
    for t1 in range(max(0, t - (n - p)), min(t, p) + 1):
        t2 = t - t1
        
        Z = [] 
        
        # Z': Primera mitad (peso t1)
        for pos1 in combinations(range(p), t1):
            e_prime = GF.Zeros(n)
            e_prime[list(pos1)] = 1
            v0 = H @ e_prime
            Z.append((tuple(v0.tolist()), 0, tuple(e_prime.tolist())))
            
        # Z'': Segunda mitad (peso t2)
        for pos2 in combinations(range(p, n), t2):
            e_double = GF.Zeros(n)
            e_double[list(pos2)] = 1
            v1 = H @ e_double
            v1_plus_v = v1 + v  
            Z.append((tuple(v1_plus_v.tolist()), 1, tuple(e_double.tolist())))

        Z.sort()

        # Buscar las colisiones
        for i in range(len(Z) - 1):
            sind_actual = Z[i][0]
            sind_sig = Z[i+1][0]
            bandera_actual = Z[i][1]
            bandera_sig = Z[i+1][1]
            
            if (sind_actual == sind_sig) and (bandera_actual == 0 and bandera_sig == 1):
                e_izq = GF(Z[i][2])
                e_der = GF(Z[i+1][2])
                candidato = e_izq + e_der
                
                if np.array_equal(H @ candidato, v):
                    solucion_global = candidato
                    break 
                    
        if solucion_global is not None:
            break 
            
    if solucion_global is not None:
        print(f"\nSolución encontrada en t = {t}")
        print(f"Vector de error: {solucion_global}")
        if np.array_equal(solucion_global,e): s = "Exito!" 
        else: s = "No exito"
        print(f"Comprobación: {s}")
        break
        

if solucion_global is None:
    print(f"\nFallo: No se encontraron soluciones en el intervalo {t_range}.")
    print("Probabilidad: El error original b tenía un peso excepcionalmente alto, fuera del promedio esperado por la cota GV.")