import galois
import numpy as np
from tabulate import tabulate
import itertools
from random_generator_matrix import random_generator_matrix
from matriz_chequeo_paridad import matriz_chequeo_paridad



def sdp_espacio_soluciones(H, s, GF, verbose = False):
    n_minus_k , n = H.shape
    k = n - n_minus_k
    q = GF.order

    T = GF.Zeros((n_minus_k, k+1))
    for i in range(n_minus_k):
        for j in range(k):
            T[i][j] = -H[i][j]
        T[i][k] = s[i][0]
    if(verbose):
        print(f"T:")
        print(tabulate(T, tablefmt="plain"))

    espacio_soluciones = GF.Zeros((q**k, n))
    pesos = np.zeros(q**k, dtype=int)

    prod_q_k = list(itertools.product(list(GF.elements), repeat=k))

    for r in range(q**k):
        solucion = GF.Zeros(n)
        for i in range(k):
            solucion[i] = prod_q_k[r][i]
        for i in range(k, n):
            for t in range(k):
                solucion[i] += prod_q_k[r][t]*T[i-k][t]
            solucion[i] += T[i-k][k]
        espacio_soluciones[r] = solucion
        pesos[r] = np.count_nonzero(solucion)
    
    
    if(verbose):
        print(f"Espacio_soluciones:")
        print(tabulate(espacio_soluciones, tablefmt="plain"))

    indice_min_peso = np.argmin(pesos)
    e_min_peso = espacio_soluciones[indice_min_peso].reshape(n, 1)
    if(verbose):
        print(f"e:")
        print(e_min_peso)
    return e_min_peso



n = 5
k = 2
q = 2
GF = galois.GF(q)

def run_single_test():
    G = random_generator_matrix(n, k, GF)
    H, _ = matriz_chequeo_paridad(G, GF)

    y = GF.Random((n, 1))
    s = H@y
    while np.all(s == 0):
        y = GF.Random((n, 1))
        s = H@y
   
    e_min_peso = sdp_espacio_soluciones(H, s, GF)

    x = y + e_min_peso
    verification = H@x
    sucess = np.all(verification == 0)
    return sucess, G, H, y, e_min_peso, verification

success_count = 0
error_count = 0
error_details = []
num_pruebas = 50

for i in range(num_pruebas):
    try:
        success, G, H, y, e, verification = run_single_test()
        
        if success:
            success_count += 1
        else:
            error_count += 1
            error_details.append({
                'iteration': i,
                'G': G.copy(),
                'H': H.copy(),
                'y': y.copy(),
                'e': e.copy(),
                'verification': verification.copy(),
            })
            print(f"Error en iteración {i}")
            
    except Exception as e:
        error_count += 1
        print(f"Excepción en iteración {i}: {str(e)}")
        import traceback
        traceback.print_exc()


print(f"\n{'='*50}")
print(f"RESULTADOS FINALES")
print(f"{'='*50}")
print(f"Total de iteraciones: {num_pruebas}")
print(f"Éxitos: {success_count}")
print(f"Errores: {error_count}")
print(f"Tasa de éxito: {success_count*(100/num_pruebas)}%")

if error_details:
    print(f"\nDetalles de los primeros 3 errores:")
    for i, error in enumerate(error_details[:3]):
        print(f"\nError {i + 1} (Iteración {error['iteration']}):")
        print("Matriz G:")
        print(tabulate(error['G'], tablefmt="plain"))
        print("Matriz H:")
        print(tabulate(error['H'], tablefmt="plain"))
        print("y:")
        print(tabulate(error['y'], tablefmt="plain"))
        print("e:")
        print(e)
        print("verification:")
        print(tabulate(error['verification'], tablefmt="plain"))