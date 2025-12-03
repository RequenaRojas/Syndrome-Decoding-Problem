import galois
import numpy as np
from tabulate import tabulate
import itertools
from random_generator_matrix import random_generator_matrix, random_parity_chech_matrix
from matriz_chequeo_paridad import matriz_chequeo_paridad


def sdp_cosets(H, s, GF, verbose = False):
    n_minus_k , n = H.shape
    k = n - n_minus_k
    q = GF.order

    G , _= matriz_chequeo_paridad(H, GF)
    if verbose:
        print(f"G:")
        print(tabulate(G, tablefmt="plain"))

    z = GF.Zeros(n)
    z[-(n-k):] = s

    k_list = list(itertools.product(list(GF.elements), repeat=k))
    espacio_producto_k = GF(k_list)

    codewords = espacio_producto_k @ G

    coset = codewords + z

    pesos = np.count_nonzero(coset.view(np.ndarray), axis=1)
    if verbose:
        print(f"Espacio_coset:")
        print(tabulate(coset, tablefmt="plain"))
        print(pesos)
    
    indice_min_peso = np.argmin(pesos)
    e_min_peso = coset[indice_min_peso]

    if verbose:
        print(f"e:")
        print(e_min_peso)
    
    return e_min_peso, G



n = 5
k = 2
q = 2
GF = galois.GF(q)

def run_single_test():
    H = random_parity_chech_matrix(n, n-k, GF)
    y = GF.Random(n)
    s = H@y
    while np.all(s == 0):
        y = GF.Random(n)
        s = H@y
   
    e_min_peso, G = sdp_cosets(H, s, GF)

    x = y + e_min_peso
    verification = H@x
    sucess = np.all(verification == 0)
    return sucess, G, H, y, s, e_min_peso, verification

success_count = 0
error_count = 0
error_details = []
num_pruebas = 500

for i in range(num_pruebas):
    try:
        success, G, H, y, s, e, verification = run_single_test()
        
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
        print("s:")
        print(tabulate(error['s'], tablefmt="plain"))
        print("e:")
        print(e)
        print("verification:")
        print(tabulate(error['verification'], tablefmt="plain"))