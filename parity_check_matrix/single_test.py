import galois # type: ignore
import numpy as np
from tabulate import tabulate # type: ignore
from matriz_chequeo_paridad import matriz_chequeo_paridad
from random_generator_matrix import random_generator_matrix



n = 10
k = 5
q = 2
GF = galois.GF(q)

def run_single_test():
    G = random_generator_matrix(n, k, GF)
    H = matriz_chequeo_paridad(G, GF)[0]
    verification = G@(H.T)
    return np.all(verification == 0), G, H, verification, k

success_count = 0
error_count = 0
error_details = []

for i in range(500):
    try:
        success, G, H, verification, k = run_single_test()
        
        if success:
            success_count += 1
        else:
            error_count += 1
            error_details.append({
                'iteration': i,
                'G': G.copy(),
                'H': H.copy(),
                'verification': verification.copy()
            })
            print(f"Error en iteración {i}")
            
    except Exception as e:
        error_count += 1
        print(f"Excepción en iteración {i}: {str(e)}")
        import traceback
        traceback.print_exc()

# Resultados
print(f"\n{'='*50}")
print(f"RESULTADOS FINALES")
print(f"{'='*50}")
print(f"Total de iteraciones: 500")
print(f"Éxitos: {success_count}")
print(f"Errores: {error_count}")
print(f"Tasa de éxito: {success_count/5}%")

if error_details:
    print(f"\nDetalles de los primeros 3 errores:")
    for i, error in enumerate(error_details[:3]):
        print(f"\nError {i + 1} (Iteración {error['iteration']}):")
        print("Matriz G:")
        print(tabulate(error['G'], tablefmt="plain"))
        print("Matriz H:")
        print(tabulate(error['H'], tablefmt="plain"))
        print("G * H^T:")
        print(tabulate(error['verification'], tablefmt="plain"))



