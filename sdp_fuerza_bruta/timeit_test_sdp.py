import galois
import timeit
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

# --- CONFIGURACIÓN ---
# Rango de n: Empezamos bajo porque SDP es exponencial. 
# Precaución: Si subes n > 24, podría tardar mucho.
N_values = list(range(4, 22, 2))  
q = 2
Resultados_Conj = []
Resultados_Coset = []

print(f"--- INICIANDO BENCHMARK SDP (q={q}) ---")
print(f"Guardando resultados en CSV al finalizar...\n")

for n in N_values:
    k = int(n / 2)
    
    # SETUP: Importamos tus funciones dentro del entorno de timeit
    # Nota: Generamos H directamente aquí para asegurar que funcione 
    # incluso si te falta la función 'random_parity_chech_matrix'.
    SETUP_CODE = f"""
import galois
import numpy as np
from matriz_chequeo_paridad import matriz_chequeo_paridad
from sdp_conj_sol import sdp_espacio_soluciones
from sdp_coset import sdp_cosets

n = {n}
k = {k}
q = {q}
GF = galois.GF(q)

# 1. Generar H aleatoria de rango completo
H = GF.Random((n-k, n))
while np.linalg.matrix_rank(H) < n-k:
    H = GF.Random((n-k, n))

# 2. Generar síndrome aleatorio no nulo
s = GF.Random((n-k, 1))
while np.all(s == 0): s = GF.Random((n-k, 1))

# Aplanamos s para sdp_coset si lo requiere como vector 1D
s_flat = s.flatten()
"""

    # --- MEDICIÓN 1: Espacio de Soluciones ---
    t_conj = timeit.Timer(stmt="sdp_espacio_soluciones(H, s, GF)", setup=SETUP_CODE)
    # Ejecutamos 1 vez (number=1) y tomamos el mínimo de 3 repeticiones
    tiempos_conj = t_conj.repeat(repeat=3, number=1)
    mejor_conj = min(tiempos_conj)
    Resultados_Conj.append(mejor_conj)

    # --- MEDICIÓN 2: Líderes de Coset ---
    # Nota: Usamos s_flat o s según como tu función sdp_cosets espere la entrada
    t_coset = timeit.Timer(stmt="sdp_cosets(H, s_flat, GF)", setup=SETUP_CODE)
    tiempos_coset = t_coset.repeat(repeat=3, number=1)
    mejor_coset = min(tiempos_coset)
    Resultados_Coset.append(mejor_coset)

    print(f"✅ n={n} | Soluciones: {mejor_conj:.5f}s | Cosets: {mejor_coset:.5f}s")

# --- GUARDADO EN CSV ---
NOMBRE_ARCHIVO_CSV = 'benchmark_sdp_results.csv'
ruta_completa = os.path.join(os.getcwd(), NOMBRE_ARCHIVO_CSV)

print(f"\n💾 Guardando resultados en: {ruta_completa}")

with open(ruta_completa, mode='w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Cabeceras: Agregamos columnas para ambos tiempos
    escritor_csv.writerow(['n', 'k', 'q', 'Tiempo_Soluciones_s', 'Tiempo_Cosets_s', 'Tasa_R'])

    for i, n in enumerate(N_values):
        k = int(n / 2)
        tasa_r = 0.5
        t_sol = Resultados_Conj[i]
        t_cos = Resultados_Coset[i]
        
        # Escribimos la fila con formato decimal fijo
        escritor_csv.writerow([n, k, q, f"{t_sol:.6f}", f"{t_cos:.6f}", tasa_r])

print(f"✅ CSV generado exitosamente.")

# --- GRAFICACIÓN ---
plt.figure(figsize=(10, 6))
plt.plot(N_values, Resultados_Conj, marker='o', label='Espacio Soluciones')
plt.plot(N_values, Resultados_Coset, marker='s', label='Líderes de Coset')

# Escala Logarítmica (Vital para ver la exponencial)
plt.yscale('log') 

plt.title(f'Comparación de Complejidad SDP (q={q})')
plt.xlabel('Longitud del Código (n)')
plt.ylabel('Tiempo de Ejecución (s) [Escala Log]')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('benchmark_sdp_plot.png', dpi=300)
print("✅ Gráfica guardada como 'benchmark_sdp_plot.png'")