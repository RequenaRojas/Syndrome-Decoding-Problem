import galois
import timeit
import matplotlib.pyplot as plt

matriz_chequeo_paridad = "matriz_chequeo_paridad(G, GF)"
resultados_benchmark = {}
for n in range(100, 1001, 100):
    # Tasa R = 0.5
    k = int(n/2)
    q = 2
    NUM_ITERACIONES = 1
    SETUP_CODE = f"""
import galois
from random_generator_matrix import random_generator_matrix
from matriz_chequeo_paridad import matriz_chequeo_paridad
n = {n}
k = {k}
q = {q}

GF = galois.GF(q)
G = random_generator_matrix(n, k, GF)
    """

    t = timeit.Timer(stmt=matriz_chequeo_paridad, setup=SETUP_CODE)
    tiempos = t.repeat(repeat=3, number=1)

    mejor_tiempo_seg = min(tiempos)
    resultados_benchmark[n] = mejor_tiempo_seg
    print(f"✅ n={n} (k={k}): Mejor tiempo = {mejor_tiempo_seg:.6f} s")

print("\n--- RESUMEN DE TIEMPOS DE ESCALABILIDAD ---")
for n, tiempo in resultados_benchmark.items():
    print(f"n={n}: {tiempo:.6f} segundos")

N_values = list(resultados_benchmark.keys())
Tiempo_values = list(resultados_benchmark.values())

plt.figure(figsize=(10, 6)) 
plt.plot(N_values, Tiempo_values, marker='o', linestyle='-', color='b')


plt.title('Escalabilidad del Algoritmo Matriz de Chequeo de Paridad')
plt.xlabel('Tamaño de la Matriz (n)')
plt.ylabel('Tiempo de Ejecución (segundos)')
plt.grid(True, linestyle='--', alpha=0.6) 


plt.show()

plt.savefig('escalabilidad_matriz_paridad.png', dpi=300) # dpi=300 asegura alta resolución