import galois
import timeit
import matplotlib.pyplot as plt
import csv 
import os  

matriz_chequeo_paridad = "matriz_chequeo_paridad(G, GF)"
resultados = {}

n = 500
k = 250
Q_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
for q in Q_values:
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
    resultados[q] = mejor_tiempo_seg
    print(f"✅ q={q} : Mejor tiempo = {mejor_tiempo_seg:.6f} s")

print("\n--- RESUMEN DE TIEMPOS VARIANDO Q ---")
for q, tiempo in resultados.items():
    print(f"q={q}: {tiempo:.6f} segundos")

Tiempo_values = list(resultados.values())

plt.figure(figsize=(10, 6)) 
plt.plot(Q_values, Tiempo_values, marker='o', linestyle='-', color='b')


plt.title('Algoritmo Matriz de Chequeo de Paridad')
plt.xlabel('Tamaño campo GF(q)')
plt.ylabel('Tiempo de Ejecución (segundos)')
plt.text(
    x=0.05,
    y=0.90,
    s=r'$n = 500 \quad \text{y} \quad k = 250$',
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(facecolor='lightgray', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5')
)
plt.grid(True, linestyle='--', alpha=0.6) 

plt.savefig('timeit_q_matriz_chequeo_paridad.svg', dpi=300) 

NOMBRE_ARCHIVO_CSV = 'timeit_q_matriz_chequeo_paridad.csv'
ruta_completa = os.path.join(os.getcwd(), NOMBRE_ARCHIVO_CSV)

print(f"\n💾 Guardando resultados en: {ruta_completa}")

with open(ruta_completa, mode='w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    escritor_csv.writerow(['n', 'k', 'q','Tiempo_Mejor_Segundos', 'Tasa_R'])
    tasa_r = 0.5
    for q, tiempo in resultados.items():
        escritor_csv.writerow([n, k, q,f"{tiempo:.6f}", tasa_r])

print(f"✅ Los datos se han guardado con éxito en {NOMBRE_ARCHIVO_CSV}")
