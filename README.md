# Syndrome Decoding Problem (SDP) Solver & Benchmarking

Repositorio dedicado a la implementación computacional y análisis de complejidad de algoritmos para la resolución del Problema de Decodificación por Síndrome (SDP) en el contexto de la criptografía post-cuántica. 

Este proyecto fue desarrollado como parte de las actividades de investigación académica en la ESFM-IPN.

## 📌 Características Principales

* **Generación de Parámetros:** Scripts para la creación estocástica de matrices de comprobación de paridad ($H$) para códigos lineales aleatorios sobre campos finitos.
* **Búsqueda Exhaustiva:** Implementación determinista (fuerza bruta) para resolver instancias del SDP con parámetros reducidos.
* **Information Set Decoding (ISD):**  * Implementación optimizada del algoritmo de **Dumer** (1991) utilizando técnicas de encuentro en el medio (*meet-in-the-middle*).
* **Benchmarking:** Entorno de pruebas automatizado para perfilar y comparar la complejidad temporal y el escalamiento de los algoritmos frente a variaciones en $n$, $k$ y $q$.
