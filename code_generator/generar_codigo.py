import numpy as np
import galois
from itertools import combinations
GF = galois.GF(2)
def random_parity_chech_matrix(n, r, GF):
    """
    Genera una matriz de paridad aleatoria de rango completo.
    
    :param n: Longitud del código (longitud de la palabra).
    :param r: Número de filas de la matriz (r = n - k).
    :param GF: Objeto del Cuerpo de Galois (p. ej. galois.GF).
    """
    H = GF.Random((r, n))
    
    while H.rank() != r:
        H = GF.Random((r, n))
        
    return H

n, k = 16, 8
r = n-k
p = n // 2
H = random_parity_chech_matrix(n, r,GF)
