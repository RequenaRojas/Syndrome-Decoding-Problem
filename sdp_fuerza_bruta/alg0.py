import galois # type: ignore
import numpy as np
from RREF import RREF
from matriz_chequeo_paridad import matriz_chequeo_paridad
from tabulate import tabulate # type: ignore
def generar_combinaciones(aux, k, q):
    if len(aux) >= q**k: return aux
    aux1 = []
    for cad in aux:
        for i in range(q):
            aux1.append(np.insert(cad, 0, i))
    return generar_combinaciones(aux1,k,q )

n = 4
m = 3
q = 2
Fq = galois.GF(q)
G = Fq.Random((m, n))
"""
G = Fq([
[1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
[1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
[1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
[0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
[1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
[1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
])
#"""
print(f"Generator matrix G =")
print(tabulate(G, tablefmt="plain"))
H , k , col_piv = matriz_chequeo_paridad(G, Fq)
print(f" k = {k}")
print(f"\nCheck parity matrix H = \n")
print(tabulate(H, tablefmt="plain"))
print(f"\n Columnas pivote = \n", col_piv)

    
y = Fq.Random(n)
print(f"\ny  = \n", y)
#y = Fq([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
s = H@y.T #Sindrome s != 0
while np.all(s == 0):
    y = Fq.Random(n)
    s = H@y.T
print(f"\nSíndrome s = \n", s)
s_col = s.reshape(-1, 1)
Hs = np.hstack([H, s_col])
print(f"\nSistema Lineal Hs = \n")
print(tabulate(Hs, tablefmt="plain"))

Hs_rref, Hs_row_piv, Hs_col_piv = RREF(Hs, Fq)
Hs_rref = Fq(Hs_rref)
print(f"\nRREF Hs = \n")
print(tabulate(Hs_rref, tablefmt="plain"))
print(f" \nCoordenadas pivote Hs:\n")
print(Hs_row_piv, Hs_col_piv)

not_Hs_col_piv = [i for i in range(n) if i not in Hs_col_piv]


S = np.array(generar_combinaciones([[i] for i in range(q)], k, q))
print(S)

candidatos = []
for comb in S:
    cand = Fq.Zeros(n)
    for l in range(n):
        if l in Hs_col_piv:
            i = Hs_row_piv[Hs_col_piv.index(l)]
            cand[l] = Hs_rref[i][n]
            for j in range(l+1, n):
                if j in Hs_col_piv: continue
                cand[l] -= Hs_rref[i][j]*comb[not_Hs_col_piv.index(j)]
        else:
            cand[l] = comb[not_Hs_col_piv.index(l)]
    candidatos.append(cand)

candidatos = Fq(candidatos)
"""
print("\nCandidatos:")
print(tabulate(candidatos, tablefmt="plain"))
"""
# Verificación (que tengan el mismo sindrome)
for cand in candidatos:
    ver = H@cand.T
    if not np.all(ver == s):
        print("Error")
        break

e_min = min(candidatos, key=lambda v: np.count_nonzero(v))
print("\n e_min = ", e_min)






