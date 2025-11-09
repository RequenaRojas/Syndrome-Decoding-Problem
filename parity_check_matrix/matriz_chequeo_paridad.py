import galois # type: ignore
import numpy as np
from RREF import RREF
from tabulate import tabulate # type: ignore

def aux(row_piv, col_piv, j):
    s = col_piv[row_piv.index(j)]
    if s in row_piv:
        return aux(row_piv, col_piv, s)
    else: return s

def matriz_chequeo_paridad_sin_k(G, GF, verbose=False):
    m, n = G.shape
    G_rref, row_piv, col_piv = RREF(G, GF)
    if verbose:
        print(f"\nCoordenadas pivotes:\n{row_piv}\n{col_piv}\n")
        print(f"\nForma RREF G: \n")
        print(tabulate(G, tablefmt="plain"))

    k = len(row_piv)
    if k != m:
        not_row_piv = [x for x in range(m) if x not in row_piv]
        for p in not_row_piv:
            G_rref = np.delete(G_rref, p, axis=0)
            for s in range(k):
                if row_piv[s] > p: row_piv[s] = row_piv[s]-1

    if verbose:
        print(f"\n k = {k}\n")
        print(f"\nCoordenadas pivotes:\n{row_piv}\n{col_piv}\n")
        print(f"\nForma RREF G: \n")
        print(tabulate(G_rref, tablefmt="plain"))

    for j in range(n):
        if j in col_piv: continue
        if j in row_piv:     
            s = aux(row_piv, col_piv, j)
            row_piv.append(s)
            col_piv.append(j)
        else:
            row_piv.append(j)
            col_piv.append(j)
    if verbose: 
        print(f"\n Permutación:\n{row_piv}\n{col_piv}\n")

    minus_A_transpote = GF.Zeros((n-k, k))

    aux1 = row_piv[k:].copy()
    aux1.sort()
    for l in range(n-k):
        s = col_piv[row_piv.index(aux1[l])]
        minus_A_transpote[l, :] = [-G_rref[r, s] for r in range(k)]

    H_est = np.hstack((minus_A_transpote, GF.Identity(n-k)))
    if verbose:
        print(f"\n H_est (forma estándar) = \n")
        print(tabulate(H_est, tablefmt="plain"))

    H = GF.Zeros((n-k, n))
    for t in range(n):
        H[:,col_piv[t]] = H_est[:, row_piv[t]]
    
    
    return H, k, col_piv
def matriz_chequeo_paridad(G, GF, verbose=False):
    k, n = G.shape
    G_rref, row_piv, col_piv = RREF(G, GF)
    if verbose:
        print(f"\nCoordenadas pivotes:\n{row_piv}\n{col_piv}\n")
        print(f"\nForma RREF G: \n")
        print(tabulate(G, tablefmt="plain"))

    if verbose:
        print(f"\nCoordenadas pivotes:\n{row_piv}\n{col_piv}\n")
        print(f"\nForma RREF G: \n")
        print(tabulate(G_rref, tablefmt="plain"))

    for j in range(n):
        if j in col_piv: continue
        if j in row_piv:     
            s = aux(row_piv, col_piv, j)
            row_piv.append(s)
            col_piv.append(j)
        else:
            row_piv.append(j)
            col_piv.append(j)
    if verbose: 
        print(f"\n Permutación:\n{row_piv}\n{col_piv}\n")

    minus_A_transpote = GF.Zeros((n-k, k))

    aux1 = row_piv[k:].copy()
    aux1.sort()
    for l in range(n-k):
        s = col_piv[row_piv.index(aux1[l])]
        minus_A_transpote[l, :] = [-G_rref[r, s] for r in range(k)]

    H_est = np.hstack((minus_A_transpote, GF.Identity(n-k)))
    if verbose:
        print(f"\n H_est (forma estándar) = \n")
        print(tabulate(H_est, tablefmt="plain"))

    H = GF.Zeros((n-k, n))
    for t in range(n):
        H[:,col_piv[t]] = H_est[:, row_piv[t]]
    
    
    return H, col_piv

"""
n = 7
m = 3
q = 2
Fq = galois.GF(q)

G = Fq.Random((m, n))

print(f"Generator matrix G =")
print(tabulate(G, tablefmt="plain"))

H = matriz_chequeo_paridad(G, Fq, False)


print(f"\nCheck parity matrix H = \n")
print(tabulate(H, tablefmt="plain"))

print(f"\n G*H^t = \n")
print(tabulate(G@(H.T), tablefmt="plain"))
"""



