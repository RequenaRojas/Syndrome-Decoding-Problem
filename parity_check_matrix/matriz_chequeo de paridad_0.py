import galois
import numpy as np
from tabulate import tabulate

def aux(row_piv, col_piv, j):
    s = col_piv[row_piv.index(j)]
    if s in row_piv:
        return aux(row_piv, col_piv, s)
    else: return s

n = 7
m = 3
q = 2
Fq = galois.GF(q)

G_original = Fq.Random((m, n))
"""
G_original = Fq([ 
    
[1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
[1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
[1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
[0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

])
#"""
G = G_original.copy()

print(f"Generator matrix G_original:")
print(tabulate(G_original, tablefmt="plain"))


row_piv = []
col_piv = []
R = []
added = False
for j in range(n): 
    if len(col_piv) == m: break
    for i in range(m):
        if G[i, j] != 0: 
            for r in row_piv:
                if i == r:                 
                    R.append(i)
                    added = True
                    break
            if added: 
                added = False
                if i == m-1: R.clear()
                continue
            row_piv.append(i)
            col_piv.append(j)
            break
        else:
            if i == m-1: R.clear()
    #print(f" \ni = {i}, j = {j}")
    #print(R)
    for k in R:
        #print("#R")
        mul = (-G[k][j])/G[i][j] 
        for t in range(j+1,n):
            G[k][t] += mul*G[i][t]
    R.clear()
    
    for k in range(i+1,m):
        #print("#")
        if G[k][j] == 0: continue
        mul = (-G[k][j])/G[i][j]
        for t in range(j+1,n):
            G[k][t] += mul*G[i][t]

#Hacer entrada del pivote 1
for i in row_piv:
    j = col_piv[row_piv.index(i)]
    for t in range(j+1,n):
        if t in col_piv: continue
        mul = G[i][j]
        if mul != Fq(1): G[i][t] *= G[i][j]**-1

k = len(row_piv)
# Eliminar filas cero
if k != m:
    not_row_piv = [x for x in range(m) if x not in row_piv]
    for p in not_row_piv:
        #print(p)
        G = np.delete(G, p, axis=0)
        #Actualizar pivotes
        for s in range(k):
            if row_piv[s] > p: row_piv[s] = row_piv[s]-1


print(f"\nCoordenadas pivotes:\n{row_piv}\n{col_piv}\n")
#print(tabulate(G, tablefmt="plain"))


print(f"\nForma RREF G: \n")
print(tabulate(G, tablefmt="plain"))
#print(f"\nCoordenadas pivotes:\n{row_piv}\n{col_piv}\n")

# Complementar permutación
for j in range(n):
    if j in col_piv: continue
    if j in row_piv:     
        s = aux(row_piv, col_piv, j)
        row_piv.append(s)
        col_piv.append(j)
    else:
        row_piv.append(j)
        col_piv.append(j)
print(f"\n Permutación:\n{row_piv}\n{col_piv}\n")


# Obtener -A^t
minus_A_transpote = Fq.Zeros((n-k, k))

aux1 = row_piv[k:].copy()
#print(aux1)
aux1.sort()
#print(aux1)
for l in range(n-k):
    s = col_piv[row_piv.index(aux1[l])]
    #print(f"\nl = {l}, s = {s} \n")
    #print(G[:, s])
    minus_A_transpote[l, :] = [-G[r, s] for r in range(k)]

H_est = np.hstack((minus_A_transpote, Fq.Identity(n-k)))
print(f"\n H_est (forma estándar) = \n")
print(tabulate(H_est, tablefmt="plain"))


H = Fq.Zeros((n-k, n))
for k in range(n):
    H[:,col_piv[k]] = H_est[:, row_piv[k]]

print(f"\n H = \n")
print(tabulate(H, tablefmt="plain"))

print(f"\n G*H^t = \n")
print(tabulate(G_original@(H.T), tablefmt="plain"))


