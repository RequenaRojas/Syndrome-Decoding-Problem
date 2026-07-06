import numpy as np

def RREF(G_original, Fq):
    m, n = G_original.shape
    G = G_original.copy()
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
        for k in R:
            mul = (-G[k][j])/G[i][j] 
            for t in range(j,n):
                G[k][t] += mul*G[i][t]
        R.clear()
        
        for k in range(i+1,m):
            if G[k][j] == 0: continue
            mul = (-G[k][j])/G[i][j]
            for t in range(j,n):
                G[k][t] += mul*G[i][t]

    for i in row_piv:
        j = col_piv[row_piv.index(i)]
        for t in range(j,n):
            if t in col_piv: continue
            mul = G[i][j]
            if mul != Fq(1): G[i][t] *= G[i][j]**-1
    return G, row_piv, col_piv

"""
n = 7
m = 3
q = 2
Fq = galois.GF(q)
G = Fq.Random((m, n))

print(f"G:")
print(tabulate(G, tablefmt="plain"))

G_rref, row_piv, col_piv = RREF(G, Fq)

print(f"G_rref:")
print(tabulate(G_rref, tablefmt="plain"))
"""
