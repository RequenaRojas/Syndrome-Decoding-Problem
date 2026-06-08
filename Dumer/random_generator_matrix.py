import numpy as np

def random_generator_matrix(n, k, GF):
    P = GF.Random((k, n-k))
    G = np.hstack((GF.Identity(k), P))
    
    K = GF.Random((k, k))
    while np.linalg.matrix_rank(K) < k:
        K = GF.Random((k, k))
    
    G = K@G
    return G

def random_parity_chech_matrix(n, n_minus_k, GF):
    P = GF.Random((n_minus_k, n-n_minus_k))
    H = np.hstack((P, GF.Identity(n_minus_k)))
    return H
    