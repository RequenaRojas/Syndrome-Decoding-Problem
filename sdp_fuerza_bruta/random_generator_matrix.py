import numpy as np

def random_generator_matrix(n, k, GF):
    P = GF.Random((k, n-k))
    # Matriz G = (I_k | P) de forma sistemática
    G = np.hstack((GF.Identity(k), P))
    
    """
    Para obtener una matriz G no necesariamente
    de forma sistemática, generó una matrix in-
    Q de tamaño kxk y devuelvo QG

    Para generar una matriz aleatoria invertib-
    le probamos y descartamos, el cual es efic-
    iente para Fq (para q=2 y n grande ~29%, p-
    ara q grande es casi 100%)
    """
    
    K = GF.Random((k, k))
    while np.linalg.matrix_rank(K) < k:
        K = GF.Random((k, k))
    
    G = K@G
    return G
    