import numpy as np
import itertools


def lcg(seed, a=1664525, c=1013904223, m=2**32):
    """
    Linear Congruential Generator
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    seed: float
        Seed
    a: float
        Multiplier
        Defaults to ``1664525``
    c: float
        Increment
        Defaults to ``1013904223``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    float:
        Sequence of Pseudo-Random Numbers, having a period, that is sensitive to the choice of a and m.

    """
    # Infinite sequence generator (randomness, "bounded" by supplied parameters)
    while True:
        seed = (a * seed + c) % m
        yield seed


def mlcg(seed, a=1664525, m=2**32):
    """
    Multiplicative / Lehmer Linear Congruential Generator
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    seed: float
        Seed
    a: float
        Multiplier
        Defaults to ``1664525``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    float:
        Sequence of Pseudo-Random Numbers, having a period, that is sensitive to the choice of a and m.

    """
    while True:
        seed = (a * seed) % m
        yield seed
  
        
def lcgList(N:int, range:tuple, seed:float=42, a:float=1664525, c:float=1013904223, m:float=2**32):
    """
    Continuous
    Returns normalized list of LCG-generated random values
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    N: int
        Number of random numbers to be generated
    range: tuple
        Range from where the numbers will be sampled
    seed: float
        Seed
        Defaults to ``42``
    a: float
        Multiplier
        Defaults to ``1664525``
    c: float
        Increment
        Defaults to ``1013904223``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    numpy.ndarray
        Normalized list of LCG-generated random values

    """
    rnList = np.array(list(itertools.islice(lcg(seed, a, c, m), 0, N))) / m
    start, end = range
        
    return end * rnList + start

        
def mlcgList(N:int, range:tuple, seed:float=42, a:float=1664525, m:float=2**32):
    """
    Continuous
    Returns normalized list of MLCG-generated random values
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    N: int
        Number of random numbers to be generated
    range: tuple
        Range from where the numbers will be sampled
    seed: float
        Seed
        Defaults to ``42``
    a: float
        Multiplier
        Defaults to ``1664525``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    numpy.ndarray
        Normalized list of MLCG-generated random values

    """    
    start, end = range

    rnList = np.array(list(itertools.islice(mlcg(seed, a, m), 0, N))) / m
    return end * rnList + start


# Gauss-Seidel Iteration
def gauss_seidel(A, b, init_val, iter_lim=1000, tol=1e-8, info=False):
    """
    Solves Ax = b linear systems using Gauss-Seidel Iteration

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    init_val: numpy.ndarray
        Contains an initial guess for x
    iter_lim: int
        Maximum number of iterations
        Defaults to ``100``
    tol: float
        Tolerance value
        Defaults to ``1e-8``
    info: bool
        Whether to store residue & iteration steps
        Defaults to ``False``
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
    
    Optionally Returns
    ------------------
    meta: list
        List containing iteration steps & residue per step
    
    """
    CONV_FLAG = False # Convergence Flag
    var = init_val # Vector, X
    
    # To store residue & iteration steps
    meta = []
    
    for i in range(iter_lim):
        var_new = np.zeros_like(var) # stores updated values of all variables (Vector, X)

        for j in range(A.shape[0]):
            # Matrix Multiplying all elements, before A's diagonal (in a row) with all corresponding vars (in Vector, X), that now have updated values
            l = np.dot(A[j, :j], var_new[:j]) # Note, the only change from jacobi_iter() is changing "var" to "var_new"
            # Matrix Multiplying all elements, after A's diagonal (in a row) with all corresponding vars (in Vector, X), that do not have updated values yet
            u = np.dot(A[j, j + 1:], var[j + 1:])
            # Updating values of vars
            var_new[j] = (b[j] - l - u) / A[j, j]

        meta.append([i, np.linalg.norm(var - var_new)]) # Storing iteration step and residue

        # Checks, how close the updated values are, to the previous iteration's values and breaks the loop, if close enough (defined by "tol")
        if np.allclose(var, var_new, atol=tol, rtol=0.):
            CONV_FLAG = True
            break

        var = var_new # Storing the new solution

    # If solution is not obtained (no convergence), after iter_lim iterations | Note that, this "else" block belongs to the previous "for" statement and not any "if" statement
    else:
        CONV_FLAG = False
    
    if not CONV_FLAG:
        raise Exception(f"Solution did not converge, after the specified limit of {iter_lim} iterations.")
    
    # Returning convergence flag, metadata, solution and associated error
    if info:
        return CONV_FLAG, meta, var, A @ var - b
    
    # Returning convergence flag, solution and associated error
    return CONV_FLAG, var, A @ var - b


# Gassian Elimination / Gauss-Jordan
def gauss_elim(A, b):
    """
    Solves Ax = b linear systems using Gassian Elimination

    Parameters
    ----------
    A: numpy.ndarray
        Contains coefficients of the variables (Matrix, A)
    b: numpy.ndarray
        Contains the constants on RHS (Vector, b)
    
    Returns
    -------
    bool
        Whether the process converged within ``iter_lim``
    numpy.ndarray
        Obtained solution
    float
        Error in the obtained solution
    
    """
    # Prepping the Augmented Matrix
    aug_mat = np.concatenate((A, np.reshape(b, (-1, 1))), axis=1)
    # Convergence Flag
    CONV_FLAG = True
    # Position of leading nonzero, nondiagonal-element in a row / pivot
    lead = 0
    
    # aug_mat.shape[0] == No. of rows
    # aug_mat[0].shape or aug_mat.shape[1] == Number of Columns
    rowCount = aug_mat.shape[0]
    columnCount = aug_mat.shape[1]

    for r in range(rowCount):
        if lead >= columnCount:
            CONV_FLAG = False
            break
        i = r

        # Finding the pivot in a column
        while aug_mat[i][lead] == 0:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:
                    CONV_FLAG = False
                    break

        aug_mat[i], aug_mat[r] = aug_mat[r], aug_mat[i] # Swapping rows
        lv = aug_mat[r][lead]
        aug_mat[r] = [mrx / float(lv) for mrx in aug_mat[r]]
        for i in range(rowCount):
            if i != r:
                lv = aug_mat[i][lead]         
                aug_mat[i] = [iv - lv*rv for rv,iv in zip(aug_mat[r], aug_mat[i])]
        lead += 1
    
    if not CONV_FLAG:
        raise Exception(f"Solution did not converge.")
    
    # Returning convergence flag, solution and associated error
    return CONV_FLAG, aug_mat[:, -1], A @ aug_mat[:, -1] - b
