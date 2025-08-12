# Script written by Giacomo Nagaro. Inspiration for Runge-Kutta
# method came from Rafael Umeda.

import numpy as np

def mk_unit_Ham(diagonal:list, off_diagonals:list) -> np.array:
    ''' Building a square matrix from a list of diagonals and
    a structured list of off diagonals. We assume symmetry.

    Parameters:
        diagonal (list[float]): List of n floats
        off_diagonals (list[list[float]]): list of off diagonal
            elements grouped by their offset from the diagonal

    Returns:
        numpy.array: a n by n matrix '''
    n = len(diagonal)
    matrix = np.zeros((n,n),dtype=np.complex64)
    for i in range(n):
        matrix[i,i] = diagonal[i]
    for offset, values in enumerate(off_diagonals):
        k = offset + 1
        expected_vals= n-k
        assert expected_vals>=0, f'Exceeding bounds of the {n}x{n} matrix.'
        if len(values) == n - k:
            for i in range(n - k):
                matrix[i, i + k] = values[i]
                matrix[i + k, i] = values[i]
        else:
            raise ValueError(f"Inconsistent length for offset {k} with n = {n}")
    return matrix

def mk_periodic_Ham(n:int, unit_Ham:np.array,linker:float = 0.0) -> np.array:
    '''Use the unit matrix describing a single unit cell and repeat
    n times into a square matrix.

    Parameters:
        n(int): the number of times to repeat the unit matrix
        unit_Ham(np.array): unit Hamiltonian that will be repeated
        linker(float): value for the diagonal element between blocks

    Returns:
        np.array: a mn by mn size matrix where m by m is the size
            of the unit cell'''
    assert unit_Ham.shape[0] == unit_Ham.shape[1], 'Need a SQUARE matrix!'
    m = unit_Ham.shape[0]
    size = n * m
    matrix = np.zeros((size,size),dtype=np.complex64)
    for i in range(n):
        start_position = i * m
        matrix[start_position:start_position+m,start_position:start_position+m] = unit_Ham
    if linker != 0.0:
        for i in range(n-1):
            adjacent = (i+1) * m
            matrix[adjacent, adjacent-1] = matrix[adjacent-1,adjacent] = linker
    return matrix

def mk_mask(size:int) -> np.array:
    '''Creates a size by size matrix filled with ones except on the diagonal.

    Parameters:
        size(int): the output of matrix.shape[0] for a square matrix

    Outputs:
        matrix(np.array): an array of ones with zeros along the diagonal'''
    matrix = np.ones((size,size),dtype=np.complex64) - np.eye(size,dtype=np.complex64)
    return matrix

def randomize_diagonal(matrix:np.array,value:float,) -> np.array:
    '''Add/Subtract each diagonal element by a fraction of energy.

    Parameters:
    matrix(np.array): the square matrix to be modified
    value(float): the limit of how much the diagonal can shift

    Outputs:
    matrix(np.array): the original matrix with a modified diagonal'''
    assert matrix.shape[0] == matrix.shape[1], 'Matrix must be square!'
    matrix[np.diag_indices(matrix.shape[0])] += value*np.random.uniform(-0.5,0.5,matrix.shape[0])
    return matrix

def RKF45(Ham:np.array, rho:np.array, mask:np.array, rt:float, delta:float) -> np.array:
    '''Runge-Kutta-Fehlberg method to solving the Liouville master
    equation with the Lindblad incoherence model incorporated.

    Parameters:
    Ham (np.array) - N by N matrix representing the Hamiltonian
    rho (np.array) - N by N matrix representing the time-dependent
        electron density
    mask (np.array) - N by N matrix used for matrix multiplication
    rt (float) - the dephasing rate in THz
    delta (float) - the size of the time step in ps.

    Output:
    rho (np.array) - N by N matrix after a single time propagation step'''
    prefactor = -1j/(6.582119569e-1) # -i/hbar where hbar is in meV*ps

    def rhodot(r):
        return prefactor * (Ham @ r - r @ Ham) - rt * mask * r

    k1 = rhodot(rho)
    k2 = rhodot(rho + 0.25*delta*k1)
    k3 = rhodot(rho + delta*(3/32*k1 + 9/32*k2))
    k4 = rhodot(rho + delta*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3))
    k5 = rhodot(rho + delta*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4))
    k6 = rhodot(rho + delta*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5))

    rho = rho + delta*(16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 +2/55*k6)

    if not np.all(np.isfinite(rho)):
        print("rho trace:", np.trace(rho))
        raise OverflowError("Non-finite values detected in rho during RKF45")

    return rho

def spread(radial_distances: list, rho: np.array, offset:int = 0) -> np.array:
    '''Computing the mean-square displacement given distances and
    site populations.

    Parameters:
    radial_distances(list): list of distances, in Angstroms, between sites
    offset(int): how many sites off the center of the simulation to set as the origin
    rho(np.array): a density matrix representing the occupations of sites

    Outputs:
    msd(float): the mean-square displacement of the density
    '''
    rad = np.array(radial_distances,dtype=np.float64)
    num_sites = len(rad)
    universe_size = rho.shape[0]
    distance_array = np.zeros(universe_size,dtype=np.float64)
    for i in range(universe_size-1):
        distance_array[i+1] = distance_array[i] + rad[i%num_sites]
    distance_array -= distance_array[universe_size//2+offset]
    squared_distance_array = distance_array**2

    populations = np.diag(rho/np.trace(rho)).real
    if np.any(np.isnan(populations)) or np.any(np.isinf(populations)):
        print("Invalid population values in spread()")
        print("rho trace:", np.trace(rho))
        raise ValueError("NaN or Inf in population vector")
    msd = np.dot(squared_distance_array, populations) - np.dot(distance_array, populations)**2
    return msd

def diff_coefficient(start:int,stop:int,spread_list:list,delta:float) -> float:
    '''Compute the diffusion coefficient in squared centimeters per
    second given a list of mean-square displacements.

    Parameters:
    start(int): which index to begin for the passage of time
    stop(int): which index to end the passage of time
    spread_list(list[float]): a list of floats that represent the
        mean-square displacement of the density
    delta(float): the amount of time in a single step (in picoseconds)

    Outputs:
    D(float): the diffusion coefficient in squared centimeters per second'''

    D = (spread_list[stop]-spread_list[start])*1e-4/(2*(stop-start)*delta)
    return D

def propagate(rho:np.array, Ham:np.array, mask:np.array, gamma:float, delta:float, distances:list,) -> tuple:
    rho = RKF45(Ham, rho, mask, gamma, delta)
    rho = (rho + rho.conj().T)/2
    rho/= np.trace(rho)
    msd = spread(distances,rho)
    pop = np.diag(rho).real
    return rho, (pop, msd)

