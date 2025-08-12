# This script was written by Giacomo Nagaro,
# and adapted for the data provided by Dr. Luke Nambi Mohanam
# Usage of this script goes as follows
# $ python single_run.py gamma e_rand e_rand_rate trial
# where the last four arguments are user-defined.

import numpy as np
from tqdm import tqdm
import conduct
import sys

### --- Parameter Definitions --- ###
OmcS = {'energies':[71.235, 179.83, 0.0, 169.93, 56.58, 24.00],
        'couplings':[[2.08, 16.63, 4.31, 31.53, 3.09]],
        'interchain':21.38,
        'distances':[8.126, 8.594, 8.687, 8.377, 5.493, 7.385],
        }

energies  = OmcS['energies']   # meV
couplings = OmcS['couplings']  # meV
interchain= OmcS['interchain'] # meV
distances = OmcS['distances']  # Angstroms

# Variables
gamma = float(sys.argv[-4]) # THz, 50 THz was used by Rafael Umeda
e_rand = float(sys.argv[-3]) # meV, 0 meV was used by Rafael Umeda
e_rand_rate = float(sys.argv[-2]) # THz, 0.00833 was used by Rafael Umeda
trial = int(sys.argv[-1]) 
size = 120 # Decision from Rafael Umeda
steps = 25000 # Decision from Rafael Umeda
step_size = 0.001 # picoseconds

# Initialization
Hamiltonian = conduct.mk_periodic_Ham(
        size//len(energies)+1,
        conduct.mk_unit_Ham(energies,couplings),
        interchain
        )
Hamiltonian = Hamiltonian[:size,:size]
assert Hamiltonian.shape == (size,size), 'Error in Hamiltonian creation.'
Ham = Hamiltonian.copy()
mask = conduct.mk_mask(size)
rho0 = np.zeros((size,size),dtype=np.complex64)
rho0[size//2,size//2] = 1.0
np.random.seed(trial)

pop_t=np.zeros((steps+1,size+2))
pop_t[0]=np.array([0.0,*np.diagonal(rho0.real),0.0])

for i in tqdm(range(1,steps+1),desc='Propagating'):
    time_at = i * step_size
    if i == 1:
        rho, (pop, msd) = conduct.propagate(Ham=Hamiltonian, rho=rho0, mask=mask, gamma=gamma, delta=step_size, distances=distances)
    if time_at%round(1/e_rand_rate,3)==0:
        Hamiltonian = conduct.randomize_diagonal(Ham,e_rand)
        rho, (pop, msd) = conduct.propagate(Ham=Hamiltonian, rho=rho, mask=mask, gamma=gamma, delta=step_size, distances=distances)
    else:
        rho, (pop, msd) = conduct.propagate(Ham=Hamiltonian, rho=rho, mask=mask, gamma=gamma, delta=step_size, distances=distances)
    pop_t[i]=np.array([time_at,*np.diagonal(rho.real),msd])
# Uncomment line below to save a human-readable file.
# np.savetxt(f'OmcS_{gamma}_{e_rand}_{e_rand_rate}_{trial}.csv',pop_t,delimiter=',',fmt='%.8e')
np.save(f'OmcS_{gamma}_{e_rand}_{e_rand_rate}_{trial}.npy',pop_t,allow_pickle=False)

