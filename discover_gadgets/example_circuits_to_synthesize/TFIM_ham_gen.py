import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def pauli_matrices():
    """Return the Pauli matrices."""
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)
    return sigma_x, sigma_y, sigma_z, identity

def construct_hamiltonian(J, h, N):
    """Construct the Hamiltonian matrix for the transverse field Ising model.
    J = the hopping matrix element,
    h = strength of magnetic field
    N = number of spins = number of qubits.

    Some information:

    -- At h<1, the system is in the ordered phase. In this phase the ground state breaks the spin-flip symmetry. 
               Thus, the ground state is in fact two-fold degenerate.
    -- At h=1, we observe quantum phase transition.
    -- At h>1, the system is in the disordered phase. Here, the ground state does preserve the spin-flip symmetry, 
               and is nondegenerate.
    """

    sigma_x, _, sigma_z, _ = pauli_matrices()
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    # Interaction terms: -J * sigma_i^z * sigma_{i+1}^z
    for i in range(N - 1):
        term = np.kron(np.eye(2**i), np.kron(sigma_z, sigma_z))
        term = np.kron(term, np.eye(2**(N - i - 2)))
        H -= J * term

    # Transverse field terms: -h * sigma_i^x
    for i in range(N):
        term = np.kron(np.eye(2**i), sigma_x)
        term = np.kron(term, np.eye(2**(N - i - 1)))
        H -= h * term
    
    return H


def TFIM_hardness_measure_plot():
    J = 1 # HOPPIN STRENGTH
    for q in [2]:
        N = q # NUMBER OF QUBITS
        energy_sep_avg = []
        x_axis = np.arange(0.0, 1+0.01, 0.001)
        for h in x_axis:
            h = np.round(h,2)
            H = construct_hamiltonian(J, h, N)
            eigs = list(sorted(la.eig(H)[0].real))
            smallest_eig = list(sorted(eigs))[0]
            energy_sep = []
            for el in [1]:
                energy_sep.append(np.abs(smallest_eig - eigs[el]))
            energy_sep_avg.append(np.mean(energy_sep))    
        plt.semilogx( x_axis, energy_sep_avg, '--o', label = f'qubits: {N}')
    plt.legend(ncol = 3)
    plt.ylabel('$\\Delta E$', fontsize = 12)
    plt.xlabel('$h$', fontsize = 12)
    plt.savefig('delta_with_h_log.pdf')
    plt.savefig('delta_with_h_log.png')
    plt.show()


def hamitlonian_contruct_with_hardness(mag_field_strength_list):
    """
    J fixed
        -- h << J hard problem. 
            - Ferromagnetic region
            - Phase transition might occurs
            - |Ground - Excited| ~ very small
            - Many local minima
        -- h >> J easier problem
            - Paramagnetic region
    """
    J = 1 # HOPPIN STRENGTH
    N = 2 # NUMBER OF QUBITS
    for h in mag_field_strength_list:
        print(h)
        H = construct_hamiltonian(J, h, N)
        print(f'J:{J}, h:{h}, N:{N}')
        print('Sum of pauli coeff:', (N-1)*-J+ N*-h)
        ham = dict()
        ham['hamiltonian'], ham['eigvals'] = H, la.eig(H)[0].real
        print('Minimum eigenvalue', np.min(ham['eigvals']))
        np.savez(f'hamiltonians/tfim_{N}q_j{J}_h{h}.npz', **ham)
        ham = np.load(f"hamiltonians/tfim_{N}q_j{J}_h{h}.npz")
        _, eigvals = ham['hamiltonian'], ham['eigvals']
        print('Eigenvalues', eigvals)
        print('x-x-x-x-x-x-x-x DONE x-x-x-x-x-x-x-x')
        print()

if __name__ == "__main__":
    mag_field_strength_list = [0.001, 0.1, 1, 1.5]
    hamitlonian_contruct_with_hardness(mag_field_strength_list)
