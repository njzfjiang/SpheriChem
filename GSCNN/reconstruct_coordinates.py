"""
Utility functions for reconstructing 3D coordinates from Coulomb matrix

Coulomb matrix definition:
- C_ii = 0.5 * Z_i^2.4 (diagonal elements)
- C_ij = Z_i * Z_j / |R_i - R_j| (off-diagonal elements, i≠j)

Therefore, we can calculate interatomic distances from off-diagonal elements:
|R_i - R_j| = Z_i * Z_j / C_ij
"""

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from scipy.optimize import minimize


def coulomb_to_distances(C, Z):
    n_atoms = len(Z)
    D = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                D[i, j] = 0.0
            else:
                # R_i - R_j| = Z_i * Z_j / C_ij
                if C[i, j] > 1e-10:
                    D[i, j] = Z[i] * Z[j] / C[i, j]
                else:
                    D[i, j] = 10.0
    
    return D


def reconstruct_coords_mds(C, Z, n_dim=3):
    D = coulomb_to_distances(C, Z)
    
    mds = MDS(n_components=n_dim, dissimilarity='precomputed', random_state=42)
    R = mds.fit_transform(D)
    
    return R


def reconstruct_coords_simple(C, Z):
    n_atoms = len(Z)
    R = np.zeros((n_atoms, 3))
    D = coulomb_to_distances(C, Z)
    R[0] = [0, 0, 0]
    
    if n_atoms > 1:
        R[1] = [D[0, 1], 0, 0]
    
    if n_atoms > 2:
        d01 = D[0, 1]
        d02 = D[0, 2]
        d12 = D[1, 2]
        cos_angle = (d01**2 + d02**2 - d12**2) / (2 * d01 * d02)
        cos_angle = np.clip(cos_angle, -1, 1)
        sin_angle = np.sqrt(1 - cos_angle**2)
        R[2] = [d02 * cos_angle, d02 * sin_angle, 0]
    
    for i in range(3, n_atoms):
        try:
            ref_indices = [0, 1, 2] if i > 3 else [0, 1]
            
            A = []
            b = []
            for ref_idx in ref_indices:
                if ref_idx < i:
                    pass
            
            if len(ref_indices) >= 3:
                p0, p1, p2 = R[ref_indices[0]], R[ref_indices[1]], R[ref_indices[2]]
                d0, d1, d2 = D[i, ref_indices[0]], D[i, ref_indices[1]], D[i, ref_indices[2]]
                
                def objective(x):
                    pos = x.reshape(1, 3)
                    error = 0
                    for idx, ref_idx in enumerate(ref_indices):
                        dist = np.linalg.norm(pos - R[ref_idx:ref_idx+1])
                        error += (dist - D[i, ref_idx])**2
                    return error
                
                weights = 1.0 / (D[i, ref_indices] + 1e-6)
                weights = weights / weights.sum()
                initial_guess = np.sum(weights[:, None] * R[ref_indices], axis=0)
                
                result = minimize(objective, initial_guess, method='BFGS')
                if result.success:
                    R[i] = result.x
                else:
                    R[i] = initial_guess
            else:
                weights = 1.0 / (D[i, :i] + 1e-6)
                weights = weights / weights.sum()
                R[i] = np.sum(weights[:, None] * R[:i], axis=0)
                
        except Exception as e:
            if i > 0:
                weights = 1.0 / (D[i, :i] + 1e-6)
                weights = weights / weights.sum()
                R[i] = np.sum(weights[:, None] * R[:i], axis=0)
    
    return R


def reconstruct_coords_from_coulomb(C, Z, method='mds'):
    valid_mask = Z > 0
    if valid_mask.sum() == 0:
        return np.zeros((len(Z), 3))
    
    C_valid = C[valid_mask][:, valid_mask]
    Z_valid = Z[valid_mask]
    
    if method == 'mds':
        try:
            R_valid = reconstruct_coords_mds(C_valid, Z_valid, n_dim=3)
        except Exception as e:
            print(f"MDS failed, using simple method: {e}")
            R_valid = reconstruct_coords_simple(C_valid, Z_valid)
    else:
        R_valid = reconstruct_coords_simple(C_valid, Z_valid)
    
    R = np.zeros((len(Z), 3))
    R[valid_mask] = R_valid
    
    return R


def reconstruct_coords_batch(C_matrices, Z_list, method='mds', verbose=True):
    R_list = []
    n_molecules = len(Z_list)
    
    for i in range(n_molecules):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing molecule {i+1}/{n_molecules}...")
        
        C = C_matrices[i]
        Z = Z_list[i]
        
        if isinstance(Z, np.ndarray) and Z.ndim > 0:
            n_atoms = len(Z)
        else:
            n_atoms = 1
            Z = np.array([Z])
        
        if C.shape[0] != n_atoms:
            C = C[:n_atoms, :n_atoms]
        
        R = reconstruct_coords_from_coulomb(C, Z, method=method)
        R_list.append(R)
    
    return R_list

