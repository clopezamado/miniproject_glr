import numpy as np
import networkx as nx
import torch.nn as nn
from torch_geometric.utils import to_networkx
import scipy.sparse as sp
import torch
from torch_geometric.utils import get_laplacian
import cvxpy as cp
from sklearn.metrics.pairwise import euclidean_distances

    

def get_evolution_dirichlet_energy(G, listFeatures):
    num_layers = len(listFeatures)
    energy = np.zeros(num_layers)
    
    # Precompute degrees for all nodes
    degrees = {node: len(list(G.neighbors(node))) for node in G.nodes}

    # Iterate over edges instead of nodes to reduce redundant calculations
    for u, v in G.edges:
        for k in range(num_layers):
            diff = listFeatures[k][u] - listFeatures[k][v]
            energy[k] += np.linalg.norm(diff) ** 2 * (1 / degrees[u] + 1 / degrees[v])
    
    # Normalize by the number of nodes
    #energy /= len(G.nodes)
    
    return energy
def get_dirichlet_energy_matrix(G, listFeatures):
    # Number of layers
    num_layers = len(listFeatures)

    # Compute normalized Laplacian matrix
    A = nx.adjacency_matrix(G).tocsc()  # Adjacency matrix (sparse)
    degrees = np.array(A.sum(axis=1)).flatten()  # Degree vector
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))  # D^(-1/2)
    L_norm = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt  # L_norm = I - D^(-1/2) A D^(-1/2)
    L_norm_dense = torch.tensor(L_norm.toarray(), dtype=torch.float32)
    # Compute Dirichlet energy for each layer
    energies = np.zeros(num_layers)
    for k,X in enumerate(listFeatures):
        X_t=X.T.numpy()
        energies[k] = np.trace(X_t @ L_norm @ X.numpy())
        if energies[k]<0:
            hola=1

    return energies
def clip_negative_eigenvalues(matrix):
    """
    Clips negative eigenvalues of a symmetric matrix to zero.

    Args:
        matrix (torch.Tensor): A symmetric matrix of shape (n, n).

    Returns:
        torch.Tensor: The modified matrix with non-negative eigenvalues.
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # Clip negative eigenvalues to zero
    eigenvalues_clipped = torch.clamp(eigenvalues, min=1e-15)
    # Reconstruct the matrix
    clipped_matrix = eigenvectors @ torch.diag(eigenvalues_clipped) @ eigenvectors.T
    return clipped_matrix
def get_dirichlet_energy_matrix_2(G, listFeatures):
    # Number of layers
    num_layers = len(listFeatures)
    # Number of nodes
    n = G.num_nodes
    #Normalized laplacian
    laplacian_edge_index, laplacian_edge_weight = get_laplacian(edge_index=G.edge_index, edge_weight=None,normalization='sym',num_nodes=n)
    L_norm=torch.sparse.FloatTensor(laplacian_edge_index, laplacian_edge_weight, torch.Size([n, n]))
    L_norm=L_norm.to(torch.float64).detach().to_dense()
    L_norm=clip_negative_eigenvalues(L_norm)
    L_norm=L_norm.numpy()
    #eigenvalues = torch.linalg.eigvalsh(L_norm)
    #print("Eigenvalues of L_norm:", eigenvalues)
    #if eigenvalues[eigenvalues < 0].numel() > 0:
    #    hola=1
    #Numerical stability
    
    # Compute Dirichlet energy for each layer
    energies = np.zeros(num_layers)

    for k,X in enumerate(listFeatures):
        X=X.to(torch.float64).numpy()
        P = cp.Variable((X.shape[1], X.shape[1]), symmetric=True)
        #Numerical stability
        X[np.abs(X) < 1e-16] = 0
        expr = X.T @ L_norm @ X
        constraints = [P >> 0, P == expr]
        objective = cp.Minimize(0)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        e=P.value
        if not (np.all(np.diag(e) >= 0)):
            hola=1
        #energies[k] = torch.matmul(X.T, torch.matmul(L_norm, X)).trace().item()
        #e=np.dot(np.dot(X.T, L_norm), X)
        #Numerical stability
        e[np.abs(e) < 1e-16] = 0
        energies[k] = np.diag(e).sum()
    #    if energies[k]<0:
    #        hola=1
    return energies

def get_dirichlet_energy_model(model:nn.Module,data):
    
    embeddings_perlayer=model.generate_node_embeddings_perlayer(data.x,data.edge_index)
    #G = to_networkx(data)
    #energy=get_dirichlet_energy_matrix(G,embeddings_perlayer)
    energy=get_dirichlet_energy_matrix_2(data,embeddings_perlayer)
    return energy

def get_node_similarity(model:nn.Module,data):
    embeddings_perlayer=model.generate_node_embeddings_perlayer(data.x,data.edge_index)
    similarities=[]
    for X in embeddings_perlayer:
        distance_matrix = np.square(euclidean_distances(X))
        similarities.append(np.sum(distance_matrix))
    return similarities
