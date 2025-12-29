import torch
import torch.nn.functional as F

def create_gaussian_orthogonal_matrix(feature_dim, dim, device='cpu'):
    if feature_dim < dim:
        A = torch.randn(dim, feature_dim)
        Q, R = torch.linalg.qr(A, mode='reduced')
        random_projection = Q * torch.sign(torch.diag(R))
        random_projection = random_projection.T.to(device)
    else:
        A = torch.randn(feature_dim, dim)
        Q, R = torch.linalg.qr(A, mode='reduced')
        random_projection = Q * torch.sign(torch.diag(R))
        random_projection = random_projection.to(device)

def create_srht_matrix(feature_dim, dim, device='cpu'):
    """
    Create SRHT projection matrix of shape (feature_dim, dim)
    Can be used as: compressed = features @ random_projection
    """
    m = feature_dim
    d = dim

    m_pad = 1 << (m - 1).bit_length()
    D = torch.diag(torch.randint(0, 2, (m_pad,), device=device) * 2 - 1).float()

    HD = fast_hadamard_transform(D)

    if m_pad > m:
        indices = torch.randperm(m_pad, device=device)[:m]
        HD = HD[indices, :]
    
    if d < m_pad:
        col_indices = torch.randperm(m_pad, device=device)[:d]
        projection = HD[:, col_indices]
    else:
        projection = F.pad(HD, (0, d - m_pad), 'constant', 0)

    projection = projection / torch.sqrt(torch.tensor(d, dtype=torch.float32))
    
    return projection

def fast_hadamard_transform(x):
    """Fast Walsh-Hadamard Transform (in-place)"""
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[..., j]
                v = x[..., j + h]
                x[..., j] = u + v
                x[..., j + h] = u - v
        h *= 2
    return x

def create_database_friendly_matrix(feature_dim, dim, device='cpu'):
    """
    Database-friendly random projection (Dasgupta et al.)
    Entries: ±1 with prob 1/4, 0 with prob 1/2
    """
    rand_vals = torch.rand(feature_dim, dim, device=device)

    projection = torch.zeros((feature_dim, dim), device=device)
    projection[rand_vals < 0.25] = 1.0
    projection[(rand_vals >= 0.25) & (rand_vals < 0.5)] = -1.0
    
    projection = projection * torch.sqrt(torch.tensor(3.0, dtype=torch.float32))
    
    return projection

def create_sparse_random_matrix(feature_dim, dim, s=3, device='cpu'):
    """
    Sparse random matrix (Achlioptas, 2003)
    s=3: entries are ±√3 with prob 1/6, 0 with prob 2/3
    s=1: entries are ±1 with prob 1/2 (very sparse)
    """
    prob = 1.0 / (2 * s)
    mask = torch.rand(feature_dim, dim, device=device) < prob
    
    signs = torch.randint(0, 2, (feature_dim, dim), device=device) * 2 - 1
    projection = (mask.float() * signs.float()).to(device)
    
    projection = projection * torch.sqrt(torch.tensor(s, dtype=torch.float32))
    
    return projection

def create_very_sparse_matrix(feature_dim, dim, device='cpu'):
    """
    Very sparse random projection (Li et al., 2006)
    Entries: ±1 with prob 1/(2√d), 0 with prob 1-1/√d
    """
    d_sqrt = int(torch.sqrt(torch.tensor(dim, dtype=torch.float32)).item())
    prob = 1.0 / (2 * d_sqrt)
    
    mask = torch.rand(feature_dim, dim, device=device) < prob
    signs = torch.randint(0, 2, (feature_dim, dim), device=device) * 2 - 1
    projection = (mask.float() * signs.float()).to(device)
    
    projection = projection * torch.sqrt(torch.tensor(d_sqrt, dtype=torch.float32))
    
    return projection

def create_jl_gaussian_matrix(feature_dim, dim, device='cpu'):
    """
    Standard Gaussian matrix with JL Lemma guarantees
    Preserves distances with high probability
    """
    projection = torch.randn(feature_dim, dim, device=device)
    
    projection = projection / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
    
    return projection

def create_orthogonal_gaussian_mix(feature_dim, dim, alpha=0.5, device='cpu'):
    """
    Mixture of orthogonal and Gaussian projections
    alpha: blend factor (0=full Gaussian, 1=full orthogonal)
    """
    if feature_dim >= dim:
        A_orth = torch.randn(feature_dim, dim, device=device)
        Q, _ = torch.linalg.qr(A_orth, mode='reduced')
        orth_part = Q
    else:
        A_orth = torch.randn(dim, feature_dim, device=device)
        Q, _ = torch.linalg.qr(A_orth, mode='reduced')
        orth_part = Q.T
    
    gauss_part = torch.randn(feature_dim, dim, device=device) / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
    
    projection = alpha * orth_part + (1 - alpha) * gauss_part

    projection = projection / torch.norm(projection, dim=0, keepdim=True).clamp(min=1e-8)
    
    return projection