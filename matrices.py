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

    return random_projection

def create_optimal_feature_set(data_loader, n_classes, feature_dim, max_total_samples=200, device='cpu'):
    """
    Creates an optimal feature set for orthogonal matrix creation
    """
    reservoir = []
    reservoir_labels = []
    reservoir_size = min(1000, max_total_samples * 5)

    class_sums = torch.zeros(n_classes, feature_dim, device=device)
    class_counts = torch.zeros(n_classes, device=device)
    
    total_samples = 0
    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        batch_size = features.shape[0]
        
        for i in range(n_classes):
            mask = labels == i
            if mask.any():
                class_features = features[mask]
                class_sums[i] += class_features.sum(dim=0)
                class_counts[i] += mask.sum().item()
        
        for feat, label in zip(features, labels):
            total_samples += 1
            
            if len(reservoir) < reservoir_size:
                reservoir.append(feat)
                reservoir_labels.append(label)
            else:
                j = torch.randint(0, total_samples, (1,)).item()
                if j < reservoir_size:
                    reservoir[j] = feat
                    reservoir_labels[j] = label
    
    if reservoir:
        reservoir_tensor = torch.stack(reservoir, dim=0)
        reservoir_labels_tensor = torch.tensor(reservoir_labels, device=device)
    else:
        return torch.zeros(0, feature_dim, device=device)

    valid_classes = class_counts > 0
    class_means = []
    for i in range(n_classes):
        if valid_classes[i]:
            class_mean = class_sums[i] / class_counts[i]
            class_means.append(class_mean.unsqueeze(0))

    def select_diverse_samples(features, n_samples):
        """Select n_samples diverse points from features"""
        if len(features) <= n_samples:
            return features
        
        selected = []
        selected_indices = []

        start_idx = torch.randint(0, len(features), (1,)).item()
        selected.append(features[start_idx].unsqueeze(0))
        selected_indices.append(start_idx)

        for _ in range(n_samples - 1):
            selected_tensor = torch.cat(selected, dim=0)

            distances = torch.cdist(features, selected_tensor)
            min_distances = distances.min(dim=1).values

            farthest_idx = min_distances.argmax().item()

            if farthest_idx not in selected_indices:
                selected.append(features[farthest_idx].unsqueeze(0))
                selected_indices.append(farthest_idx)
        
        return torch.cat(selected, dim=0)

    diverse_samples = select_diverse_samples(reservoir_tensor, min(50, len(reservoir_tensor)))

    boundary_samples = []
    if len(class_means) >= 2 and len(reservoir_tensor) > 0:
        reservoir_means = torch.stack([class_means[i].squeeze() for i in range(len(class_means))])

        distances = torch.cdist(reservoir_tensor, reservoir_means)

        sorted_dist, sorted_idx = distances.sort(dim=1)
        boundary_mask = (sorted_dist[:, 1] - sorted_dist[:, 0]) < 0.5
        
        if boundary_mask.any():
            boundary_samples = reservoir_tensor[boundary_mask][:20]
    
    combined = []
    
    if class_means:
        combined.append(torch.cat(class_means, dim=0))
    
    combined.append(diverse_samples)
    
    if boundary_samples is not None and len(boundary_samples) > 0:
        if isinstance(boundary_samples, list):
            boundary_samples = torch.stack(boundary_samples, dim=0)
        combined.append(boundary_samples)
    
    if len(reservoir_tensor) > 0:
        n_random = min(10, len(reservoir_tensor))
        random_indices = torch.randperm(len(reservoir_tensor))[:n_random]
        combined.append(reservoir_tensor[random_indices])

    if combined:
        final_features = torch.cat(combined, dim=0)
        
        if len(final_features) > max_total_samples:
            final_features = select_diverse_samples(final_features, max_total_samples)
        
        return final_features
    else:
        return torch.zeros(0, feature_dim, device=device)

def create_data_aware_orthogonal_matrix(feature_dim, dim, sample_data=None, device='cpu'):
    """
    Creates orthogonal matrix aligned with data principal components
    """
    if feature_dim >= dim:
        A = torch.randn(feature_dim, dim, device=device)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        base = Q
    else:
        A = torch.randn(dim, feature_dim, device=device)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        base = Q.T
    
    if sample_data is not None and len(sample_data) > 0:
        sample_data = sample_data - sample_data.mean(dim=0, keepdim=True)
        sample_data = sample_data / (sample_data.std(dim=0, keepdim=True) + 1e-8)
        
        projected = sample_data @ base

        cov = projected.T @ projected / (len(projected) - 1)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        
        idx = eigvals.argsort(descending=True)
        eigvecs_sorted = eigvecs[:, idx]
        
        k = max(1, dim // 2)
        rotation = eigvecs_sorted[:, :k]
        
        if rotation.shape[1] < dim:
            complement = torch.randn(dim, dim - rotation.shape[1], device=device)
            complement, _ = torch.linalg.qr(complement, mode='reduced')
            rotation = torch.cat([rotation, complement], dim=1)

        adjusted = base @ rotation

        if feature_dim >= dim:
            Q, _ = torch.linalg.qr(adjusted, mode='reduced')
            return Q
        else:
            Q, _ = torch.linalg.qr(adjusted.T, mode='reduced')
            return Q.T
    
    return base

def create_class_separating_projection(feature_dim, dim, features, labels, n_classes, device='cpu'):
    if feature_dim >= dim:
        A = torch.randn(feature_dim, dim, device=device)
        Q, R = torch.linalg.qr(A, mode='reduced')
        base_proj = Q * torch.sign(torch.diag(R))
    else:
        A = torch.randn(dim, feature_dim, device=device)
        Q, R = torch.linalg.qr(A, mode='reduced')
        base_proj = (Q * torch.sign(torch.diag(R))).T

    if len(features) > 0 and len(torch.unique(labels)) >= 2:
        projected = features @ base_proj

        class_means_proj = []
        for i in range(n_classes):
            mask = labels == i
            if mask.any():
                class_mean = projected[mask].mean(dim=0, keepdim=True)
                class_means_proj.append(class_mean)
        
        if len(class_means_proj) >= 2:
            class_means_proj = torch.cat(class_means_proj, dim=0)

            global_mean = class_means_proj.mean(dim=0, keepdim=True)
            centered = class_means_proj - global_mean

            U, _, Vt = torch.linalg.svd(centered.T, full_matrices=False)

            if U.shape[1] < dim:
                pad = torch.eye(dim, device=device)[:, U.shape[1]:]
                rotation = torch.cat([U, pad], dim=1)
            else:
                rotation = U[:, :dim]

            adjusted_proj = base_proj @ rotation

            if feature_dim >= dim:
                Q, _ = torch.linalg.qr(adjusted_proj, mode='reduced')
                return Q
            else:
                Q, _ = torch.linalg.qr(adjusted_proj.T, mode='reduced')
                return Q.T
    
    return base_proj

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