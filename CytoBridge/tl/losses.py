import torch
import ot
from geomloss import SamplesLoss

def calc_ot_loss(x1, x2, lnw1, method = 'emd_detach'):
    '''
    Calculate the OT loss between two sets of data.
    x1: the generated data
    x2: the subset of data
    lnw1: the log weights of the generated data
    method: the method to calculate the OT loss
        'emd_detach': use the EMD loss, detach the mass part, so the growth network will not be trained. More stable.
        'sinkhorn': use the Sinkhorn loss, the growth network will be trained.
    '''

    # Calculate weights
    m1 = torch.exp(lnw1)
    m1 = m1 / m1.sum()
    m2 = torch.ones(x2.shape[0],1).float().to(x1.device)
    m2 = m2 / m2.sum()
    m1 = m1.squeeze(1)
    m2 = m2.squeeze(1)
    if method == 'emd_detach':
        M = torch.cdist(x1, x2)**2
        pi = ot.emd(m1.detach().cpu().numpy(), m2.detach().cpu().numpy(), M.detach().cpu().numpy())
        pi = torch.tensor(pi).float().to(x1.device)
        loss_ot = torch.sum(pi * M)
    # TODO: maybe add emd without detach
    elif method == 'sinkhorn_detach':
        loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=0.1)
        loss_ot = loss_fn(m1.detach(), x1, m2.detach(), x2)
    elif method == 'sinkhorn':
        loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=0.1)
        loss_ot = loss_fn(m1, x1, m2, x2)
    return loss_ot

def calc_mass_loss(x1, x2, lnw1, relative_mass, global_mass = False):
    '''
    Calculate the mass loss between two sets of data.
    x1: the generated data
    x2: the subset of data
    lnw1: the log weights of the generated data
    relative_mass: the relative mass of the subset of data
    '''
    m1 = torch.exp(lnw1)
    local_mass_loss = calc_mass_matching_loss(x1, x2, lnw1, relative_mass)
    if global_mass:
        global_mass_loss = torch.norm(torch.sum(m1) - relative_mass, p=2)**2
        mass_loss = global_mass_loss + local_mass_loss
    else:
        mass_loss = local_mass_loss
    return mass_loss


def calc_mass_matching_loss(data_t1, x_t_last, lnw_t_last, relative_mass, dim_reducer=None):
    """
    Calculate mass matching loss, optionally on dimensionality-reduced data.

    Parameters:
    - data_t1: tensor or array, shape (n_samples, n_features)
    - x_t_last: tensor or array, shape (m_samples, n_features)
    - lnw_t_last: tensor, shape (m_samples, 1)
    - relative_mass: float
    - dim_reducer: callable, optional (default=None)
        A function or object that takes a tensor/array and returns reduced data. If None, no reduction is performed.

    Returns:
    - local_mass_loss: float, the mass loss value
    """
    # Step 0: If dimension reducer is provided, reduce data dimensions
    if dim_reducer is not None:
        data_t1_reduced = dim_reducer(data_t1.detach().cpu().numpy())
        x_t_last_reduced = dim_reducer(x_t_last.detach().cpu().numpy())
    else:
        data_t1_reduced = data_t1
        x_t_last_reduced = x_t_last
    
    batch_size = data_t1.shape[0]
    
    # Ensure data is PyTorch tensor
    data_t1_reduced = torch.as_tensor(data_t1_reduced)
    x_t_last_reduced = torch.as_tensor(x_t_last_reduced)
    
    # Step 1: Find closest point in x_t_last for each point in data_t1
    distances = torch.cdist(data_t1_reduced, x_t_last_reduced)  # Calculate pairwise distances
    _, indices = torch.min(distances, dim=1)  # Find index of closest point
    
    # Step 2: Calculate weights from lnw_t_last
    weights = torch.exp(lnw_t_last).squeeze(1)
    
    # Step 3: Count occurrences in x_t_last for each point in data_t1
    count = torch.zeros_like(weights)
    for idx in indices:
        count[idx] += 1
    
    # Step 4: Calculate local mass loss
    relative_count = count / batch_size
    local_mass_loss = torch.norm(weights - relative_mass * relative_count, p=2)**2
    
    return local_mass_loss
