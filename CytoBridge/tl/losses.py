import torch
import ot
from geomloss import SamplesLoss
from CytoBridge.tl.models import DynamicalModel


def calc_ot_loss(x1, x2, lnw1, method='emd_detach', return_pi=False):
    '''
    Calculate the Optimal Transport (OT) loss between two sets of data.
    
    Args:
        x1: Generated data (tensor), shape (num_generated, num_features).
        x2: Subset of reference data (tensor), shape (num_reference, num_features).
        lnw1: Log-weights of the generated data (tensor), shape (num_generated, 1).
        method: Method to compute the OT loss:
            'emd_detach': Earth Mover's Distance (EMD) with detached weights (no gradient flow through weights).
            'sinkhorn': Sinkhorn algorithm with gradients preserved for weights and data.
            'sinkhorn_detach': Sinkhorn algorithm with detached weights (no gradient through weights).
            'sinkhorn_knopp_unbalanced': Unbalanced Sinkhorn-Knopp algorithm with gradients.
            'sinkhorn_knopp_unbalanced_detach': Unbalanced Sinkhorn-Knopp with detached weights.
        return_pi: If True, returns both the OT loss and the transport matrix (pi); otherwise returns only the loss.
    
    Returns:
        If return_pi is True: Tuple (loss_ot, pi), where loss_ot is the OT loss and pi is the transport matrix.
        Otherwise: Scalar tensor representing the OT loss.
    '''
    m1 = torch.exp(lnw1)
    m1 = m1 / m1.sum()  # Normalize generated weights to form a probability distribution
    m2 = torch.ones(x2.shape[0], 1).float().to(x1.device)
    m2 = m2 / m2.sum()  # Uniform distribution over reference data
    m1 = m1.squeeze(1)  # Remove singleton dimension
    m2 = m2.squeeze(1)
    pi = None  # Initialize transport matrix

    if method == 'emd_detach':
        # Squared Euclidean cost matrix
        M = torch.cdist(x1, x2) **2
        if return_pi:  # Compute and return transport matrix if required
            pi_np = ot.emd(
                m1.detach().cpu().numpy(),  # Detach weights to block gradients
                m2.detach().cpu().numpy(),
                M.detach().cpu().numpy()
            )
            pi = torch.tensor(pi_np).float().to(x1.device)
            loss_ot = torch.sum(pi * M)
        else:  # Compute loss without storing the full transport matrix
            pi_np = ot.emd(
                m1.detach().cpu().numpy(),
                m2.detach().cpu().numpy(),
                M.detach().cpu().numpy()
            )
            loss_ot = torch.sum(torch.tensor(pi_np).float().to(x1.device) * M)

    elif method == 'sinkhorn_detach':
        M = torch.cdist(x1, x2)** 2
        if return_pi:
            pi_np = ot.sinkhorn(
                m1.detach().cpu().numpy(),
                m2.detach().cpu().numpy(),
                M.detach().cpu().numpy(),
                reg=0.1  # Regularization strength (matches 'blur' in geomloss)
            )
            pi = torch.tensor(pi_np).float().to(x1.device)
            loss_ot = torch.sum(pi * M)
        else:
            # Use geomloss for efficient Sinkhorn loss computation
            loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=0.1)
            loss_ot = loss_fn(m1.detach(), x1, m2.detach(), x2)

    elif method == 'sinkhorn':
        if return_pi:
            M = torch.cdist(x1, x2) **2
            pi_np = ot.sinkhorn(
                m1.cpu().numpy(),  # Preserve gradients through weights
                m2.cpu().numpy(),
                M.cpu().numpy(),
                reg=0.1
            )
            pi = torch.tensor(pi_np).float().to(x1.device)
            loss_ot = torch.sum(pi * M)
        else:
            # Directly compute Sinkhorn loss with geomloss (preserves gradients)
            loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=0.1)
            loss_ot = loss_fn(m1, x1, m2, x2)

    elif method == 'sinkhorn_knopp_unbalanced_detach':
        M = torch.cdist(x1, x2, p=2)** 2
        if return_pi:
            pi_np = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1.detach().cpu().numpy(),
                b=m2.detach().cpu().numpy(),
                M=M.detach().cpu().numpy(),
                reg=0.05,  # Entropy regularization
                reg_m=0.01  # Unbalanced mass regularization
            )
            pi = torch.tensor(pi_np, dtype=torch.float32, device=x1.device)
            loss_ot = torch.sum(pi * M)
        else:
            # Compute loss without storing pi
            pi_np = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1.detach().cpu().numpy(),
                b=m2.detach().cpu().numpy(),
                M=M.detach().cpu().numpy(),
                reg=0.05,
                reg_m=0.01
            )
            loss_ot = torch.sum(torch.tensor(pi_np).float().to(x1.device) * M)

    elif method == 'sinkhorn_knopp_unbalanced':
        M = torch.cdist(x1, x2, p=2) **2
        if return_pi:
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1, b=m2, M=M, reg=0.05, reg_m=0.1
            )
            loss_ot = torch.sum(pi * M)
        else:
            # Compute loss and discard pi to save memory
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1, b=m2, M=M, reg=0.05, reg_m=0.1
            )
            loss_ot = torch.sum(pi * M)
            pi = None  # Explicitly release pi

    else:
        raise ValueError(f'Unknown method (OT_Loss): {method}')

    if return_pi:
        return loss_ot, pi
    else:
        return loss_ot


def calc_mass_loss(x1, x2, lnw1, relative_mass, global_mass = False):
    '''
    Calculate the mass loss between generated data and a reference subset, ensuring mass conservation.
    
    Args:
        x1: Generated data (tensor), shape (num_generated, num_features).
        x2: Reference subset data (tensor), shape (num_reference, num_features).
        lnw1: Log-weights of the generated data (tensor), shape (num_generated, 1).
        relative_mass: Relative mass of the reference subset (scalar), i.e., its proportion of the total mass.
        global_mass: If True, include a global mass conservation term; otherwise, use only local mass matching.
    
    Returns:
        Scalar tensor representing the total mass loss.
    '''
    m1 = torch.exp(lnw1)  # Convert log-weights to actual weights
    # Compute local mass matching loss (alignment between generated and reference masses)
    local_mass_loss = calc_mass_matching_loss(x1, x2, lnw1, relative_mass)
    if global_mass:
        # Add global term: penalize deviation of total generated mass from relative_mass
        global_mass_loss = torch.norm(torch.sum(m1) - relative_mass, p=2)** 2
        mass_loss = global_mass_loss + local_mass_loss
    else:
        mass_loss = local_mass_loss
    return mass_loss


def calc_mass_matching_loss(x_t_last, data_t1, lnw_t_last, relative_mass, dim_reducer=None):
    """
    Calculate the local mass matching loss, optionally using dimensionality-reduced data for efficiency.
    
    Parameters:
        x_t_last: Generated data at the final time step (tensor/array), shape (m_samples, n_features).
        data_t1: Reference data at the target time step (tensor/array), shape (n_samples, n_features).
        lnw_t_last: Log-weights of the generated data (tensor), shape (m_samples, 1).
        relative_mass: Relative mass of the reference subset (scalar).
        dim_reducer: Optional callable for dimensionality reduction (e.g., PCA). 
            If provided, distances are computed on reduced data; otherwise, raw data is used.
    
    Returns:
        Scalar tensor representing the local mass matching loss.
    """
    # Step 0: Apply dimensionality reduction if specified
    if dim_reducer is not None:
        # Reduce reference and generated data to lower dimensions
        data_t1_reduced = dim_reducer(data_t1.detach().cpu().numpy())
        x_t_last_reduced = dim_reducer(x_t_last.detach().cpu().numpy())
    else:
        data_t1_reduced = data_t1
        x_t_last_reduced = x_t_last
    
    batch_size = data_t1.shape[0]  # Number of reference samples
    
    # Ensure data is in PyTorch tensor format
    data_t1_reduced = torch.as_tensor(data_t1_reduced)
    x_t_last_reduced = torch.as_tensor(x_t_last_reduced)
    
    # Step 1: Find the closest generated sample for each reference sample
    distances = torch.cdist(data_t1_reduced, x_t_last_reduced)  # Pairwise distances (n_samples x m_samples)
    _, indices = torch.min(distances, dim=1)  # Indices of closest generated samples
    
    # Step 2: Convert log-weights to actual weights
    weights = torch.exp(lnw_t_last).squeeze(1)                                               
    
    # Step 3: Count how many reference samples map to each generated sample
    count = torch.zeros_like(weights)
    for idx in indices:
        count[idx] += 1                                                                        
     
    # Step 4: Compute loss as L2 norm between generated weights and scaled reference counts
    relative_count = count / batch_size  # Normalize counts to proportions
    local_mass_loss = torch.norm(weights - relative_mass * relative_count, p=2)**2              
    
    return local_mass_loss

def calc_score_matching_loss(
    model: DynamicalModel,
    t: torch.Tensor,
    x: torch.Tensor,
    sigma: float = 0.1,
    lambda_penalty: float = 1.0
) -> torch.Tensor:
    """
    Calculate the score matching loss for training a dynamical model, including a denoising term and penalty.
    
    Args:
        model: Instance of DynamicalModel with a score network (score_net).
        t: Time points (tensor), shape (batch_size,) or (batch_size, 1).
        x: Data samples (tensor), shape (batch_size, latent_dim).
        sigma: Standard deviation of the noise added to data for denoising loss.
        lambda_penalty: Coefficient for the positivity penalty on the score network output.
    
    Returns:
        Scalar tensor representing the total score matching loss.
    """
    # Expand time tensor to match data batch size
    if t.dim() == 1:
        t = t.unsqueeze(1)
    t_expanded = t.expand(x.size(0), 1)
    # Concatenate data and time for input to the score network
    net_input = torch.cat([x, t_expanded], dim=1)

    # 1. Compute base score (not used directly in loss but context for network behavior)
    score = model.score_net.compute_gradient(net_input)  # [batch_size, latent_dim]

    # 2. Add noise to data and compute score on noisy data
    noise = torch.randn_like(x)  # Gaussian noise
    x_noisy = x + sigma * noise  # Noisy version of input data
    net_input_noisy = torch.cat([x_noisy, t_expanded], dim=1)
    score_noisy = model.score_net.compute_gradient(net_input_noisy)

    # 3. Denoising loss: penalize mismatch between noisy score and denoising target
    denoising_loss = torch.mean(
        torch.norm(score_noisy + (x_noisy - x) / (sigma **2), dim=1)** 2
    )

    # 4. Positivity penalty: discourage negative outputs from the score network
    positive_st = torch.relu(model.score_net(net_input))  # ReLU to isolate positive components
    penalty = lambda_penalty * torch.max(positive_st)  # Penalty on maximum positive value

    return denoising_loss + penalty