import torch
import ot
from geomloss import SamplesLoss
from CytoBridge.tl.models import DynamicalModel
import torch.nn as nn
from CytoBridge.utils.utils import compute_integral

def calc_ot_loss(x1, x2, lnw1, method='emd_detach', return_pi=False):
    '''
    Calculate the OT loss between two sets of data.
    x1: the generated data
    x2: the subset of data
    lnw1: the log weights of the generated data
    method: the method to calculate the OT loss
        'emd_detach': use EMD with detached weights
        'sinkhorn': use Sinkhorn with gradients
        'sinkhorn_detach': use Sinkhorn with detached weights
        'sinkhorn_knopp_unbalanced': unbalanced Sinkhorn with gradients
        'sinkhorn_knopp_unbalanced_detach': unbalanced Sinkhorn with detached weights
    return_pi: if True, compute and return (loss_ot, pi); else return loss_ot only
    '''
    m1 = torch.exp(lnw1)
    m1 = m1 / m1.sum()
    m2 = torch.ones(x2.shape[0], 1).float().to(x1.device)
    m2 = m2 / m2.sum()
    m1 = m1.squeeze(1)
    m2 = m2.squeeze(1)
    pi = None  
    if method == 'emd_detach':
        M = torch.cdist(x1, x2) ** 2
        if return_pi: 
            pi_np = ot.emd(
                m1.detach().cpu().numpy(),
                m2.detach().cpu().numpy(),
                M.detach().cpu().numpy()
            )
            pi = torch.tensor(pi_np).float().to(x1.device)
            loss_ot = torch.sum(pi * M)
        else:  
            pi_np = ot.emd(
                m1.detach().cpu().numpy(),
                m2.detach().cpu().numpy(),
                M.detach().cpu().numpy()
            )
            loss_ot = torch.sum(torch.tensor(pi_np).float().to(x1.device) * M)

    elif method == 'sinkhorn_detach':
        M = torch.cdist(x1, x2) ** 2
        if return_pi:
            pi_np = ot.sinkhorn(
                m1.detach().cpu().numpy(),
                m2.detach().cpu().numpy(),
                M.detach().cpu().numpy(),
                reg=0.1  
            )
            pi = torch.tensor(pi_np).float().to(x1.device)
            loss_ot = torch.sum(pi * M)
        else:
            loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=0.1)
            loss_ot = loss_fn(m1.detach(), x1, m2.detach(), x2)

    elif method == 'sinkhorn':
        if return_pi:
            M = torch.cdist(x1, x2) ** 2
            pi_np = ot.sinkhorn(
                m1.cpu().numpy(),
                m2.cpu().numpy(),
                M.cpu().numpy(),
                reg=0.1
            )
            pi = torch.tensor(pi_np).float().to(x1.device)
            loss_ot = torch.sum(pi * M)
        else:
            loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=0.1)
            loss_ot = loss_fn(m1, x1, m2, x2)

    elif method == 'sinkhorn_knopp_unbalanced_detach':
        M = torch.cdist(x1, x2, p=2) ** 2
        if return_pi:
            pi_np = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1.detach().cpu().numpy(),
                b=m2.detach().cpu().numpy(),
                M=M.detach().cpu().numpy(),
                reg=0.05,
                reg_m=0.01
            )
            pi = torch.tensor(pi_np, dtype=torch.float32, device=x1.device)
            loss_ot = torch.sum(pi * M)
        else:
            pi_np = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1.detach().cpu().numpy(),
                b=m2.detach().cpu().numpy(),
                M=M.detach().cpu().numpy(),
                reg=0.05,
                reg_m=0.01
            )
            loss_ot = torch.sum(torch.tensor(pi_np).float().to(x1.device) * M)

    elif method == 'sinkhorn_knopp_unbalanced':
        M = torch.cdist(x1, x2, p=2) ** 2
        if return_pi:
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1, b=m2, M=M, reg=0.05, reg_m=0.1
            )
            loss_ot = torch.sum(pi * M)
        else:
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a=m1, b=m2, M=M, reg=0.05, reg_m=0.1
            )
            loss_ot = torch.sum(pi * M)
            pi = None  

    else:
        raise ValueError(f'Unknown method (OT_Loss): {method}')

    
    if return_pi:
        return loss_ot, pi
    else:
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


def calc_mass_matching_loss(x_t_last, data_t1, lnw_t_last, relative_mass, dim_reducer=None):
    """
    Calculate mass matching loss, optionally on dimensionality-reduced data.

    Parameters:
    - x_t_last: tensor or array, shape (m_samples, n_features)
    - data_t1: tensor or array, shape (n_samples, n_features)
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
    distances = torch.cdist(data_t1_reduced, x_t_last_reduced)  # Calculate pairwise distances  (n,m)
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
import torch
import torch.nn as nn

def calc_score_matching_loss(
    model: DynamicalModel,
    t: torch.Tensor,
    x: torch.Tensor,
    sigma: float = 0.1,
    lambda_penalty: float = 1.0
) -> torch.Tensor:
    # Concatenate time and state (consistent with model input)
    if t.dim() == 1:
        t = t.unsqueeze(1)
    t_expanded = t.expand(x.size(0), 1)
    net_input = torch.cat([x, t_expanded], dim=1)

    # 1. Compute score function (via gradient of log_prob, consistent with DeepRUOT)
    score = model.score_net.compute_gradient(net_input)  # [batch_size, latent_dim]

    # 2. Compute score for noisy data
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    net_input_noisy = torch.cat([x_noisy, t_expanded], dim=1)
    score_noisy = model.score_net.compute_gradient(net_input_noisy)

    # 3. Denoising score matching loss (consistent with DeepRUOT's score_loss)
    denoising_loss = torch.mean(
        torch.norm(score_noisy + (x_noisy - x) / (sigma **2), dim=1)** 2
    )

    # 4. Penalty term (optional, maintain consistency with original logic)
    positive_st = torch.relu(model.score_net(net_input))  # Penalty for positive log_prob
    penalty = lambda_penalty * torch.max(positive_st)

    return denoising_loss + penalty

class Density_loss(nn.Module):
    def __init__(self, hinge_value=0.01):
        self.hinge_value = hinge_value
        pass

    def __call__(self, source, target, groups = None, to_ignore = None, top_k = 5):
        if groups is not None:
            c_dist = torch.stack([
                torch.cdist(source[i], target[i]) 
                for i in range(1,len(groups))
                if groups[i] != to_ignore
            ])
        else:
            c_dist = torch.stack([
                torch.cdist(source, target)                 
            ])
        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values -= self.hinge_value
        values[values<0] = 0
        loss = torch.mean(values)
        return loss

def calc_pinn_loss(
    self, 
    t: torch.Tensor, 
    x: torch.Tensor, 
    sigma: float, 
    use_mass: bool, 
    trace_df_dz,  # Assume the divergence calculation utility function already exists
    device
) -> torch.Tensor:
    """
    """
    P1=0
    t = torch.tensor(t, device=x.device, dtype=torch.float32)  # Convert dtype of t
    batch_size = x.shape[0]
    t_expand = t.expand(batch_size, 1)  # Ensure time dimension matches
    t_expand.requires_grad_(True)

    outputs = self.model(t_expand, x, lnw=torch.zeros(batch_size, 1, device=x.device),except_interaction = False)
    vv = outputs['velocity']  # Velocity field [batch_size, latent_dim]
    gg = outputs['growth']    
    ss = outputs['score']
    rrho = torch.exp(ss * 2 / (sigma ** 2))  # Density estimation [batch_size, 1]
    rrho_t = torch.autograd.grad(
        outputs=rrho,
        inputs=t_expand,
        grad_outputs=torch.ones_like(rrho),
        create_graph=True  # Retain computation graph for backpropagation
    )[0]

    # 4. Compute spatial divergence of the product of velocity field and density ∇·(v·ρ)
    vv_rho = vv * rrho  # [batch_size, latent_dim]
    ddiv_v_rho = trace_df_dz(vv_rho, x).unsqueeze(1)  # [batch_size, 1]
    
    # 5. Handle interaction term (if model includes it)

    if self.use_interaction and 'interaction' in outputs:
        interaction_estimate = compute_integral(
            x, t_expand, self,self.interaction_net,self.score_net, sigma=sigma
        )  # [batch_size, latent_dim]
        rho_interaction_estimate = interaction_estimate * rrho  # [batch_size, latent_dim]
        div_interaction = trace_df_dz(rho_interaction_estimate, x).unsqueeze(1)  # [batch_size, 1]
    
    # 6. Construct continuity equation residual (physical constraint)
    if self.use_interaction and 'interaction' in outputs:
        if use_mass:
            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho - gg * rrho + div_interaction)
        else:
            # Without mass term: ∂ρ/∂t + ∇·(v·ρ) + ∇·(interaction·ρ) = 0
            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho + div_interaction)
    else:
        if use_mass:
            # With mass term, without interaction: ∂ρ/∂t + ∇·(v·ρ) - g·ρ = 0
            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho - gg * rrho)
        else:
            # Without mass term and interaction: ∂ρ/∂t + ∇·(v·ρ) = 0
            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho)
    
    # Return average residual (absolute value ensures non-negativity)
    return torch.mean(ppinn_loss)