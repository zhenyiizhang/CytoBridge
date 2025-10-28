# --------------------------------------------------------------------------------------------------
# 1. Velocity computation: load trained model and compute latent‐space velocity
# --------------------------------------------------------------------------------------------------
from CytoBridge.utils import load_model_from_adata
import torch
import math
import numpy as np

def compute_velocity(adata, device='cuda'):
    """
    Calculate velocity based on the trained model 
    """
    model = load_model_from_adata(adata)
    model.to(device)
    # Calculate velocity and growth on the dataset for downstream analysis
    all_times = torch.tensor(adata.obs['time_point_processed']).unsqueeze(1).float().to(device)
    all_data = torch.tensor(adata.obsm['X_latent']).float().to(device)
    net_input = torch.cat([all_data, all_times], dim = 1)
    with torch.no_grad():
        velocity = model.velocity_net(net_input)
    adata.layers['velocity_latent'] = velocity.detach().cpu().numpy()

    return adata

# --------------------------------------------------------------------------------------------------
# 2. SDE core: Itô SDE with diagonal noise for latent state + log-weight
# --------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch
from typing import Tuple
import torch
from torch.nn import Module


class CytoSDE(Module):
    """
    SDE definition for CytoBridge models (Ito-type with diagonal noise).
    Computes drift (f) and diffusion (g) terms for latent state (z) and log-weight (lnw).
    
    Attributes
    ----------
    noise_type : str
        Type of noise ("diagonal" = independent noise per dimension).
    sde_type : str
        Type of SDE ("ito" = Ito calculus).
    model : torch.nn.Module
        Pre-trained CytoBridge model (must include velocity/score/growth networks as components).
    sigma : float
        Global diffusion coefficient scaling noise magnitude.
    """
    noise_type = "diagonal"  # Diagonal noise (independent across dimensions)
    sde_type = "ito"         # Ito-type SDE

    def __init__(self, model, sigma=1.0):
        super().__init__()
        self.model = model    # Trained CytoBridge model (contains velocity/score/growth nets)
        self.sigma = sigma    # Diffusion coefficient for noise scaling

    def f(self, t, y):
        """
        Compute drift term f(t, y) for the SDE.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (shape: [1]).
        y : tuple[torch.Tensor, torch.Tensor]
            Current state (z, lnw), where:
            - z: Latent state (shape: [batch_size, latent_dim])
            - lnw: Log-weight for particles (shape: [batch_size, 1])
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Drift terms for z and lnw (same shapes as input y).
        """
        z, lnw = y
        # Expand time to match batch size of z
        t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
        # Input to velocity network: latent features + time
        vel_in = torch.cat([z, t_expand], 1)
        
        # Compute drift for z (base velocity + score correction if available)
        drift_z = self.model.velocity_net(vel_in)
        if "score" in self.model.components:
            _, gradients = self.model.compute_score(t, z)  # Assume compute_score returns (score, gradients)
            drift_z += gradients
        
        # Compute drift for lnw (growth term if available, else zero)
        if "growth" in self.model.components:
            drift_lnw = self.model.growth_net(vel_in)
        else:
            drift_lnw = torch.zeros_like(lnw)
        
        return (drift_z, drift_lnw)

    def g(self, t, y):
        """
        Compute diffusion term g(t, y) for the SDE.
        Noise is applied to z (latent state) but not to lnw (log-weight).
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (shape: [1]).
        y : tuple[torch.Tensor, torch.Tensor]
            Current state (z, lnw) (shapes: [batch_size, latent_dim], [batch_size, 1]).
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Diffusion terms for z and lnw (same shapes as input y).
        """
        z, lnw = y
        return (torch.ones_like(z) * self.sigma,  # Diffusion for z
                torch.ones_like(lnw) * self.sigma * 0.0)  # No diffusion for lnw


# --------------------------------------------------------------------------------------------------
# 3. Euler–Maruyama integrator for Itô SDE
# --------------------------------------------------------------------------------------------------
import math
import torch


def euler_sdeint(sde, y0, dt, ts):
    """
    Numerical integration of SDE using Euler-Maruyama method (for Ito-type SDEs).
    
    Parameters
    ----------
    sde : CytoSDE
        SDE object with f (drift) and g (diffusion) methods.
    y0 : tuple[torch.Tensor, torch.Tensor]
        Initial state (z0, lnw0):
        - z0: Initial latent state (shape: [batch_size, latent_dim])
        - lnw0: Initial log-weight (shape: [batch_size, 1])
    dt : float
        Integration time step (fixed).
    ts : torch.Tensor
        Target time points to record (shape: [n_time_points], sorted in ascending order).
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - z_traj: Latent state trajectories (shape: [n_time_points, batch_size, latent_dim])
        - lnw_traj: Log-weight trajectories (shape: [n_time_points, batch_size, 1])
    """
    # Initialize state and current time
    y, t = y0, ts[0].item()
    out = []  # Store states at target time points
    ts_lst = ts.tolist()  # Convert target times to list for iteration
    idx = 0  # Index for target time points

    # Iterate until exceeding maximum target time
    while t <= ts_lst[-1] + 1e-8:
        # Record state if current time matches target time (within tolerance)
        if t >= ts_lst[idx] - 1e-8:
            out.append(y)
            idx += 1
            if idx >= len(ts_lst):
                break  # Exit if all target times are recorded
        
        # Convert current time to tensor (matches model input format)
        t_tensor = torch.tensor([t], dtype=torch.float32, device=y[0].device)
        
        # Compute drift and diffusion terms
        f_z, f_lnw = sde.f(t_tensor, y)
        g_z, g_lnw = sde.g(t_tensor, y)
        
        # Generate Gaussian noise (scaled by sqrt(dt) for Ito calculus)
        noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
        noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
        
        # Euler-Maruyama update: y_{t+dt} = y_t + f*dt + g*noise
        y = (y[0] + f_z * dt + g_z * noise_z,
             y[1] + f_lnw * dt + g_lnw * noise_lnw)
        
        # Step forward in time
        t += dt

    # Fill missing target times with last recorded state (due to numerical tolerance)
    while len(out) < len(ts_lst):
        out.append(out[-1])

    # Reshape output to [n_time_points, batch_size, ...]
    z_traj = torch.stack([o[0] for o in out])
    lnw_traj = torch.stack([o[1] for o in out])
    return z_traj, lnw_traj


# --------------------------------------------------------------------------------------------------
# 4. High-level wrapper: generate SDE trajectories and save
# --------------------------------------------------------------------------------------------------
def generate_sde_trajectories(
    adata,
    exp_dir: str,
    device: str | torch.device,
    sigma: float,
    n_time_steps: int,
    sample_traj_num: int,
    sample_cell_tra_consistent: bool,
    init_time: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    ----
    sde_traj : np.ndarray
         (n_steps, n_sample_cells, latent_dim)
    sde_point : np.ndarray
         (n_time_points, n_cells_used, latent_dim)
    w_point : np.ndarray
         (n_time_points, n_cells_used, 1) 
    """
    device = torch.device(device)
    os.makedirs(exp_dir, exist_ok=True)
    # ---------- Load Pre-trained Model ----------
    print("Loading model from adata...")
    model = load_model_from_adata(adata).to(device).float()  # Assume load_model_from_adata is predefined
    model.eval()
    print(f"Model components: {model.components}")

    time_key = "time_point_processed"
    all_times = sorted(adata.obs[time_key].unique())
    ts_points = torch.tensor(all_times, dtype=torch.float32, device=device)
    ts_continuous = torch.linspace(
        all_times[0], all_times[-1], n_time_steps,
        dtype=torch.float32, device=device
    )

    init_mask = adata.obs[time_key] == init_time
    if not init_mask.any():
        raise ValueError(f"No data found for init_time={init_time}")
    x0 = torch.tensor(adata[init_mask].obsm["X_latent"], dtype=torch.float32, device=device)
    x0 = x0.requires_grad_(True)
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
    initial_state = (x0, lnw0)

    # ---------- 4. SDE Integration ----------
    sde = CytoSDE(model, sigma).to(device)
    dt = (all_times[-1] - all_times[0]) / (n_time_steps - 1)
    sde_traj, traj_lnw = euler_sdeint(sde, initial_state, dt, ts_continuous)
    sde_point, point_lnw = euler_sdeint(sde, initial_state, dt, ts_points)

    # ---------- 5. Sampling / Consistency Filtering ----------
    sample_traj_num = min(sample_traj_num, x0.shape[0])
    traj_cell_indices = torch.randperm(x0.shape[0], device=device)[:sample_traj_num]


    sde_traj = sde_traj[:, traj_cell_indices]
    traj_lnw = traj_lnw[:, traj_cell_indices]
    print(f"sample_cell_tra_consistent=False: "
            f"Using {sample_traj_num} cells for trajectories, all cells for predictions")

    # ---------- 6. Convert to numpy / Normalize weights ----------
    sde_traj = sde_traj.detach().cpu().numpy()
    sde_point = sde_point.detach().cpu().numpy()
    w_point = torch.exp(point_lnw).detach().cpu().numpy()
    w_point /= w_point.sum(axis=1, keepdims=True)

    # ---------- 7. Save ----------
    np.save(os.path.join(exp_dir, "sde_trajec.npy"), sde_traj)
    np.save(os.path.join(exp_dir, "sde_point.npy"), sde_point)
    np.save(os.path.join(exp_dir, "sde_weight.npy"), w_point)

    return sde_traj, sde_point, w_point


# --------------------------------------------------------------------------------------------------
# 5. Euler integrator with splitting (branching & extinction) for ODE
# --------------------------------------------------------------------------------------------------
import torch
import math
import numpy as np
from torchdiffeq import odeint

def euler_odeint_split(ode_func, initial_state, ts, noise_std=0.01):
    device = initial_state[0].device
    t0, tf = ts[0].item(), ts[-1].item()
    current_t = t0
    current_z, current_lnw, current_m = initial_state
    w_prev = torch.exp(current_lnw)  # initial weight
    
    # keep initial shapes for later reference
    initial_size = current_z.shape[0]
    feature_size = current_z.shape[1]  # feature dimension

    ts_list = ts.tolist()
    # -------------------------- key modification 1: store all time-step history (for back-tracking copy) --------------------------
    # history format: [time_step0_data, time_step1_data, ...], each element is (z, lnw)
    history_z = [current_z.clone()]  # particle feature history at initial step (t0)
    history_lnw = [current_lnw.clone()]  # log-weight history at initial step (t0)
    idx_next = 1  # start iteration from step 1 (t0 already stored in history)

    while current_t <= tf + 1e-8 and idx_next < len(ts_list):
        t_tensor = torch.tensor([current_t], device=device)
        f_z, f_lnw, f_m = ode_func(t_tensor, (current_z, current_lnw, current_m))

        # 1. Euler update for current step
        dt_step = ts_list[idx_next] - current_t
        new_z   = current_z   + f_z   * dt_step
        new_lnw = current_lnw + f_lnw * dt_step
        new_m   = current_m   + f_m   * dt_step
        current_t = ts_list[idx_next]

        w_next = torch.exp(new_lnw)
        ratio = w_next / w_prev
        current_size = new_z.shape[0]  # current particle number

        # -------------------------- 2. Splitting: add “back-track copy history” logic --------------------------
        mask_split = (ratio >= 1).squeeze(-1)
        # data after split: contains “original split particles + new split particles”
        split_z = new_z[mask_split].clone()  # keep original split particles (not dropped)
        split_lnw = new_lnw[mask_split].clone()
        split_m = new_m[mask_split].clone()
        # record original split particle indices (for later back-track copy history)
        original_split_indices = torch.where(mask_split)[0]
        
        if mask_split.any():
            r_split = ratio[mask_split].squeeze(-1)
            z_split_original = new_z[mask_split]  # original split particles (to be split)
            lnw_split_original = new_lnw[mask_split]
            m_split_original = new_m[mask_split]

            # compute split number for each original split particle (same as original logic)
            floor_r = torch.floor(r_split)
            frac_r = r_split - floor_r
            rand_f = torch.rand_like(frac_r)
            mj = floor_r.int() + (rand_f < frac_r).int()  # split mj new particles from each original particle
            valid = mj > 0

            if valid.any():
                # filter valid original particle indices
                valid_mask = valid
                valid_mj = mj[valid_mask]  # valid split numbers
                valid_original_indices = original_split_indices[valid_mask]  # global indices of valid original split particles
                valid_z = z_split_original[valid_mask]
                valid_lnw = lnw_split_original[valid_mask]
                valid_m = m_split_original[valid_mask]

                # generate new split particles (current step data)
                new_split_z_list = []
                new_split_lnw_list = []
                for i in range(len(valid_mj)):
                    mj_i = valid_mj[i].item()  # the i-th original particle splits mj_i new particles
                    original_idx_i = valid_original_indices[i].item()  # global index of the i-th original particle

                    # a. generate current-step new split particle data (with noise)
                    rep_z = valid_z[i].unsqueeze(0).repeat(mj_i, 1)  # replicate mj_i times
                    rep_lnw = valid_lnw[i].unsqueeze(0).repeat(mj_i, 1)
                    noise = torch.normal(0, noise_std, size=rep_z.shape, device=device)
                    new_z_i = rep_z + noise
                    new_lnw_i = rep_lnw

                    # b. key: back-track copy original particle history (fill new particles' previous time steps)
                    # traverse all historical time steps (from t0 to current step minus 1), copy original particle history
                    for hist_idx in range(len(history_z)):
                        # original particle data at historical time step
                        hist_z_original = history_z[hist_idx][original_idx_i].unsqueeze(0)
                        hist_lnw_original = history_lnw[hist_idx][original_idx_i].unsqueeze(0)
                        # replicate to new particles' history (each new particle has same history as original)
                        history_z[hist_idx] = torch.cat([history_z[hist_idx], hist_z_original.repeat(mj_i, 1)], dim=0)
                        history_lnw[hist_idx] = torch.cat([history_lnw[hist_idx], hist_lnw_original.repeat(mj_i, 1)], dim=0)

                    # c. collect current-step new split particle data
                    new_split_z_list.append(new_z_i)
                    new_split_lnw_list.append(new_lnw_i)

                # merge all new split particles' current-step data
                if new_split_z_list:
                    new_split_z = torch.cat(new_split_z_list, dim=0)
                    new_split_lnw = torch.cat(new_split_lnw_list, dim=0)
                    new_split_m = valid_m.unsqueeze(0).repeat_interleave(valid_mj, dim=1).squeeze(0)
                    # add new split particles to current step split result (original + new)
                    split_z = torch.cat([split_z, new_split_z], dim=0)
                    split_lnw = torch.cat([split_lnw, new_split_lnw], dim=0)
                    split_m = torch.cat([split_m, new_split_m], dim=0)

        # -------------------------- 3. Extinction: change to “set NaN instead of drop” --------------------------
        mask_extinct = ~mask_split
        # keep all extinct particle indices, set dead particles to NaN (not dropped, ensure consistent indexing)
        extinct_z = new_z[mask_extinct].clone()
        extinct_lnw = new_lnw[mask_extinct].clone()
        extinct_m = new_m[mask_extinct].clone()
        
        if mask_extinct.any():
            r_ext = ratio[mask_extinct].squeeze(-1)
            # generate survival mask: True=survive, False=dead (set to NaN)
            keep_mask = torch.rand_like(r_ext) < r_ext
            # set dead particles to NaN (keep index, no further update)
            extinct_z[~keep_mask] = torch.nan
            extinct_lnw[~keep_mask] = torch.nan
            extinct_m[~keep_mask] = torch.nan

        # -------------------------- 4. Merge: keep all particle indices (split + extinct) --------------------------
        current_z = torch.cat([split_z, extinct_z], dim=0)
        current_lnw = torch.cat([split_lnw, extinct_lnw], dim=0)
        current_m = torch.cat([split_m, extinct_m], dim=0)
        w_prev = torch.exp(current_lnw)  # update weight

        # -------------------------- 5. save current step data to history (for later split back-track) --------------------------
        history_z.append(current_z.clone())
        history_lnw.append(current_lnw.clone())
        idx_next += 1

    # -------------------------- 6. unify shapes for all time steps (history already complete, simply stack) --------------------------
    # history already contains all particles (original + split) with complete time series, stack directly
    out_z = torch.stack(history_z, dim=0)  # shape: (time_steps, num_particles, feature_size)
    out_lnw = torch.stack(history_lnw, dim=0)  # shape: (time_steps, num_particles, 1)
    
    return out_z, out_lnw


# --------------------------------------------------------------------------------------------------
# 6. High-level wrapper: generate ODE trajectories with optional splitting
# --------------------------------------------------------------------------------------------------
from typing import List
import os, time
from ..tl.methods import ODEFunc


def generate_ode_trajectories(
        model,
        adata,
        n_trajectories: int = 50,
        n_bins: int = 100,
        device: str = "cuda",
        samples_key: str = 'time_point_processed',
        exp_dir: str = None,          
        split_true: str = False
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:

    device = torch.device(device)
    model.to(device).eval()

    X_raw = adata.obsm["X_latent"]

    unique_times = np.sort(adata.obs[samples_key].unique())
    X = [X_raw[adata.obs[samples_key] == t] for t in unique_times]

    init_time = unique_times[0]
    init_mask = adata.obs[samples_key] == init_time
    init_idx = np.where(init_mask)[0]
    replace = len(init_idx) < n_trajectories

    chosen_init_idx = np.random.choice(init_idx, size=n_trajectories, replace=replace)
    init_x = torch.tensor(X_raw[chosen_init_idx], dtype=torch.float32, device=device)

    from ..tl.methods import ODEFunc
    ode_func = ODEFunc(model, use_mass=True, score_use=False).to(device)

    t_min, t_max = float(unique_times[0]), float(unique_times[-1])
    t_bins  = torch.linspace(t_min, t_max, n_bins, device=device)
    t_point = torch.tensor(unique_times, dtype=torch.float32, device=device)

    init_lnw = torch.zeros(n_trajectories, 1, dtype=torch.float32, device=device, requires_grad=True)
    init_m   = torch.zeros_like(init_lnw)
    initial_state = (init_x, init_lnw, init_m)
    if split_true :
        traj_x, traj_lnw ,_ = euler_odeint_split(ode_func, initial_state, t_bins)
        point_x, point_lnw,_  = euler_odeint_split(ode_func, initial_state, t_point)
    else :
        traj_x, traj_lnw, _ = odeint(ode_func, initial_state, t_bins,  method='euler')
        point_x, point_lnw, _ = odeint(ode_func, initial_state, t_point, method='euler')

    traj_array  = traj_x.detach().cpu().numpy()          # (T_bins, M, D)
    traj_lnw    = traj_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)
    point_array = point_x.detach().cpu().numpy()         # (T, M, D)
    point_lnw   = point_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)
    # save
    if exp_dir is not None:
        import os
        os.makedirs(exp_dir, exist_ok=True)
        np.save(os.path.join(exp_dir, "ode_traj.npy"), traj_array)
        np.save(os.path.join(exp_dir, "ode_traj_lnw.npy"), traj_lnw)
        np.save(os.path.join(exp_dir, "ode_point.npy"),  point_array)
        np.save(os.path.join(exp_dir, "ode_point_lnw.npy"), point_lnw)
        print(f"[generate_ode_trajectories] trajectories saved to {exp_dir}")

    return point_array, traj_array


# --------------------------------------------------------------------------------------------------
# 7. Stand-alone utility: simulate single SDE trajectory
# --------------------------------------------------------------------------------------------------
def simulate_trajectory(model, x0, sigma, time, dt, device):
    x0 = x0.requires_grad_(True)
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device, dtype=torch.float32) / x0.shape[0])
    initial_state = (x0, lnw0)

    class CytoSDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"
        def __init__(self, model, sigma=1.0):
            super().__init__()
            self.model = model
            self.sigma = sigma
        def f(self, t, y):
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            vel_in = torch.cat([z, t_expand], 1)
            drift_z = self.model.velocity_net(vel_in)
            if "score" in self.model.components:
                score, gradients = self.model.compute_score(t, z)
                drift_z += gradients
            if "growth" in self.model.components:
                drift_lnw = self.model.growth_net(vel_in)
            else:
                drift_lnw = torch.zeros_like(lnw)
            return (drift_z, drift_lnw)
        def g(self, t, y):
            z, lnw = y
            return (torch.ones_like(z) * self.sigma,
                    torch.ones_like(lnw) * 0.0)

    sde = CytoSDE(model, sigma).to(device)

    def euler_sdeint(sde, y0, dt, ts_lst):
        y, t = y0, ts_lst[0]
        out = []
        idx = 0
        while t <= ts_lst[-1] + 1e-8:
            if t >= ts_lst[idx] - 1e-8:
                out.append(y)
                idx += 1
                if idx >= len(ts_lst):
                    break
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            f_z, f_lnw = sde.f(t_tensor, y)
            g_z, g_lnw = sde.g(t_tensor, y)
            noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
            noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
            y = (y[0] + f_z * dt + g_z * noise_z,
                 y[1] + f_lnw * dt + g_lnw * noise_lnw)
            t += dt
        while len(out) < len(ts_lst):
            out.append(out[-1])
        return torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out])

    sde_point, point_lnw = euler_sdeint(sde, initial_state, dt, time)

    sde_point = sde_point.detach().cpu().numpy()
    w_point = np.exp(point_lnw.detach().cpu().numpy())

    return sde_point, w_point