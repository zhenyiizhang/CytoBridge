from CytoBridge.utils import load_model_from_adata
import torch
import math
import numpy as np
from torchdiffeq import odeint

def compute_velocity(adata, device='cuda'):
    '''
    Calculate velocity based on the trained model 
    '''
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
def compute_growth(adata, device='cuda'):
    '''
    Calculate velocity based on the trained model 
    '''
    model = load_model_from_adata(adata)
    model.to(device)
    # Calculate velocity and growth on the dataset for downstream analysis
    all_times = torch.tensor(adata.obs['time_point_processed']).unsqueeze(1).float().to(device)
    all_data = torch.tensor(adata.obsm['X_latent']).float().to(device)
    net_input = torch.cat([all_data, all_times], dim = 1)
    with torch.no_grad():
        growth = model.growth_net(net_input)
    adata.obsm['growth_rate'] = growth.detach().cpu().numpy()

    return adata
#%%
def euler_odeint_split(ode_func, initial_state, ts, noise_std=0.01):
    """
    Integrate a cell-population ODE with discrete cell division and death events.
    The solver uses an Euler scheme and keeps track of lineage relationships,
    split times, alive status and death steps for every trajectory.

    Parameters
    ----------
    ode_func : callable
        Right-hand side of the ODE: f(t, (z, lnw, m)) -> (dz/dt, d(lnw)/dt, dm/dt).
    initial_state : tuple of torch.Tensor
        (z, lnw, m) at t = ts[0].
    ts : torch.Tensor
        Strictly increasing time points at which the solution is returned.
    noise_std : float, optional
        Gaussian noise added to daughter-cell latent states after division.

    Returns
    -------
    z_hist : torch.Tensor
        Stacked latent states over time, shape (len(ts), n_trajectories, dim_z).
        Dead trajectories are padded with NaN.
    lnw_hist : torch.Tensor
        Stacked log-weights over time, shape (len(ts), n_trajectories, 1).
    """
    device = initial_state[0].device
    t0, tf = ts[0].item(), ts[-1].item()
    current_t = t0
    current_z, current_lnw, current_m = initial_state
    w_prev = torch.exp(current_lnw)  # previous weights for ratio computation

    # Lineage bookkeeping
    replication_parents = torch.zeros(current_z.shape[0], 1, dtype=torch.long, device=device)
    replication_parents[:] = -1  # -1: no parent
    split_times = torch.full((current_z.shape[0], 1), -1.0, device=device)
    alive_mask = torch.ones(current_z.shape[0], 1, dtype=torch.bool, device=device)
    death_step = torch.full((current_z.shape[0], 1), -1, dtype=torch.long, device=device)  # -1: alive

    feature_size = current_z.shape[1]
    m_size = current_m.shape[1]
    ts_list = ts.tolist()
    n_bins = len(ts_list)

    # History lists: one entry per output time step
    history_z = [current_z.clone()]
    history_lnw = [current_lnw.clone()]
    history_m = [current_m.clone()]
    history_parents = [replication_parents.clone()]
    history_split_times = [split_times.clone()]
    history_alive = [alive_mask.clone()]
    history_death_step = [death_step.clone()]

    idx_next = 1
    check_frequency = 1
    step_counter = 0

    while current_t <= tf + 1e-8 and idx_next < n_bins:
        t_tensor = torch.tensor([current_t], device=device)
        f_z, f_lnw, f_m = ode_func(t_tensor, (current_z, current_lnw, current_m))

        dt_step = ts_list[idx_next] - current_t
        new_z = current_z + f_z * dt_step
        new_lnw = current_lnw + f_lnw * dt_step
        new_m = current_m + f_m * dt_step
        next_t = ts_list[idx_next]

        w_next = torch.exp(new_lnw)
        ratio = w_next / w_prev  # weight ratio for split/extinction decisions
        current_size = new_z.shape[0]

        # Division and extinction logic
        if step_counter % check_frequency == 0:
            # 1. Cell division: only alive cells can divide
            mask_split = (ratio >= 1).squeeze(-1) & alive_mask.squeeze(-1)
            split_count = mask_split.sum().item()
            total_new_cells = 0

            if split_count > 0:
                r_split = ratio[mask_split].squeeze(-1)
                floor_r = torch.floor(r_split)
                frac_r = r_split - floor_r
                rand_f = torch.rand_like(frac_r)
                mj = floor_r.int() + (rand_f < frac_r).int()
                mj_new = torch.max(torch.zeros_like(mj), mj - 1)
                total_new_cells = mj_new.sum().item()

                if total_new_cells > 0:
                    new_z_list, new_lnw_list, new_m_list = [], [], []
                    new_parents_list, new_split_times_list = [], []
                    new_death_step_list = []

                    for i in range(split_count):
                        if mj_new[i] == 0:
                            continue
                        parent_idx = torch.where(mask_split)[0][i]
                        rep_z = new_z[parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                        rep_lnw = new_lnw[parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                        rep_m = new_m[parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                        rep_z += torch.normal(0, noise_std, size=rep_z.shape, device=device)

                        new_parents = torch.full((mj_new[i], 1), parent_idx, dtype=torch.long, device=device)
                        new_split = torch.full((mj_new[i], 1), next_t, device=device)
                        new_death = torch.full((mj_new[i], 1), -1, dtype=torch.long, device=device)

                        new_z_list.append(rep_z)
                        new_lnw_list.append(rep_lnw)
                        new_m_list.append(rep_m)
                        new_parents_list.append(new_parents)
                        new_split_times_list.append(new_split)
                        new_death_step_list.append(new_death)

                    # Concatenate new daughter cells
                    new_z_cells = torch.cat(new_z_list, dim=0) if new_z_list else torch.empty(0, feature_size, device=device)
                    new_lnw_cells = torch.cat(new_lnw_list, dim=0) if new_lnw_list else torch.empty(0, 1, device=device)
                    new_m_cells = torch.cat(new_m_list, dim=0) if new_m_list else torch.empty(0, m_size, device=device)
                    new_parents = torch.cat(new_parents_list, dim=0) if new_parents_list else torch.empty(0, 1, dtype=torch.long, device=device)
                    new_split_times = torch.cat(new_split_times_list, dim=0) if new_split_times_list else torch.empty(0, 1, device=device)
                    new_death_steps = torch.cat(new_death_step_list, dim=0) if new_death_step_list else torch.empty(0, 1, dtype=torch.long, device=device)

                    current_z = torch.cat([new_z, new_z_cells], dim=0)
                    current_lnw = torch.cat([new_lnw, new_lnw_cells], dim=0)
                    current_m = torch.cat([new_m, new_m_cells], dim=0)
                    replication_parents = torch.cat([replication_parents, new_parents], dim=0)
                    split_times = torch.cat([split_times, new_split_times], dim=0)
                    alive_mask = torch.cat([alive_mask, torch.ones(new_z_cells.shape[0], 1, dtype=torch.bool, device=device)], dim=0)
                    death_step = torch.cat([death_step, new_death_steps], dim=0)

                    # Expand historical records to keep dimensions consistent
                    for hist_idx in range(len(history_z)):
                        for i in range(split_count):
                            if mj_new[i] == 0:
                                continue
                            parent_idx = torch.where(mask_split)[0][i]
                            hz = history_z[hist_idx][parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                            hl = history_lnw[hist_idx][parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                            hm = history_m[hist_idx][parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                            hp = history_parents[hist_idx][parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                            hs = history_split_times[hist_idx][parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                            ha = history_alive[hist_idx][parent_idx].unsqueeze(0).repeat(mj_new[i], 1)
                            hd = history_death_step[hist_idx][parent_idx].unsqueeze(0).repeat(mj_new[i], 1)

                            history_z[hist_idx] = torch.cat([history_z[hist_idx], hz], dim=0)
                            history_lnw[hist_idx] = torch.cat([history_lnw[hist_idx], hl], dim=0)
                            history_m[hist_idx] = torch.cat([history_m[hist_idx], hm], dim=0)
                            history_parents[hist_idx] = torch.cat([history_parents[hist_idx], hp], dim=0)
                            history_split_times[hist_idx] = torch.cat([history_split_times[hist_idx], hs], dim=0)
                            history_alive[hist_idx] = torch.cat([history_alive[hist_idx], ha], dim=0)
                            history_death_step[hist_idx] = torch.cat([history_death_step[hist_idx], hd], dim=0)
            else:
                current_z, current_lnw, current_m = new_z, new_lnw, new_m

            # Recompute ratio after division so sizes match
            if total_new_cells > 0:
                new_w_cells = torch.exp(new_lnw_cells)
                w_next = torch.cat([w_next, new_w_cells], dim=0)
                new_w_prev = []
                for i in range(split_count):
                    if mj_new[i] > 0:
                        parent_idx = torch.where(mask_split)[0][i]
                        new_w_prev.append(w_prev[parent_idx].unsqueeze(0).repeat(mj_new[i], 1))
                if new_w_prev:
                    new_w_prev = torch.cat(new_w_prev, dim=0)
                    w_prev = torch.cat([w_prev, new_w_prev], dim=0)
                ratio = w_next / w_prev

            # 2. Extinction: remove cells probabilistically
            mask_extinct = (ratio < 1).squeeze(-1) & alive_mask.squeeze(-1)
            if mask_extinct.any():
                r_ext = ratio[mask_extinct].squeeze(-1)
                keep_mask = torch.rand_like(r_ext) < r_ext
                dying_mask = ~keep_mask
                dying_indices = mask_extinct.nonzero(as_tuple=True)[0][dying_mask]

                alive_mask[dying_indices] = False
                death_step[dying_indices] = idx_next  # record death step

                # Set states of dead trajectories to NaN
                current_z[dying_indices] = torch.nan
                current_lnw[dying_indices] = torch.nan
                current_m[dying_indices] = torch.nan

            w_prev = torch.exp(current_lnw)
        else:
            # Non-check step: update alive states only
            dead_mask = ~alive_mask.squeeze(-1) & ~torch.isnan(current_z).any(dim=1)
            if dead_mask.any():
                dead_indices = dead_mask.nonzero(as_tuple=True)[0]
                current_z[dead_indices] = torch.nan
                current_lnw[dead_indices] = torch.nan
                current_m[dead_indices] = torch.nan

            current_z, current_lnw, current_m = new_z, new_lnw, new_m
            w_prev = w_next

        # Append current state to history
        history_z.append(current_z.clone())
        history_lnw.append(current_lnw.clone())
        history_m.append(current_m.clone())
        history_parents.append(replication_parents.clone())
        history_split_times.append(split_times.clone())
        history_alive.append(alive_mask.clone())
        history_death_step.append(death_step.clone())

        idx_next += 1
        step_counter += 1

        if step_counter % check_frequency == 0:
            alive_count = alive_mask.sum().item()
            print(f"Step {step_counter}, total trajectories: {current_z.shape[0]}, alive: {alive_count}")
        if step_counter == n_bins:
            break

    # Stack history over time
    out_z = torch.stack(history_z, dim=0)
    print(out_z.shape)
    out_lnw = torch.stack(history_lnw, dim=0)
    return out_z, out_lnw
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
    np.random.seed(21)          # 把 42 换成任意整数

    chosen_init_idx = np.random.choice(init_idx, size=n_trajectories, replace=replace)
    init_x = torch.tensor(X_raw[chosen_init_idx], dtype=torch.float32, device=device)

    from ..tl.methods import ODEFunc
    ode_func = ODEFunc(model,use_mass=True,score_use=False).to(device)

    t_min, t_max = float(unique_times[0]), float(unique_times[-1])
    t_bins  = torch.linspace(t_min, t_max, n_bins, device=device)
    t_point = torch.tensor(unique_times, dtype=torch.float32, device=device)

    init_lnw = torch.ones(n_trajectories, 1, dtype=torch.float32, device=device, requires_grad=True)
    init_m   = torch.zeros_like(init_lnw)
    initial_state = (init_x, init_lnw, init_m)
    if split_true :
        traj_x, traj_lnw  = euler_odeint_split(ode_func, initial_state, t_bins)
        point_x, point_lnw  = euler_odeint_split(ode_func, initial_state, t_point)
    else :
        traj_x, traj_lnw, _ = odeint(ode_func, initial_state, t_bins,  method='euler')
        point_x, point_lnw, _ = odeint(ode_func, initial_state, t_point, method='euler')

    traj_array  = traj_x.detach().cpu().numpy()          # (T_bins, M, D)
    traj_lnw    = traj_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)
    point_array = point_x.detach().cpu().numpy()         # (T, M, D)
    point_lnw   = point_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)

    import os
    np.save(os.path.join(exp_dir, "ode_traj.npy"),   traj_array)
    np.save(os.path.join(exp_dir, "ode_traj_lnw.npy"), traj_lnw)


    print(f"[generate_ode_trajectories] trajectories saved to {exp_dir}")

    return point_array, traj_array


def simulate_trajectory(model, x0, sigma, time, dt, device):
    x0 = x0.requires_grad_(True)
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device, dtype=torch.float32) / x0.shape[0])
    initial_state = (x0, lnw0)
    sigma=0.0
    class CytoSDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"
        def __init__(self, model, sigma=sigma):
            super().__init__()
            self.model = model
            self.sigma = sigma
        def f(self, t, y):
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            vel_in = torch.cat([z, t_expand], 1)
            drift_z = self.model.velocity_net(vel_in)
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