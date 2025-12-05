import torch, math
import torch.nn as nn

class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, cutoff_sr = 0.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.cutoff_sr = cutoff_sr
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 1.0
        self.cutoff_fn = CosineCutoff(cutoff)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)
    def _initial_params(self):
        if self.cutoff == 0:
            return torch.zeros(self.num_rbf), torch.tensor(0.0)
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        end_value = torch.exp(torch.scalar_tensor(-self.cutoff_sr))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf)
        return means, betas
    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)
    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)

class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff
    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

def cal_interaction(z, lnw, interaction_potential, m=16, cutoff=1000, use_mass = True):
    w = torch.exp(lnw)
    n = w.shape[0]
    w = w * n
    batch_size, embed_dim = z.shape
    device = z.device
    perm = torch.randperm(batch_size, device=device)
    z_shuffled = z[perm]
    if batch_size % m == 0:
        num_full_groups = batch_size // m
        remainder = batch_size % m
    else:
        num_full_groups = batch_size // m - 1
        remainder = batch_size % m + m
    net_force = torch.zeros_like(z)
    if num_full_groups > 0:
        groups_z = z_shuffled[:num_full_groups * m].view(num_full_groups, m, embed_dim)
        idx = torch.triu_indices(m, m, offset=1, device=device)
        num_pairs = idx.shape[1]
        diffs = groups_z[:, idx[0], :] - groups_z[:, idx[1], :]
        mask = (diffs.norm(dim=-1) <= cutoff).float().unsqueeze(-1)
        diffs_flat = diffs.reshape(num_full_groups * num_pairs, embed_dim)
        diffs_flat.requires_grad_(True)
        potentials = interaction_potential(diffs_flat)
        grad_potentials = torch.autograd.grad(
            outputs=potentials,
            inputs=diffs_flat,
            grad_outputs=torch.ones_like(potentials),
            create_graph=True,
        )[0]
        f_pairs_flat = -grad_potentials
        f_pairs = f_pairs_flat.view(num_full_groups, num_pairs, embed_dim)
        f_pairs = f_pairs * mask
        mass_groups = w[perm[:num_full_groups * m]].view(num_full_groups, m, 1)
        force_groups = torch.zeros(num_full_groups, m, embed_dim, device=device)
        i_idx = idx[0].unsqueeze(0).expand(num_full_groups, -1)
        j_idx = idx[1].unsqueeze(0).expand(num_full_groups, -1)
        mass_j = mass_groups.gather(1, j_idx.unsqueeze(-1))
        if use_mass:
            contrib_i = f_pairs * mass_j
        else:
            contrib_i = f_pairs
        force_groups.scatter_add_(1, i_idx.unsqueeze(-1).expand(-1, -1, embed_dim), contrib_i)
        mass_i = mass_groups.gather(1, i_idx.unsqueeze(-1))
        if use_mass:
            contrib_j = -f_pairs * mass_i
        else:
            contrib_j = -f_pairs
        force_groups.scatter_add_(1, j_idx.unsqueeze(-1).expand(-1, -1, embed_dim), contrib_j)
        force_groups = force_groups / (m - 1)
        net_force[perm[:num_full_groups * m]] = force_groups.reshape(num_full_groups * m, embed_dim)
    if remainder > 0:
        group_z = z_shuffled[num_full_groups * m:]
        idx_rem = torch.triu_indices(remainder, remainder, offset=1, device=device)
        diffs = group_z[idx_rem[0]] - group_z[idx_rem[1]]
        mask_rem = (diffs.norm(dim=-1) <= cutoff).float().unsqueeze(-1)
        diffs.requires_grad_(True)
        potentials = interaction_potential(diffs)
        grad_potentials = torch.autograd.grad(
            outputs=potentials,
            inputs=diffs,
            grad_outputs=torch.ones_like(potentials),
            create_graph=True,
        )[0]
        f_pairs = -grad_potentials
        f_pairs = f_pairs * mask_rem
        mass_group = w[perm[num_full_groups * m:]].view(remainder, 1)
        force_group = torch.zeros(remainder, embed_dim, device=device)
        i_idx_rem = idx_rem[0]
        j_idx_rem = idx_rem[1]
        mass_j = mass_group[j_idx_rem]
        if use_mass:
            contrib_i = f_pairs * mass_j
        else:
            contrib_i = f_pairs
        force_group.scatter_add_(0, i_idx_rem.unsqueeze(-1).expand(-1, embed_dim), contrib_i)
        mass_i = mass_group[i_idx_rem]
        if use_mass:
            contrib_j = -f_pairs * mass_i
        else:
            contrib_j = -f_pairs
        force_group.scatter_add_(0, j_idx_rem.unsqueeze(-1).expand(-1, embed_dim), contrib_j)
        force_group = force_group / (remainder - 1)
        net_force[perm[num_full_groups * m:]] = force_group
    return net_force

def euler_sdeint(sde, initial_state, dt, ts):
    device = initial_state[0].device
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0
    output_states = []
    ts_list = ts.tolist()
    next_output_idx = 0
    while current_time <= tf + 1e-8:
        if current_time >= ts_list[next_output_idx] - 1e-8:
            output_states.append(current_state)
            next_output_idx += 1
            if next_output_idx >= len(ts_list):
                break
        t_tensor = torch.tensor([current_time], device=device)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z = sde.g(t_tensor, current_state[0])
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt 
        current_state = (new_z, new_lnw)
        current_time += dt
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    traj_z = torch.stack([state[0] for state in output_states], dim=0)
    traj_lnw = torch.stack([state[1] for state in output_states], dim=0)
    return traj_z, traj_lnw
