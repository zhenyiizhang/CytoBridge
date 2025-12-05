import torch
from torchdiffeq import odeint
import torch.nn as nn

__all__ = ['ODEFunc']


def neural_ode_step(ODE_func, x0, lnw0, t0, t1, device):
    time = torch.Tensor([t0, t1]).to(device)
    e0 = torch.zeros_like(lnw0).to(device)
    initial_state = (x0, lnw0, e0)
    x_t, lnw_t, e_t = odeint(ODE_func, initial_state, time, method='euler', options=dict(step_size=0.1))
    return x_t[-1], lnw_t[-1], e_t[-1]


# TODO: add interaction results

class ODEFunc(nn.Module):
    def __init__(self, model, sigma=0.05, model_stra=None,
                 use_mass=False, score_use=False, interaction_use=False):
        super(ODEFunc, self).__init__()
        self.model = model
        self.sigma = sigma

        if model_stra is not None:
            self.use_mass = 'g' in model_stra
            self.score_use = 's' in model_stra
            self.interaction_use = 'i' in model_stra
        else:
            # 否则使用单独传入的参数
            self.use_mass = use_mass
            self.score_use = score_use
            self.interaction_use = interaction_use

    def forward(self, t, state):

        x, lnw, m = state
        batch_size = x.shape[0]

        outputs = self.model(t, x, lnw)
        v = outputs['velocity']

        if self.use_mass and 'growth' in outputs:
            g = outputs['growth']
        else:
            g = torch.zeros(batch_size, 1, device=x.device)

        if (self.score_use and 'score' in outputs) and (
            not self.interaction_use or 
            not 'interaction' in outputs or 
            self.model.interaction_net.cutoff == 0
        ):
            s = outputs['score']
            grad_s = outputs['score_gradient']

            v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
            grad_s_norm_sq = torch.norm(grad_s, p=2, dim=1, keepdim=True) ** 2

            de_dt = ( v_norm_sq / 2
                     + grad_s_norm_sq / 2
                     - (0.5 * self.sigma ** 2 * g + s * g)
                     + g ** 2) * torch.exp(lnw)
           # print("V+G+s/V+s")

        elif (self.score_use and 'score' in outputs) and (self.interaction_use and 'interaction' in outputs):
            s = outputs['score']
            grad_s = outputs['score_gradient']
            norm_grad_s = torch.norm(grad_s, p=2, dim=1).unsqueeze(1).requires_grad_(True)

            de_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                        (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1) + g ** 2) * torch.exp(lnw)
            #print("V+S+G+I/V+S+I")
        else:
            v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
            de_dt = (0.5 * v_norm_sq + g ** 2) * torch.exp(lnw)
            #print("V/V+G/V+I/V+G+I")

        if self.interaction_use and 'interaction' in outputs:
            net_force = outputs['interaction']
            v = v + net_force
            #print("net_force = outputs['interaction']v = v + net_force")


        dx_dt = v
        dlnw_dt = g
        dm_dt = de_dt

        self._check_dim_consistency(x, dx_dt, "x")
        self._check_dim_consistency(lnw, dlnw_dt, "lnw")
        self._check_dim_consistency(m, dm_dt, "m")

        return dx_dt.float(), dlnw_dt.float(), dm_dt.float()

    def _check_dim_consistency(self, var, deriv, name):
        """Check dimension consistency"""
        assert var.shape == deriv.shape, \
            f"Dimension mismatch: {name} shape {var.shape} and derivative shape {deriv.shape} are not consistent"



class ODEFunc3(nn.Module):
    def __init__(self, f_net,sf2m_score_model,sigma, use_mass, thre):
        super(ODEFunc3, self).__init__()
        self.f_net = f_net
        self.interaction_potential = f_net.interaction_net
        self.sf2m_score_model = sf2m_score_model
        self.sigma=sigma
        self.use_mass = use_mass
        self.thre = thre
    def forward(self, t, state):
        z, lnw, m = state
        w = torch.exp(lnw)
        z.requires_grad_(True)
        lnw.requires_grad_(True)
        m.requires_grad_(True)
        t.requires_grad_(True)
        v, g, _, _ = self.f_net(t, z)
        v.requires_grad_(True)
        g.requires_grad_(True)
        time=t.expand(z.shape[0],1)
        time.requires_grad_(True)
        s=self.sf2m_score_model(time,z)
        dz_dt = v
        dlnw_dt = g
        z=z.requires_grad_(True)
        grad_s = torch.autograd.grad(outputs=s, inputs=z,grad_outputs=torch.ones_like(s),create_graph=True)[0]
        norm_grad_s = torch.norm(grad_s, dim=1).unsqueeze(1).requires_grad_(True)
        net_force = cal_interaction(z, lnw, self.interaction_potential, threshold = self.thre, use_mass = self.use_mass).float()
        if self.use_mass:
            if self.interaction_potential.cutoff != 0:
                dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                        (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1) + g ** 2) * w
            else:
                dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                        (norm_grad_s ** 2) / 2 - (1 / 2 * self.sigma ** 2 *g + s* g) + g ** 2) * w
        else:
            dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                    (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1))
        return dz_dt.float()+net_force.float(), dlnw_dt.float(), dm_dt.float()
