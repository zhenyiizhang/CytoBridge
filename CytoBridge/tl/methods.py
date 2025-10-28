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

    def __init__(self, model, sigma=0.05, use_mass=False, score_use=False):
        super(ODEFunc, self).__init__()
        self.model = model
        self.sigma = sigma
        self.use_mass = use_mass
        self.score_use = score_use

    def forward(self, t, state):

        x, lnw, m = state
        batch_size = x.shape[0]

        outputs = self.model(t, x)
        v = outputs['velocity']

        if self.use_mass and 'growth' in outputs:
            g = outputs['growth']
        else:
            g = torch.zeros(batch_size, 1, device=x.device)

        if self.score_use and 'score' in outputs:
            s = outputs['score']

            grad_s = outputs['score_gradient']

            v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
            grad_s_norm_sq = torch.norm(grad_s, p=2, dim=1, keepdim=True) ** 2

            de_dt = ( v_norm_sq / 2
                     + grad_s_norm_sq / 2
                     - (0.5 * self.sigma ** 2 * g + s * g)
                     + g ** 2) * torch.exp(lnw)

        else:
            v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
            de_dt = (0.5 * v_norm_sq + g ** 2) * torch.exp(lnw)

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


