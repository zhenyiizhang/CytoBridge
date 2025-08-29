import torch
from torchdiffeq import odeint
import torch.nn as nn

__all__ = ['ODEFunc']

def neural_ode_step(ODE_func, x0, lnw0, t0, t1, device):
    time = torch.Tensor([t0, t1])
    time = time.to(device)

    # initialize the energy
    e0 = torch.zeros_like(lnw0).to(device)
    # initialize the state of neural ODE
    initial_state = (x0, lnw0, e0)
    # solve the neural ODE
    x_t, lnw_t, e_t = odeint(ODE_func, initial_state, time, method='euler', options=dict(step_size=0.1))
    # get the final state
    x_1 = x_t[-1]
    lnw_1 = lnw_t[-1]
    e_1 = e_t[-1]

    return x_1, lnw_1, e_1

class ODEFunc(nn.Module):
    def __init__(self, model):
        super(ODEFunc, self).__init__()
        self.model = model

    def forward(self, t, state):
        x, lnw, _= state
        outputs = self.model(t, x)
        v = outputs['velocity']
        if 'growth' in outputs:
            g = outputs['growth']
        else:
            g = torch.zeros_like(lnw).to(x.device)
        # TODO: add score results and interaction results
        
        dx_dt = v
        dlnw_dt = g
        w = torch.exp(lnw)
        de_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        return dx_dt.float(), dlnw_dt.float(), de_dt.float()
