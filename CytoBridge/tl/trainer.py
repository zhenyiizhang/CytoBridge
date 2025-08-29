import torch
import numpy as np
from CytoBridge.tl.methods import neural_ode_step
from CytoBridge.utils.utils import sample
from CytoBridge.tl.losses import calc_ot_loss, calc_mass_loss
from CytoBridge.tl import methods
import ot

class TrainingPipeline:
    def __init__(self, model, config, batch_size, device):
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None
        self.device = device
        self.model.to(device)
        # TODO: maybe there is a better way to do this
        # ideally the model itself should be enough to decide as g=0, but the mass matching loss invloves the training of v，so there is an option
        if 'growth' in self.config['model']['components']:
            self.use_mass = True
        else:
            self.use_mass = False

    def _setup_stage(self, stage_params):
        """Initialize or reset optimizer and scheduler according to the current stage parameters."""
        lr = stage_params['lr']
        print(f"Setting up optimizer for new stage with LR: {lr}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=stage_params.get('scheduler_step_size', 100),
            gamma=stage_params.get('scheduler_gamma', 0.5)
        )

    def train(self, data, time_points):
        training_plan = self.config['training']['plan']
        base_defaults = self.config['training']['defaults']

        for stage_config in training_plan:
            stage_params = base_defaults.copy()
            stage_params.update(stage_config) # Stage-specific config overrides default config

            print(f"\n--- Starting Stage: {stage_params['name']} ---")
            print(f"  Mode: {stage_params['mode']}, Epochs: {stage_params['epochs']}")
            
            self._setup_stage(stage_params)
            
            if stage_params['mode'] == 'neural_ode':
                self.run_neural_ode_stage(stage_params, data, time_points)
            elif stage_params['mode'] == 'flow_matching':
                print("  Flow Matching mode is not yet implemented. Skipping.")
                # TODO: Implement flow matching stage
                pass
            else:
                raise ValueError(f"Unknown training mode: {stage_params['mode']}")
        
        # TODO: it is also possible to maintain a best model checkpoint
        return self.model
    
    def run_neural_ode_stage(self, stage_params, data, time_points):
        """Run a complete Neural ODE training stage."""
        epochs = stage_params['epochs']
        assert stage_params['method'] in methods.__all__, "Method must be a valid method in methods.py"
        ode_func = getattr(methods, stage_params['method'])(self.model)
        for epoch in range(epochs):
            # Call single epoch training logic
            loss = self.train_neural_ode_epoch(stage_params, data, time_points, ode_func, self.device)
            
            if self.scheduler:
                self.scheduler.step()
            if epoch % 10 == 0:
                print(f"  Stage '{stage_params['name']}', Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def train_neural_ode_epoch(self, staged_params, data, time_points, ODE_func, device,):
        # Parse parameters
        lambda_ot = staged_params['lambda_ot']
        lambda_mass = staged_params['lambda_mass']
        lambda_energy = staged_params['lambda_energy']
        global_mass = staged_params['global_mass']
        OT_loss = staged_params['OT_loss']
        # Sample initial data
        x0 = sample(data[0], self.batch_size).to(device)
        # Initialize log weights
        lnw0 = torch.log(torch.ones(self.batch_size,1) / (self.batch_size)).to(device)
        # Record the mass of the initial data
        mass_0 = data[0].shape[0]

        loss_epoch = 0
        for idx in range(1, len(time_points)):
            self.optimizer.zero_grad()
            t0 = time_points[idx-1]
            t1 = time_points[idx]
            # Sample data at time t1
            data_t1 = sample(data[idx], self.batch_size).to(device)
            # Calculate relative mass
            mass_1 = data[idx].shape[0]
            relative_mass = mass_1 / mass_0

            # Run neural ODE
            x1, lnw1, e1 = neural_ode_step(ODE_func, x0, lnw0, t0, t1, device)

            # calculate the loss
            loss_ot = calc_ot_loss(x1, data_t1, lnw1, OT_loss)
            if self.use_mass:
                loss_mass = calc_mass_loss(x1, data_t1, lnw1, relative_mass, global_mass)
            else:
                loss_mass = 0

            # TODO: add pinn loss
            # print(loss_ot.item(), loss_mass.item(), e1.mean().item())
            loss = lambda_ot * loss_ot + lambda_mass * loss_mass + lambda_energy * e1.mean()
            loss.backward()
            self.optimizer.step()

            # Update states
            x0 = x1.clone().detach()
            lnw0 = lnw1.clone().detach()

            loss_epoch += loss.item()
        
        return loss_epoch / (len(time_points) - 1)
    
    def train_flow_matching_epoch(self, data, time_points):
        # TODO: Implement flow matching epoch
        pass
    
    def evaluate(self, data, time_points):
        # TODO: this function needs to be improved
        print(f"\n--- Starting Stage: Evaluation ---")
        device = self.device
        x0 = data[0].to(device)
        # Initialize log weights
        lnw0 = torch.log(torch.ones(x0.shape[0],1) / (x0.shape[0])).to(device)
        ODE_func = getattr(methods, 'ODEFunc')(self.model)
        for idx in range(1, len(time_points)):
            t0 = time_points[0]
            t1 = time_points[idx]
            x1, lnw1, e1 = neural_ode_step(ODE_func, x0, lnw0, t0, t1, device)
            data_t1 = data[idx]
            m1 = torch.exp(lnw1)
            m1 = m1 / m1.sum()
            m2 = torch.ones(data_t1.shape[0],1).float()
            m2 = m2 / m2.sum()
            m1 = m1.squeeze(1)
            m2 = m2.squeeze(1)
            M = torch.cdist(x1, data_t1, p=2)
            W1 = ot.emd2(m1.detach().cpu().numpy(), m2.detach().cpu().numpy(), M.detach().cpu().numpy(), numItermax=1e7)
            print(f'Time point {t1}, W1: {W1}')
        return W1