import torch
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from CytoBridge.tl.methods import neural_ode_step, ODEFunc
from CytoBridge.utils.utils import sample
from CytoBridge.tl.losses import calc_ot_loss, calc_mass_loss
from CytoBridge.tl import methods
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.analysis import simulate_trajectory
import math
import os
import ot
from torchdiffeq import odeint
from torch.optim.lr_scheduler import StepLR


class TrainingPipeline:
    """
    End-to-end training wrapper for a latent Neural-ODE model with optimal-transport
    and mass-conservation losses. Supports multi-stage optimisation schedules,
    learning-rate schedulers, and automatic checkpointing.
    """

    def __init__(self, model, config, batch_size, device, data):
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None  # will be replaced per stage
        self.device = device
        self.model.to(device)

        self.use_mass = 'growth' in self.config['model']['components']

        # Centralised RHS for the ODE
        self.ode_func = ODEFunc(
            model=self.model,
            sigma=config['training']['defaults'].get('sigma', 0.05),
            use_mass=self.use_mass,
        )

        # Lightweight logger and experiment directory
        self.logger = self._setup_logger()
        self.exp_dir = self.config.get('ckpt_dir', './results')
        os.makedirs(self.exp_dir, exist_ok=True)

        # Build a DataFrame compatible with downstream helpers
        self.df = self._prepare_df(data)
        self.groups = sorted(self.df.samples.unique())  # sorted time-point labels

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _setup_logger(self):
        class SimpleLogger:
            @staticmethod
            def info(msg):
                print(f"[INFO] {msg}")
        return SimpleLogger()

    def _prepare_df(self, data):
        """
        Convert a list of per-time-point tensors into a long-format DataFrame
        with columns ['x1', 'x2', 'samples'] where 'samples' is the
        time-point index stored as float64.
        """
        frames = []
        for t_idx, x in enumerate(data):
            x_np = x.cpu().detach().numpy()
            frames.append(
                pd.DataFrame({
                    'x1': x_np[:, 0],
                    'x2': x_np[:, 1],
                    'samples': np.full(x_np.shape[0], float(t_idx))
                })
            )
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Stage builder
    # ------------------------------------------------------------------ #
    def _setup_stage(self, stage_params):
        """
        Configure optimizer and optional learning-rate scheduler for the
        current training stage. Enables / freezes submodule gradients
        according to the stage configuration.
        """
        lr = stage_params['lr']
        print(f"\n====  {stage_params['name']}  ====")

        params = []
        for name, module in self.model.named_children():
            for p in module.parameters():
                p.requires_grad = True
                params.append(p)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, params), lr=lr
        )

        # Scheduler factory
        self.scheduler = None
        sched_type = stage_params.get('scheduler_type')
        if sched_type == 'cosine':
            cosine_epochs = stage_params.get('cosine_epochs', 500)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cosine_epochs, eta_min=1e-5
            )
        elif sched_type == 'steplr':
            self.scheduler = StepLR(
                optimizer=self.optimizer,
                step_size=stage_params['scheduler_step_size'],
                gamma=stage_params['scheduler_gamma']
            )
            print(f"  LR scheduler active: step_size={stage_params['scheduler_step_size']}, "
                  f"gamma={stage_params['scheduler_gamma']}")
        else:
            print("  No scheduler configured – constant learning rate")

        # Pretty print which modules are trainable
        for n, m in self.model.named_children():
            flag = any(p.requires_grad for p in m.parameters())
            print(f"  {n:<15}  grad={flag}")
        print("  Optimiser param shapes:",
              [p.shape for g in self.optimizer.param_groups for p in g['params']])

    # ------------------------------------------------------------------ #
    # High-level training API
    # ------------------------------------------------------------------ #
    def train(self, data, time_points):
        training_plan = self.config['training']['plan']
        base_defaults = self.config['training']['defaults']

        for stage_config in training_plan:
            stage_params = {**base_defaults, **stage_config}
            print(f"\n--- Starting Stage: {stage_params['name']} ---")
            print(f"  Mode: {stage_params['mode']}, Epochs: {stage_params['epochs']}")

            self._setup_stage(stage_params)

            if stage_params['mode'] == 'neural_ode':
                self.run_neural_ode_stage(stage_params, data, time_points)
            else:
                raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        return self.model

    # ------------------------------------------------------------------ #
    # Neural-ODE stage runner
    # ------------------------------------------------------------------ #
    def run_neural_ode_stage(self, stage_params, data, time_points):
        epochs = stage_params['epochs']
        save_strategy = stage_params.get('save_strategy', 'best')  # 'best' or 'last'

        best_loss = float('inf')
        best_state = self.model.state_dict()

        for epoch in range(epochs):
            loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)

            if epoch % 10 == 0:
                print(f"  Stage '{stage_params['name']}', Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                self.logger.info(f"Epoch {epoch:3d} achieved lower loss | total_loss {best_loss:.4f}")
                best_state = self.model.state_dict()

        # Save according to strategy
        save_state = best_state if save_strategy == 'best' else self.model.state_dict()
        save_loss = best_loss if save_strategy == 'best' else \
            self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)

        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = 'best.pth' if save_strategy == 'best' else 'last.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_name))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) saved → {ckpt_dir}/{ckpt_name}")

    # ------------------------------------------------------------------ #
    # Single epoch helpers
    # ------------------------------------------------------------------ #
    def train_neural_ode_epoch(self, stage_params, data, time_points, ode_func):
        lambda_ot = stage_params['lambda_ot']
        lambda_mass = stage_params['lambda_mass']
        lambda_energy = stage_params['lambda_energy']
        global_mass = stage_params['global_mass']
        ot_loss_type = stage_params['OT_loss']

        # Initial condition sampling
        x0 = sample(data[0], self.batch_size).to(self.device)
        lnw0 = torch.log(torch.ones(self.batch_size, 1, device=self.device) / self.batch_size)
        mass_0 = data[0].shape[0]

        total_loss = 0.0
        for idx in range(1, len(time_points)):
            self.optimizer.zero_grad()

            t0, t1 = time_points[idx - 1], time_points[idx]
            data_t1 = sample(data[idx], self.batch_size).to(self.device)
            mass_1 = data[idx].shape[0]
            relative_mass = mass_1 / mass_0

            x1, lnw1, e1 = neural_ode_step(ode_func, x0, lnw0, t0, t1, self.device)

            loss_ot = calc_ot_loss(x1, data_t1, lnw1, ot_loss_type)
            loss_mass = calc_mass_loss(x1, data_t1, lnw1, relative_mass, global_mass) if self.use_mass else 0.0
            loss_energy = e1.mean()

            loss = lambda_ot * loss_ot + lambda_mass * loss_mass + lambda_energy * loss_energy

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Detach for next interval
            x0 = x1.detach()
            lnw0 = lnw1.detach()

            total_loss += loss.item()

        return total_loss / (len(time_points) - 1)

    # ------------------------------------------------------------------ #
    # Evaluation utilities
    # ------------------------------------------------------------------ #
    def evaluate(self, data, time_points):
        """
        Quick evaluation: simulate full trajectory and report Wasserstein-1
        distances to held-out samples at each time point.
        """
        print("\n--- Starting Evaluation ---")
        device = self.device
        x0 = data[0].to(device)
        for p in self.model.parameters():
            p.requiresgrad = False    
        
        sigma = getattr(self, 'sigma', None) or 0.0
        point, weight = simulate_trajectory(
            self.model, x0, sigma, time_points, dt=0.01, device=device
        )

        wasserstein_scores = []
        for idx in range(1, len(time_points)):
            t0, t1 = time_points[0], time_points[idx]
            data_t1 = data[idx].detach().cpu().numpy()
            x1 = point[idx]
            m1 = weight[idx]
            tmv = abs(m1.sum() - data[idx].shape[0] / data[0].shape[0])
            m1 = m1 / m1.sum()

            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
            cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

            w1 = ot.emd2(m2, m1.reshape(-1), cost_matrix, numItermax=int(1e7))
            print(f"  Time point {t1}: Wasserstein-1 distance = {w1:.4f}")
            print(f"  Time point {t1}: TMV = {tmv:.4f}")
            wasserstein_scores.append(w1)

        return wasserstein_scores