# Import necessary libraries for data processing, deep learning, optimal transport, and utility functions
import numpy as np, pandas as pd
import torch  # PyTorch library for tensor operations and deep learning
import random  # For random number generation
import yaml  # For YAML file parsing (configuration)
from copy import deepcopy  # For deep copying objects
from tqdm import tqdm  # For progress bar visualization
import ot  # Optimal Transport library (POT)
import math  # For mathematical operations (e.g., sqrt, pi)

# This block of code is adapted from the torchCFM library (Conditional Flow Matching in PyTorch)
import math
import warnings  # For handling warning messages
from functools import partial  # For creating partial functions with fixed parameters
from typing import Optional  # For type hinting optional parameters

import numpy as np
import ot as pot  # Alias optimal transport library as pot
import torch
import matplotlib.pyplot as plt  # For plotting (commented out in original code)
import warnings  # Re-import for clarity (redundant but preserved as in original)
from warnings import catch_warnings, simplefilter  # For context-aware warning control

from typing import Union


# Suppress numerical error warnings from the ot library (specifically line 751 in _sinkhorn.py)
warnings.filterwarnings(
    "ignore",
    message="Numerical errors at iteration.*",  # Match the warning message pattern
    category=UserWarning,  # Specify the warning category to suppress
    module="ot.unbalanced._sinkhorn"  # Specify the exact module generating the warning
)


class OTPlanSampler:
    """
    Class for sampling Optimal Transport (OT) plans and matching pairs between source (x0) and target (x1) distributions.
    
    Supports multiple OT methods (exact, Sinkhorn, unbalanced Sinkhorn, partial OT) to compute transport plans,
    and provides methods to sample indices or data pairs based on the computed OT plan.
    """

    def __init__(
            self,
            method: str,
            reg: float = 0.05,
            reg_m: float = 1.0,
            normalize_cost: bool = False,
            warn: bool = True,
    ) -> None:
        """
        Initialize the OTPlanSampler with specified OT method and hyperparameters.
        
        Args:
            method: OT method to use. Options: "exact" (EMD), "sinkhorn", "unbalanced_sinkhorn", "partial".
            reg: Entropic regularization coefficient (used for Sinkhorn-based methods).
            reg_m: Mass regularization coefficient (used for unbalanced Sinkhorn).
            normalize_cost: Whether to normalize the cost matrix by its maximum value.
            warn: Whether to issue warnings (e.g., for numerical errors in OT plan).
        """
        # Assign OT function based on the specified method (partial fixes hyperparameters like reg)
        if method == "exact":
            self.ot_fn = pot.emd  # Exact Earth Mover's Distance (no regularization)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)  # Entropic Sinkhorn
        elif method == "unbalanced_sinkhorn":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)  # Unbalanced Sinkhorn
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)  # Partial OT
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store hyperparameters and settings
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        """
        Compute the OT plan (transport matrix) between source distribution x0 and target distribution x1.
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...) (supports high-dimensional data).
            x1: Target data tensor of shape (n_samples_1, ...).
        
        Returns:
            p: OT plan matrix of shape (n_samples_0, n_samples_1), where p[i,j] is the transport weight from x0[i] to x1[j].
        """
        # Define uniform marginals for x0 and x1 (each sample has equal weight)
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        
        # Flatten high-dimensional data to 2D (required for cost matrix computation)
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        # Redundant reshape (preserved as in original code)
        x1 = x1.reshape(x1.shape[0], -1)
        
        # Compute squared Euclidean distance cost matrix (M[i,j] = ||x0[i] - x1[j]||²)
        M = torch.cdist(x0, x1) ** 2
        
        # Normalize cost matrix if enabled (avoid when using minibatches to preserve scale)
        if self.normalize_cost:
            M = M / M.max()
        
        # Compute OT plan using the pre-assigned ot_fn (convert tensor to numpy for POT compatibility)
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        
        # Check for numerical issues in the OT plan
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        
        # Fallback to uniform plan if the computed plan has near-zero total mass (numerical failure)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        
        return p

    def sample_map(self, pi, batch_size, replace=True):
        """
        Sample (i,j) index pairs from the OT plan pi.
        
        Args:
            pi: OT plan matrix of shape (n_samples_0, n_samples_1).
            batch_size: Number of index pairs to sample.
            replace: Whether to sample with replacement.
        
        Returns:
            Tuple of (i_indices, j_indices): Sampled source and target indices, each of shape (batch_size,).
        """
        # Flatten the OT plan to a 1D probability vector (after normalizing to sum to 1)
        p = pi.flatten()
        p = p / p.sum()
        
        # Sample flat indices from the normalized distribution
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        
        # Convert flat indices back to 2D (i,j) pairs
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        """
        Sample matched (x0[i], x1[j]) pairs using the OT plan between x0 and x1.
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...).
            x1: Target data tensor of shape (n_samples_1, ...).
            replace: Whether to sample with replacement.
        
        Returns:
            Tuple of (x0_sampled, x1_sampled): Sampled source and target data pairs, each of shape (batch_size, ...).
        """
        # Compute OT plan between x0 and x1
        pi = self.get_map(x0, x1)
        # Sample (i,j) indices from the OT plan
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        # Return matched data pairs
        return x0[i], x1[j]

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        """
        Sample matched (x0[i], x1[j]) pairs along with their corresponding labels (y0[i], y1[j]).
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...).
            x1: Target data tensor of shape (n_samples_1, ...).
            y0: Source labels tensor of shape (n_samples_0, ...) (optional).
            y1: Target labels tensor of shape (n_samples_1, ...) (optional).
            replace: Whether to sample with replacement.
        
        Returns:
            Tuple of (x0_sampled, x1_sampled, y0_sampled, y1_sampled): Sampled data and labels (labels are None if not provided).
        """
        # Compute OT plan between x0 and x1
        pi = self.get_map(x0, x1)
        # Sample (i,j) indices from the OT plan
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        
        # Return data pairs and their labels (slice labels using sampled indices if provided)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        """
        Sample a trajectory of matched data points across multiple time steps using OT plans between consecutive time steps.
        
        Args:
            X: Tensor of shape (n_samples, n_times, ...) representing data at multiple time steps.
        
        Returns:
            to_return: Sampled trajectory tensor of shape (n_samples, n_times, ...), where each sample's path is consistent across time.
        """
        # Number of time steps in the trajectory
        times = X.shape[1]
        # Compute OT plans between each pair of consecutive time steps
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))
        
        # Initialize indices with all samples (time step 0 uses original indices)
        indices = [np.arange(X.shape[0])]
        # Sample indices for each subsequent time step using the corresponding OT plan
        for pi in pis:
            j = []
            # For each index in the previous time step, sample a matching index in the next time step
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))
        
        # Extract data points from X using the sampled indices to form the trajectory
        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        # Stack time steps back into a single tensor
        to_return = np.stack(to_return, axis=1)
        return to_return


def wasserstein(
        x0: torch.Tensor,
        x1: torch.Tensor,
        method: Optional[str] = None,
        reg: float = 0.05,
        power: int = 2,
        **kwargs,
) -> float:
    """
    Compute the Wasserstein distance between source distribution x0 and target distribution x1.
    
    Supports exact Wasserstein (EMD) and entropic regularized Wasserstein (Sinkhorn) distances.
    
    Args:
        x0: Source data tensor of shape (n_samples_0, ...).
        x1: Target data tensor of shape (n_samples_1, ...).
        method: OT method to use. Options: "exact" (EMD), "sinkhorn"; defaults to "exact".
        reg: Entropic regularization coefficient (used only for "sinkhorn" method).
        power: Power of the distance metric (1 for L1, 2 for L2; only 1 or 2 supported).
        **kwargs: Additional keyword arguments for the OT function.
    
    Returns:
        ret: Computed Wasserstein distance (scalar).
    """
    # Validate that power is either 1 or 2 (only L1 and L2 distances supported)
    assert power == 1 or power == 2
    
    # Assign OT function for distance computation (pot.emd2 for exact, pot.sinkhorn2 for regularized)
    if method == "exact" or method is None:
        ot_fn = pot.emd2  # Computes exact Wasserstein distance (sum of transport costs)
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)  # Computes regularized Wasserstein distance
    else:
        raise ValueError(f"Unknown method: {method}")

    # Define uniform marginals for x0 and x1
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    
    # Flatten high-dimensional data to 2D (required for cost matrix computation)
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    
    # Compute Euclidean distance matrix (L1 if power=1, L2 if power=2)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M ** 2  # Use squared L2 distance for Wasserstein-2
    
    # Compute Wasserstein distance (convert tensor to numpy for POT compatibility)
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    
    # Take square root for Wasserstein-2 (since we used squared distance)
    if power == 2:
        ret = math.sqrt(ret)
    
    return ret


def pad_t_like_x(t, x):
    """
    Reshape the time tensor `t` to match the dimensionality of the data tensor `x` (for broadcasting).
    
    If `t` is a scalar (float/int), it is returned as-is. Otherwise, `t` is reshaped to (batch_size, 1, ..., 1)
    to match the number of dimensions of `x`.
    
    Args:
        t: Time tensor/scalar of shape (batch_size,) or scalar.
        x: Data tensor of shape (batch_size, dim1, dim2, ...) (used to infer target shape).
    
    Returns:
        t_padded: Reshaped time tensor matching the dimensionality of `x`.
    """
    if isinstance(t, (float, int)):
        return t
    # Reshape t to (batch_size, 1, ..., 1) where the number of 1s equals (x.dim() - 1)
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """
    Base class for Conditional Flow Matching (CFM), which models the flow of data from a source distribution (x0)
    to a target distribution (x1) at a continuous time `t` (0 ≤ t ≤ 1).
    
    Provides core methods to compute time-dependent means (mu_t), variances (sigma_t), sample intermediate states (xt),
    and compute the conditional flow (ut) that drives the transformation from x0 to x1.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        """
        Initialize the ConditionalFlowMatcher with a fixed variance parameter.
        
        Args:
            sigma: Fixed variance for sampling intermediate states (xt). Set to 0 for deterministic flow.
        """
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean (mu_t) of the intermediate state distribution at time `t` (linear interpolation between x0 and x1).
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,) (padded to match x0's dimensions).
        
        Returns:
            mu_t: Mean tensor of shape (batch_size, ...), where mu_t = (1-t)*x0 + t*x1.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the variance (sigma_t) of the intermediate state distribution at time `t` (fixed to self.sigma).
        
        Args:
            t: Time tensor (unused here, preserved for consistency with subclasses).
        
        Returns:
            sigma_t: Fixed variance scalar/tensor (self.sigma).
        """
        del t  # Unused in base class (fixed sigma)
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Sample an intermediate state (xt) at time `t` from the distribution N(mu_t, sigma_t²I).
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,).
            epsilon: Noise tensor of shape (batch_size, ...) (sampled from N(0,1)).
        
        Returns:
            xt: Sampled intermediate state tensor of shape (batch_size, ...).
        """
        # Compute mean of the intermediate distribution
        mu_t = self.compute_mu_t(x0, x1, t)
        # Compute variance of the intermediate distribution
        sigma_t = self.compute_sigma_t(t)
        # Reshape sigma_t to match x0's dimensions (for broadcasting)
        sigma_t = pad_t_like_x(sigma_t, x0)
        # Sample xt = mu_t + sigma_t * epsilon
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional flow (ut) that drives the transformation from x0 to x1 (constant flow in base class).
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor (unused here, preserved for consistency with subclasses).
            xt: Intermediate state tensor (unused here, preserved for consistency with subclasses).
        
        Returns:
            ut: Conditional flow tensor of shape (batch_size, ...), where ut = x1 - x0.
        """
        del t, xt  # Unused in base class (constant flow)
        return x1 - x0

    def sample_noise_like(self, x):
        """
        Sample a standard normal noise tensor with the same shape as `x`.
        
        Args:
            x: Data tensor of shape (batch_size, ...) (used to infer noise shape).
        
        Returns:
            epsilon: Noise tensor of shape (batch_size, ...) (sampled from N(0,1)).
        """
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Sample an intermediate time `t`, intermediate state `xt`, and corresponding conditional flow `ut`.
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Pre-sampled time tensor of shape (batch_size,) (optional; sampled uniformly if None).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            If return_noise: Tuple of (t, xt, ut, epsilon)
            Else: Tuple of (t, xt, ut)
        """
        # Sample time `t` uniformly from [0,1] if not provided
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        # Ensure `t` has the same batch size as x0
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        # Sample standard normal noise
        eps = self.sample_noise_like(x0)
        # Sample intermediate state xt
        xt = self.sample_xt(x0, x1, t, eps)
        # Compute conditional flow ut
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        
        # Return noise along with other outputs if enabled
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """
        Compute a time-dependent lambda term (used in flow matching loss functions).
        
        Args:
            t: Time tensor of shape (batch_size,).
        
        Returns:
            lambda_t: Lambda tensor of shape (batch_size, ...), where lambda_t = 2*sigma_t / sigma².
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma ** 2 + 1e-8)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Conditional Flow Matcher (CFM) that uses **exact Optimal Transport (OT)** to sample matched (x0, x1) pairs.
    
    Inherits from ConditionalFlowMatcher and overrides the `sample_location_and_conditional_flow` method
    to first sample OT-matched (x0, x1) pairs before computing xt and ut.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        """
        Initialize the ExactOptimalTransportConditionalFlowMatcher.
        
        Args:
            sigma: Fixed variance for sampling intermediate states (xt) (passed to parent class).
        """
        super().__init__(sigma)
        # Initialize OT sampler with exact EMD method
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Sample OT-matched (x0, x1) pairs first, then compute xt, ut, and (optionally) noise.
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...).
            x1: Target data tensor of shape (n_samples_1, ...).
            t: Pre-sampled time tensor of shape (batch_size,) (optional; sampled uniformly if None).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            Same as parent class, but x0 and x1 are OT-matched pairs.
        """
        # Sample OT-matched (x0, x1) pairs using exact EMD
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        # Delegate to parent class to compute t, xt, ut, and (optionally) noise
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
            self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        """
        Sample OT-matched (x0, x1) pairs along with their labels, then compute xt, ut, and (optionally) noise.
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...).
            x1: Target data tensor of shape (n_samples_1, ...).
            y0: Source labels tensor of shape (n_samples_0, ...) (optional).
            y1: Target labels tensor of shape (n_samples_1, ...) (optional).
            t: Pre-sampled time tensor of shape (batch_size,) (optional; sampled uniformly if None).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            If return_noise: Tuple of (t, xt, ut, y0_sampled, y1_sampled, epsilon)
            Else: Tuple of (t, xt, ut, y0_sampled, y1_sampled)
        """
        # Sample OT-matched (x0, x1) pairs and their labels
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        
        # Delegate to parent class to compute t, xt, ut, and (optionally) noise
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Conditional Flow Matcher (CFM) that models flow from a "zero" source (instead of x0) to the target x1.
    
    Overrides parent class methods to compute mu_t as t*x1 (no x0 dependence) and sigma_t as time-dependent.
    """

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean (mu_t) of the intermediate state distribution (depends only on x1 and t).
        
        Args:
            x0: Source data tensor (unused here; flow starts from "zero" instead of x0).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,) (padded to match x1's dimensions).
        
        Returns:
            mu_t: Mean tensor of shape (batch_size, ...), where mu_t = t*x1.
        """
        del x0  # Unused (flow starts from "zero" source)
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the time-dependent variance (sigma_t) of the intermediate state distribution.
        
        Args:
            t: Time tensor of shape (batch_size,).
        
        Returns:
            sigma_t: Variance tensor of shape (batch_size,), where sigma_t = 1 - (1 - sigma)*t.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional flow (ut) for the target-only flow model.
        
        Args:
            x0: Source data tensor (unused here).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,) (padded to match x1's dimensions).
            xt: Intermediate state tensor of shape (batch_size, ...).
        
        Returns:
            ut: Conditional flow tensor of shape (batch_size, ...), where ut = (x1 - (1-sigma)*xt) / [(1 - (1-sigma)*t)].
        """
        del x0  # Unused (flow starts from "zero" source)
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class ConditionalRegularizedUnbalancedFlowMatcher(ConditionalFlowMatcher):
    """
    Conditional Flow Matcher (CFM) for **unbalanced** distributions, using regularized OT and time-dependent variance.
    
    Supports unbalanced OT (via Sinkhorn) and computes time-dependent sigma_t (sigma*sqrt(t*(1-t))). Also includes
    methods to compute additional terms (gt_samp, weights) related to unbalanced mass.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        """
        Initialize the ConditionalRegularizedUnbalancedFlowMatcher.
        
        Args:
            sigma: Scaling factor for time-dependent variance (must be strictly positive).
            ot_method: OT method to use for sampling (passed to OTPlanSampler).
        """
        # Validate sigma (must be positive to avoid numerical instability)
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        
        # Initialize parent class with sigma
        super().__init__(sigma)
        self.ot_method = ot_method
        # Initialize OT sampler with the specified method and entropy regularization (2*sigma²)
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma ** 2)

    def compute_sigma_t(self, t):
        """
        Compute the time-dependent variance (sigma_t) for unbalanced flow (sigma*sqrt(t*(1-t))).
        
        Args:
            t: Time tensor of shape (batch_size,) (padded to match data dimensions).
        
        Returns:
            sigma_t: Variance tensor of shape (batch_size, ...).
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional flow (ut) for unbalanced flow (includes time-dependent correction term).
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,) (padded to match data dimensions).
            xt: Intermediate state tensor of shape (batch_size, ...).
        
        Returns:
            ut: Conditional flow tensor of shape (batch_size, ...), combining base flow and correction term.
        """
        t = pad_t_like_x(t, x0)
        # Compute mean of the intermediate distribution (linear interpolation)
        mu_t = self.compute_mu_t(x0, x1, t)
        # Compute time-dependent correction factor (derivative of sigma_t / sigma_t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        # Compute ut: base flow (x1 - x0) + correction term (sigma_t'/(sigma_t) * (xt - mu_t))
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, t=None, uot_plan=None, idx_0=None, idx_1=None, return_noise=False):
        """
        Sample intermediate state (xt) and conditional flow (ut) for unbalanced flow, plus unbalanced mass terms (gt_samp, weights).
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Pre-sampled time tensor of shape (batch_size,) (optional; sampled uniformly if None).
            uot_plan: Unbalanced OT plan matrix (used to compute mass terms).
            idx_0: Sampled source indices (used to slice uot_plan).
            idx_1: Sampled target indices (unused here, preserved for consistency).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            If return_noise: Tuple of (t, xt, ut, epsilon, gt_samp, weights)
            Else: Tuple of (t, xt, ut, gt_samp, weights)
        """
        # Sample time `t` uniformly from [0,1] if not provided
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        # Ensure `t` has the same batch size as x0
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        # Sample standard normal noise
        eps = self.sample_noise_like(x0)
        # Sample intermediate state xt
        xt = self.sample_xt(x0, x1, t, eps)
        # Compute conditional flow ut
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        # Compute unbalanced mass terms (gt_samp: log mass ratio, weights: mass^(t-1))
        gt_samp, weights = self.compute_cond_g(uot_plan, idx_0, idx_1, t)
        
        # Return noise along with other outputs if enabled
        if return_noise:
            return t, xt, ut, eps, gt_samp, weights
        else:
            return t, xt, ut, gt_samp, weights

    def compute_cond_g(self, uot_plan: np.ndarray, idx_0: np.ndarray, idx_1: np.ndarray, t: torch.Tensor):
        """
        Compute unbalanced mass terms (gt_samp and weights) from the unbalanced OT plan.
        
        Args:
            uot_plan: Unbalanced OT plan matrix of shape (n_samples_0, n_samples_1).
            idx_0: Sampled source indices (used to slice uot_plan to the current batch).
            idx_1: Sampled target indices (unused here).
            t: Time tensor of shape (batch_size,).
        
        Returns:
            gt_samp: Log mass ratio tensor of shape (batch_size, 1).
            weights: Mass weight tensor of shape (batch_size, 1) (mass^(t-1)).
        """
        # Slice the UOT plan to include only the sampled source indices (current batch)
        selected_uot_plan = uot_plan[idx_0]  # Shape: (batch_size, n_samples_1)
        
        # Compute total mass for each sampled source (sum over target dimensions)
        source_weights = torch.tensor(
            selected_uot_plan.sum(axis=-1, keepdims=True),  # Shape: (batch_size, 1)
            dtype=torch.float32,
            device=t.device
        )
        
        # Small epsilon to avoid log(0) or division by zero
        eps = 1e-10
        
        # Compute weights: (source_mass + eps)^(t - 1) (time-dependent mass weight)
        weights = (source_weights + eps) ** (t.unsqueeze(1) - 1)
        
        # Compute gt_samp: log(source_mass + eps) - log(1 + eps) (log ratio of mass to uniform mass)
        gmaps = (torch.log(source_weights + eps) - torch.log(
            torch.ones_like(source_weights, device=source_weights.device) + eps))
        
        # Unused (preserved as in original code): mean of gmaps
        gmaps_mean = gmaps.mean()
        
        return gmaps, weights

    def guided_sample_location_and_conditional_flow(
            self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        """
        Sample OT-matched (x0, x1) pairs + labels, then compute xt, ut, and (optionally) noise.
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...).
            x1: Target data tensor of shape (n_samples_1, ...).
            y0: Source labels tensor of shape (n_samples_0, ...) (optional).
            y1: Target labels tensor of shape (n_samples_1, ...) (optional).
            t: Pre-sampled time tensor of shape (batch_size,) (optional; sampled uniformly if None).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            Same as ExactOptimalTransportConditionalFlowMatcher.guided_sample_location_and_conditional_flow.
        """
        # Sample OT-matched (x0, x1) pairs and their labels
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        
        # Delegate to parent class to compute t, xt, ut, and (optionally) noise
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Conditional Flow Matcher (CFM) for Schrodinger Bridges, using regularized OT and time-dependent variance.
    
    Similar to ConditionalRegularizedUnbalancedFlowMatcher but focuses on balanced Schrodinger Bridge problems.
    Overrides methods to sample OT-matched pairs before computing flow.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        """
        Initialize the SchrodingerBridgeConditionalFlowMatcher.
        
        Args:
            sigma: Scaling factor for time-dependent variance (must be strictly positive).
            ot_method: OT method to use for sampling (passed to OTPlanSampler).
        """
        # Validate sigma (must be positive to avoid numerical instability)
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        
        # Initialize parent class with sigma
        super().__init__(sigma)
        self.ot_method = ot_method
        # Initialize OT sampler with the specified method and entropy regularization (2*sigma²)
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma ** 2)

    def compute_sigma_t(self, t):
        """
        Compute the time-dependent variance (sigma_t) for Schrodinger Bridge flow (sigma*sqrt(t*(1-t))).
        
        Args:
            t: Time tensor of shape (batch_size,) (padded to match data dimensions).
        
        Returns:
            sigma_t: Variance tensor of shape (batch_size, ...).
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional flow (ut) for Schrodinger Bridge flow (same as unbalanced flow).
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,) (padded to match data dimensions).
            xt: Intermediate state tensor of shape (batch_size, ...).
        
        Returns:
            ut: Conditional flow tensor of shape (batch_size, ...) (same as unbalanced flow).
        """
        t = pad_t_like_x(t, x0)
        # Compute mean of the intermediate distribution (linear interpolation)
        mu_t = self.compute_mu_t(x0, x1, t)
        # Compute time-dependent correction factor (derivative of sigma_t / sigma_t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        # Compute ut: base flow + correction term
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Sample OT-matched (x0, x1) pairs first, then compute xt, ut, and (optionally) noise.
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...).
            x1: Target data tensor of shape (n_samples_1, ...).
            t: Pre-sampled time tensor of shape (batch_size,) (optional; sampled uniformly if None).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            Same as parent class, but x0 and x1 are OT-matched pairs.
        """
        # Sample OT-matched (x0, x1) pairs using the pre-configured OT sampler
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        # Delegate to parent class to compute t, xt, ut, and (optionally) noise
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
            self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        """
        Sample OT-matched (x0, x1) pairs + labels, then compute xt, ut, and (optionally) noise.
        
        Args:
            x0: Source data tensor of shape (n_samples_0, ...).
            x1: Target data tensor of shape (n_samples_1, ...).
            y0: Source labels tensor of shape (n_samples_0, ...) (optional).
            y1: Target labels tensor of shape (n_samples_1, ...) (optional).
            t: Pre-sampled time tensor of shape (batch_size,) (optional; sampled uniformly if None).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            Same as ExactOptimalTransportConditionalFlowMatcher.guided_sample_location_and_conditional_flow.
        """
        # Sample OT-matched (x0, x1) pairs and their labels
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        
        # Delegate to parent class to compute t, xt, ut, and (optionally) noise
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Conditional Flow Matcher (CFM) with **variance-preserving** time-dependent mean and flow.
    
    Uses trigonometric interpolation (cos(πt/2)*x0 + sin(πt/2)*x1) for mu_t to preserve variance,
    and corresponding trigonometric flow.
    """

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the variance-preserving mean (mu_t) using trigonometric interpolation.
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,) (padded to match x0's dimensions).
        
        Returns:
            mu_t: Mean tensor of shape (batch_size, ...), where mu_t = cos(πt/2)*x0 + sin(πt/2)*x1.
        """
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the variance-preserving conditional flow (ut) using trigonometric derivatives.
        
        Args:
            x0: Source data tensor of shape (batch_size, ...).
            x1: Target data tensor of shape (batch_size, ...).
            t: Time tensor of shape (batch_size,) (padded to match x0's dimensions).
            xt: Intermediate state tensor (unused here; flow is deterministic based on x0, x1, and t).
        
        Returns:
            ut: Conditional flow tensor of shape (batch_size, ...), derivative of mu_t with respect to t.
        """
        del xt  # Unused (flow is deterministic, no dependence on xt)
        t = pad_t_like_x(t, x0)
        # Compute ut as the derivative of mu_t w.r.t. t: (π/2) * [cos(πt/2)*x1 - sin(πt/2)*x0]
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


def get_batch(FM, X, trajectory, batch_size, n_times, return_noise=False):
    """
    Construct a training batch by sampling intermediate states (xt) and flows (ut) across consecutive time steps.
    
    Iterates over each pair of consecutive time steps in the trajectory, samples xt and ut using the flow matcher (FM),
    and aggregates results into a single batch.
    
    Args:
        FM: Instance of a ConditionalFlowMatcher subclass (e.g., VariancePreservingConditionalFlowMatcher).
        X: Full data tensor (unused here; preserved for consistency with original code).
        trajectory: Trajectory tensor of shape (n_time_steps, n_samples, ...) (data at each time step).
        batch_size: Number of samples to draw per time step pair (unused; batch size is n_samples).
        n_times: Number of time steps in the trajectory (unused; inferred from trajectory).
        return_noise: Whether to return the noise tensor used to sample xt.
    
    Returns:
        If return_noise: Tuple of (t_batch, xt_batch, ut_batch, noise_batch)
        Else: Tuple of (t_batch, xt_batch, ut_batch)
        All batches have shape (total_samples, ...), where total_samples = n_samples * (n_time_steps - 1).
    """
    # Initialize lists to store batch components (time, xt, ut, noise)
    ts = []
    xts = []
    uts = []
    noises = []

    # Iterate over each pair of consecutive time steps (t_start → t_start+1)
    for t_start in range(n_times - 1):
        # Extract source (t_start) and target (t_start+1) data from the trajectory
        x0 = trajectory[t_start]
        x1 = trajectory[t_start + 1]

        # Sample xt, ut, and (optionally) noise using the flow matcher
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        
        # Offset time to match the global time scale (t_start ≤ t ≤ t_start+1)
        ts.append(t + t_start)
        # Collect intermediate states and flows
        xts.append(xt)
        uts.append(ut)

    # Concatenate components from all time step pairs into a single batch
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    
    # Return noise along with other batch components if enabled
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut


def get_batch_size(FM, X, trajectory, batch_size, time, return_noise=False, hold_one_out=False, hold_out=None):
    """
    Construct a training batch with fixed batch size per time step pair, supporting time scaling and hold-out.
    
    Similar to get_batch, but samples a fixed number of samples (batch_size) per time step pair and scales time
    to match the global time vector. Optional hold-out functionality (unused in current implementation).
    
    Args:
        FM: Instance of a ConditionalFlowMatcher subclass.
        X: Full data tensor (unused here).
        trajectory: Trajectory tensor of shape (n_time_steps, n_samples, ...) (data at each time step).
        batch_size: Number of samples to draw per time step pair.
        time: Global time vector of shape (n_time_steps,) (specifies actual time values for each step).
        return_noise: Whether to return the noise tensor used to sample xt.
        hold_one_out: Whether to hold out a time step (unused here).
        hold_out: Index of the time step to hold out (unused here).
    
    Returns:
        If return_noise: Tuple of (t_batch, xt_batch, ut_batch, noise_batch)
        Else: Tuple of (t_batch, xt_batch, ut_batch)
        All batches have shape (total_samples, ...), where total_samples = batch_size * (n_time_steps - 1).
    """
    # Initialize lists to store batch components
    ts = []
    xts = []
    uts = []
    noises = []

    # Iterate over each pair of consecutive time steps (using indices of the time vector)
    for idx, t_start in enumerate(time[:-1]):
        # Extract source (idx) and target (idx+1) data from the trajectory
        x0 = trajectory[idx]
        x1 = trajectory[idx + 1]
        
        # Randomly sample a fixed number of indices from source and target
        indices0 = np.random.choice(len(x0), size=batch_size, replace=True)
        indices1 = np.random.choice(len(x1), size=batch_size, replace=True)

        # Slice data using sampled indices to get the batch
        x0 = x0[indices0]
        x1 = x1[indices1]

        # Sample xt, ut, and (optionally) noise using the flow matcher
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)

        # Scale time to match the global time scale: t ∈ [t_start, t_start+1]
        ts.append(t * (time[idx + 1] - time[idx]) + t_start)
        # Scale flow by inverse time step (to account for non-uniform time intervals)
        uts.append(ut / (time[idx + 1] - time[idx]))
        # Collect intermediate states
        xts.append(xt)

    # Concatenate components from all time step pairs into a single batch
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)

    # Return noise along with other batch components if enabled
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises

    return t, xt, ut


def calculate_auto_regularization(a, b, M, tolerance=1e-3):
    """
    Two-step automatic tuning of regularization parameters for **unbalanced Sinkhorn** OT.
    
    Step 1: Tune entropic regularization (`reg`) using the elbow rule on transport cost.
    Step 2: Tune mass regularization (`reg_m`) using grid search + elbow rule on transport cost.
    
    Args:
        a: Source marginal vector of shape (n_samples_0,) (uniform marginal: a = 1/n_samples_0 * ones).
        b: Target marginal vector of shape (n_samples_1,) (uniform marginal: b = 1/n_samples_1 * ones).
        M: Cost matrix of shape (n_samples_0, n_samples_1) (e.g., squared Euclidean distance).
        tolerance: Tolerance for judging OT plan stability (unused here).
    
    Returns:
        reg: Tuned entropic regularization coefficient.
        reg_m: Tuned mass regularization coefficient.
    """
    # Fallback to default parameters if cost matrix is empty
    if M.size == 0:
        print("Cost matrix is empty – fall back to defaults.")
        return 1e-5, 1.0
    
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Step 1: Tune `reg` (entropic regularization) using elbow rule on transport cost
    # ------------------------------------------------------------------
    # Lists to store regularization values and corresponding transport costs (for analysis)
    reg_list, loss_list = [], []
    # Fixed initial mass regularization for tuning `reg`
    tau_initial = 50.0
    # Default `reg` if tuning fails
    reg = 10

    # Context manager to convert specific OT warnings into exceptions (for error handling)
    class catch_specific_warning:
        def __init__(self, message, category, module):
            self.message = message  # Warning message pattern to catch
            self.category = category  # Warning category (e.g., UserWarning)
            self.module = module  # Module generating the warning (e.g., ot.unbalanced._sinkhorn)

        def __enter__(self):
            # Save current warning filters to restore later
            self.original_filters = warnings.filters.copy()
            # Set filter to raise an exception for the specific warning
            warnings.filterwarnings(
                "error",
                message=self.message,
                category=self.category,
                module=self.module
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original warning filters
            warnings.filters = self.original_filters
            return False  # Do not suppress exceptions

    # Helper function to check if an OT plan is numerically stable
    def is_stable(ot_plan, a):
        """
        Check if an OT plan is stable (finite values, positive total mass, non-negative weights).
        
        Args:
            ot_plan: OT plan matrix of shape (n_samples_0, n_samples_1).
            a: Source marginal vector (unused here; preserved for consistency).
        
        Returns:
            stable: Boolean indicating if the OT plan is stable.
        """
        return (
                np.all(np.isfinite(ot_plan)) and  # All values are finite
                np.sum(ot_plan) > 1e-8 and        # Total mass is non-negligible
                np.all(ot_plan >= 0)              # All transport weights are non-negative
        )

    # ---------- Round 1: Coarse search for `reg` (large step size) ----------
    eps1_min, eps1_step, eps_max = 5e-2, 2e-2, 10  # Search range: 0.05 to 10, step 0.02
    eps = eps1_min
    found_1st = None  # Store the first stable `reg` value

    while eps <= eps_max:
        try:
            # Catch numerical warnings from unbalanced Sinkhorn and treat as exceptions
            with catch_specific_warning(
                    message="Numerical errors at iteration.*",
                    category=UserWarning,
                    module="ot.unbalanced._sinkhorn"):
                # Compute unbalanced OT plan with current `eps` (reg) and fixed tau_initial (reg_m)
                ot_plan = ot.unbalanced.sinkhorn_unbalanced(
                    a=a,
                    b=b,
                    M=M,
                    reg=eps,
                    reg_m=[tau_initial, np.inf]  # reg_m: [mass regularization, upper bound]
                )
                # Check if the OT plan is stable
                if is_stable(ot_plan, a):
                    found_1st = eps
                    break  # Exit loop if stable `reg` is found
                else:
                    raise RuntimeError("unstable plan")  # Force retry if plan is unstable
        except (Exception, UserWarning) as exc:
            # Print error and continue searching with next `eps`
            print(f"[Round-1 eps={eps:.3e}]  failed: {type(exc).__name__}: {exc}")
            eps += eps1_step
            continue

    # If no stable `reg` found in Round 1, use default; else refine in Round 2
    if found_1st is None:
        print("No stable eps found, keep reg =", reg)
    else:
        # ---------- Round 2: Fine search for `reg` (small step size) ----------
        eps2_min = found_1st + 1e-3  # Start just above the first stable `reg`
        eps2_step = 1e-3  # Step size: 0.001
        eps = eps2_min
        found_2nd = None  # Store the refined stable `reg` value

        while eps <= eps_max:
            try:
                with catch_specific_warning(
                        message="Numerical errors at iteration.*",
                        category=UserWarning,
                        module="ot.unbalanced._sinkhorn"):
                    # Compute OT plan with refined `eps`
                    ot_plan = ot.unbalanced.sinkhorn_unbalanced(
                        a=a,
                        b=b,
                        M=M,
                        reg=eps,
                        reg_m=[tau_initial, np.inf]
                    )
                    if is_stable(ot_plan, a):
                        found_2nd = eps
                        break
                    else:
                        raise RuntimeError("unstable plan")
            except (Exception, UserWarning) as exc:
                print(f"[Round-2 eps={eps:.3e}]  failed: {type(exc).__name__}: {exc}")
                eps += eps2_step
                continue

        # Update `reg` with refined value (or keep first stable value if refinement fails)
        if found_2nd is not None:
            reg = found_2nd
        else:
            reg = found_1st

    print("Final reg =", reg)

    # ------------------------------------------------------------------
    # Step 2: Tune `reg_m` (mass regularization) using grid search + elbow rule
    # ------------------------------------------------------------------
    # Grid of `reg_m` candidates (40 log-spaced values from 0.01 to 15.85)
    reg_m_cand_2 = np.logspace(-2, 1.2, 40)
    # Lists to store valid `reg_m` values and corresponding transport costs
    reg_m_list, loss_list = [], []

    # Evaluate each `reg_m` candidate with the tuned `reg`
    for reg_m in reg_m_cand_2:
        try:
            # Compute unbalanced OT plan with current `reg_m` and tuned `reg`
            G = pot.unbalanced.sinkhorn_unbalanced(
                a=a,
                b=b,
                M=M,
                reg=reg,
                reg_m=[reg_m, np.inf]
            )
            # Check if the OT plan is valid (finite, non-negligible mass, non-negative)
            if not (np.all(np.isfinite(G)) and G.sum() > 1e-6 and G.min() >= 0):
                continue
            # Compute transport cost (sum of G[i,j] * M[i,j])
            transport_loss = float((G * M).sum())
            # Store valid `reg_m` and corresponding cost
            reg_m_list.append(reg_m)
            loss_list.append(transport_loss)
        except Exception:
            # Skip invalid `reg_m` candidates
            continue

    # ------------------------------------------------------------------
    # Apply elbow rule to select `reg_m` (point of maximum distance from the line connecting first and last points)
    # ------------------------------------------------------------------
    # Fallback to default if too few valid `reg_m` candidates
    if len(reg_m_list) < 4:
        best_reg_m = reg_m_list[np.argmin(loss_list)] if reg_m_list else 1.0
    else:
        # Convert lists to numpy arrays for computation
        x = np.array(reg_m_list, dtype=float)  # `reg_m` values (x-axis)
        y = np.array(loss_list, dtype=float)   # Transport costs (y-axis)

        # Normalize x and y to [0,1] for consistent distance calculation
        x_norm = (x - x[0]) / (x[-1] - x[0] + 1e-12)  # Avoid division by zero
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

        # Line connecting the first and last normalized points
        y0, y1 = y_norm[0], y_norm[-1]
        line_y = y0 + (y1 - y0) * x_norm  # Expected y for each x_norm on the line

        # Vector representing the line (from first to last point)
        line_vec = np.array([1.0, y1 - y0])
        line_len = np.linalg.norm(line_vec)  # Length of the line vector

        # Compute perpendicular distance from each point to the line
        distances = np.abs(np.cross(line_vec, np.column_stack([x_norm - x_norm[0],
                                                               y_norm - y_norm[0]]))) / line_len

        # Select `reg_m` with the maximum distance (elbow point)
        best_reg_m = reg_m_list[np.argmax(distances)]
        print(f"Elbow chosen reg_m = {best_reg_m:.6f}")

    return reg, best_reg_m



def compute_uot_plans(
    X, 
    t_train, 
    use_mini_batch_uot=False, 
    chunk_size=1000,
    alpha_regm=1.0,
    reg_strategy: str = "max_over_time"  
):
    """
    Enhanced UOT plan computation with flexible regularization strategies (per-time or max-over-time).
    
    Args:
        X (list/np.ndarray): List or array where each element X[i] is the data matrix at time t_train[i],
                             shape (num_samples, num_features).
        t_train (np.ndarray/torch.Tensor): Time points corresponding to X.
        use_mini_batch_uot (bool, optional): Whether to use mini-batch mode. Defaults to False.
        chunk_size (int, optional): Samples per chunk in mini-batch mode. Defaults to 1000.
        alpha_regm (float, optional): Scaling factor for the unbalance regularization parameter (reg_m).
                                      Defaults to 1.0.
        reg_strategy (str, optional): Strategy for choosing regularization parameters:
                                      - "per_time": Compute parameters independently for each time step.
                                      - "max_over_time": Precompute parameters for all time steps, use the maximum.
                                      Defaults to "max_over_time".
    
    Returns:
        list: uot_plans - List of UOT matrices (one per consecutive time pair).
        list: sampling_info_plans - List of sampling metadata (sub-plans, indices) for mini-batch; None for full-batch.
    """
    # Validate regularization strategy input
    if reg_strategy not in ["per_time", "max_over_time"]:
        raise ValueError(f"reg_strategy must be 'per_time' or 'max_over_time', got {reg_strategy}")
    
    uot_plans = []
    sampling_info_plans = []
    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Global regularization parameters (used for "max_over_time" strategy)
    global_reg = None
    global_reg_m = None

    # Precompute regularization parameters for all time steps (if using "max_over_time" strategy)
    if reg_strategy == "max_over_time":
        reg_list = []  # Store entropy regularization parameters for each time step
        reg_m_list = []  # Store unbalance regularization parameters for each time step
        print("Precomputing regularization parameters for all time points...")
        
        # Iterate all time steps to collect regularization parameters
        for i in range(len(t_train) - 1):
            X_source, X_target = X[i], X[i + 1]
            n_source, n_target = X_source.shape[0], X_target.shape[0]
            # Uniform marginals
            a = np.ones(n_source)
            b = np.ones(n_target)

            # Compute and normalize cost matrix
            cost_matrix = pot.dist(X_source, X_target)
            if cost_matrix.mean() > 100:
                cost_matrix = cost_matrix / cost_matrix.max()

            # Mini-batch mode: Use first chunk to compute parameters (consistent with mini-batch logic)
            if use_mini_batch_uot:
                group_number = n_source // chunk_size + 1
                # Shuffle and split indices to get first chunk
                source_perm = np.arange(n_source)
                np.random.shuffle(source_perm)
                target_perm = np.arange(n_target)
                np.random.shuffle(target_perm)
                source_indices_groups = np.array_split(source_perm, group_number)
                target_indices_groups = np.array_split(target_perm, group_number)
                
                first_src_idx = source_indices_groups[0]
                first_tgt_idx = target_indices_groups[0]
                
                # Extract first chunk's data
                sub_cost_matrix = cost_matrix[np.ix_(first_src_idx, first_tgt_idx)]
                sub_a = a[first_src_idx]
                sub_b = b[first_tgt_idx]
                
                # Compute parameters for first chunk
                reg, reg_m = calculate_auto_regularization(sub_a, sub_b, sub_cost_matrix)
                print(f"Time point {i} (mini-batch first sub-block): reg={reg}, reg_m={reg_m}")
            
            # Full-batch mode: Compute parameters using entire data
            else:
                reg, reg_m = calculate_auto_regularization(a, b, cost_matrix)
                print(f"Time point {i} (full matrix): reg={reg}, reg_m={reg_m}")

            # Save parameters for current time step
            reg_list.append(reg)
            reg_m_list.append(reg_m)

        # Set global parameters to the maximum of all time steps (ensures consistency across time)
        global_reg = max(reg_list)
        global_reg_m = max(reg_m_list)
        print(f"Max over all time points - reg: {global_reg}, reg_m: {global_reg_m}")

    # Compute UOT plans for each consecutive time step
    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):
        X_source, X_target = X[i], X[i + 1]
        n_source, n_target = X_source.shape[0], X_target.shape[0]
        # Uniform marginals
        a = np.ones(n_source)
        b = np.ones(n_target)

        # Compute and normalize cost matrix
        cost_matrix = pot.dist(X_source, X_target)
        if cost_matrix.mean() > 100:
            cost_matrix = cost_matrix / cost_matrix.max()

        # Set regularization parameters based on strategy
        if reg_strategy == "per_time":
            # Full-batch: Compute parameters for current time step
            if not use_mini_batch_uot:
                reg, reg_m_1 = calculate_auto_regularization(a, b, cost_matrix)
                print(f"Time point {i} (per_time): reg={reg}, reg_m={reg_m_1}")
            # Mini-batch: Parameters will be computed using first chunk later
            else:
                reg, reg_m_1 = None, None
        else:  # "max_over_time" strategy: Use precomputed global parameters
            reg, reg_m_1 = global_reg, global_reg_m
            print(f"Time point {i} (max_over_time): using reg={reg}, reg_m={reg_m_1}")

        # Full-batch UOT computation
        if not use_mini_batch_uot:
            # Convert data to tensors and move to device
            a_cuda = torch.from_numpy(a).float().to(device)
            b_cuda = torch.from_numpy(b).float().to(device)
            cost_matrix_cuda = torch.from_numpy(cost_matrix).float().to(device)
            # Scale unbalance regularization parameter with alpha_regm
            reg_m = reg_m_1 * alpha_regm
            print("final reg_m", reg_m_1)  # Note: Prints original reg_m_1 (before scaling by alpha_regm)
            
            # Solve unbalanced Sinkhorn problem
            G = pot.unbalanced.sinkhorn_unbalanced(a_cuda, b_cuda, cost_matrix_cuda, reg, [reg_m, np.inf])
            # Move UOT plan back to CPU
            G = G.cpu().numpy()
            
            # Validate marginal constraints
            assert (np.abs(G.sum(axis=0) - b) < 1).all(), "Map does not meet marginal constraints"
            sampling_info_plans.append(None)
        
        # Mini-batch UOT computation
        else:
            # Calculate number of chunks
            group_number = n_source // chunk_size + 1
            # Initialize empty full UOT plan
            G = np.zeros((n_source, n_target))
            # Shuffle indices for random chunking
            source_perm = np.arange(n_source)
            np.random.shuffle(source_perm)
            target_perm = np.arange(n_target)
            np.random.shuffle(target_perm)
            # Split indices into chunks
            source_indices_groups = np.array_split(source_perm, group_number)
            target_indices_groups = np.array_split(target_perm, group_number)

            # Store sub-plans for current time step
            uot_sub_plans = []
            
            # Process each chunk
            for src_idx, tgt_idx in zip(source_indices_groups, target_indices_groups):
                # Extract chunk-specific data
                sub_cost_matrix = cost_matrix[np.ix_(src_idx, tgt_idx)]
                sub_a = a[src_idx]
                sub_b = b[tgt_idx]

                # "per_time" strategy: Compute parameters using first chunk (reuse for others)
                if reg_strategy == "per_time" and len(uot_sub_plans) == 0:
                    reg, reg_m_1 = calculate_auto_regularization(sub_a, sub_b, sub_cost_matrix)
                    print(f"Time point {i} sub-block 0 (per_time): reg={reg}, reg_m={reg_m_1}")

                # Convert chunk data to tensors and move to device
                sub_a_cuda = torch.from_numpy(sub_a).float().to(device)
                sub_b_cuda = torch.from_numpy(sub_b).float().to(device)
                sub_cost_matrix = torch.from_numpy(sub_cost_matrix).float().to(device)
                # Scale unbalance regularization parameter
                reg_m = reg_m_1 * alpha_regm
                print("final reg_m", reg_m)
                
                # Solve UOT for the chunk
                G_sub = pot.unbalanced.sinkhorn_unbalanced(
                    sub_a_cuda, sub_b_cuda, sub_cost_matrix, reg, [reg_m , np.inf]
                )
                # Move sub-plan back to CPU
                G_sub = G_sub.cpu().numpy()
                # Assign sub-plan to full UOT plan
                G[np.ix_(src_idx, tgt_idx)] = G_sub

                # Validate chunk marginal constraints (scaled by chunk size)
                assert (np.abs(G_sub.sum(axis=0) - sub_b) < 0.1 * chunk_size).all(), \
                    "Sub-map does not meet marginal constraints"
                # Save sub-plan (float32 for memory efficiency)
                uot_sub_plans.append(G_sub.astype(np.float32))

            # Collect sampling metadata for current time step
            sampling_info = {
                'sub_plans': uot_sub_plans,
                'source_groups': source_indices_groups,
                'target_groups': target_indices_groups
            }
            sampling_info_plans.append(sampling_info)

        # Save UOT plan for current time step
        uot_plans.append(G)

    return uot_plans, sampling_info_plans


def sample_from_ot_plan(
        ot_plan: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
        batch_size: int,
        sampling_info: dict = None
):
    """
    Samples pairs of source (x0) and target (x1) points from an optimal transport (OT) plan.
    
    Args:
        ot_plan (np.ndarray): OT/UOT matrix of shape (num_source_samples, num_target_samples),
                              where ot_plan[i,j] is the transport weight between x0[i] and x1[j].
        x0 (np.ndarray): Source data matrix (time t) of shape (num_source_samples, num_features).
        x1 (np.ndarray): Target data matrix (time t+1) of shape (num_target_samples, num_features).
        batch_size (int): Number of sample pairs to generate.
        sampling_info (dict, optional): Metadata from mini-batch UOT (sub-plans, indices groups).
                                        Use if ot_plan was computed in mini-batch mode. Defaults to None.
    
    Returns:
        tuple: (x0_sampled, x1_sampled, i_indices, j_indices)
            - x0_sampled: Sampled source points (shape: (batch_size, num_features)).
            - x1_sampled: Sampled target points (shape: (batch_size, num_features)).
            - i_indices: Indices of sampled points in x0 (shape: (batch_size,)).
            - j_indices: Indices of sampled points in x1 (shape: (batch_size,)).
            Returns empty arrays if total transport mass is too small (no valid samples).
    """
    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sampling from full-batch OT plan (no mini-batch metadata)
    if sampling_info is None:
        # Convert OT plan to PyTorch tensor and move to device
        pi_cuda = torch.from_numpy(ot_plan.astype(np.float32)).to(device)
        # Compute row sums (total transport mass from each source sample)
        row_sums = pi_cuda.sum(axis=1)
        # Total transport mass across all source samples
        total_sum = row_sums.sum()
        
        # Return empty arrays if total mass is negligible (no valid transport)
        if total_sum < 1e-9: 
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Probabilities for sampling source indices (proportional to row sums)
        row_probs = row_sums / total_sum
        # Sample source indices (with replacement)
        i_samples = torch.multinomial(row_probs, num_samples=batch_size, replacement=True)
        
        # Extract rows of OT plan corresponding to sampled source indices
        selected_rows = pi_cuda[i_samples]
        # Row sums of the selected OT plan rows
        selected_row_sums = row_sums[i_samples]
        # Conditional probabilities for target indices (given sampled source indices)
        conditional_probs = selected_rows / (selected_row_sums.unsqueeze(1) + 1e-12)  # Add epsilon to avoid division by zero
        # Sample target indices (one per sampled source index)
        j_samples = torch.multinomial(conditional_probs, num_samples=1).squeeze(1)
        
        # Move sampled indices back to CPU and convert to numpy arrays
        i, j = i_samples.cpu().numpy(), j_samples.cpu().numpy()
    
    # Sampling from mini-batch OT plan (use metadata to handle sub-plans)
    else:
        # Extract mini-batch metadata: sub-plans, source/target index groups
        G_subs = sampling_info['sub_plans']
        source_indices_groups = sampling_info['source_groups']
        target_indices_groups = sampling_info['target_groups']
        
        # Compute total transport mass for each sub-plan (chunk)
        block_masses = [g.sum() for g in G_subs]
        total_mass = sum(block_masses)
        
        # Return empty arrays if total mass is negligible
        if total_mass < 1e-9: 
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Probabilities for sampling sub-plans (proportional to sub-plan mass)
        block_probs = torch.tensor(block_masses, dtype=torch.float32, device=device) / total_mass
        # Sample which sub-plans to use for each batch sample
        sampled_group_indices = torch.multinomial(block_probs, num_samples=batch_size, replacement=True)
        
        # Move sub-plans and index groups to GPU for faster processing
        G_subs_gpu = [torch.from_numpy(g).to(device) for g in G_subs]
        source_indices_gpu = [torch.from_numpy(idx).to(device) for idx in source_indices_groups]
        target_indices_gpu = [torch.from_numpy(idx).to(device) for idx in target_indices_groups]
        
        # Get unique sub-plans to sample from and their sample counts
        unique_groups, counts = torch.unique(sampled_group_indices, return_counts=True)
        # Initialize arrays to store final sampled indices
        final_i_samples = torch.empty(batch_size, dtype=torch.long, device=device)
        final_j_samples = torch.empty(batch_size, dtype=torch.long, device=device)
        
        # Sample from each unique sub-plan
        for group_idx, count in zip(unique_groups, counts):
            # Get current sub-plan and its row sums (transport mass per source sample in the sub-plan)
            g_sub = G_subs_gpu[group_idx]
            sub_row_sums = g_sub.sum(axis=1)
            
            # Skip if sub-plan has negligible mass
            if sub_row_sums.sum() < 1e-9: 
                continue
            
            # Probabilities for sampling source indices within the sub-plan
            sub_row_probs = sub_row_sums / sub_row_sums.sum()
            # Sample local source indices (within the sub-plan)
            i_local = torch.multinomial(sub_row_probs, num_samples=count.item(), replacement=True)
            
            # Extract sub-plan rows for sampled local source indices
            selected_sub_rows = g_sub[i_local]
            selected_sub_row_sums = sub_row_sums[i_local]
            # Conditional probabilities for target indices (within the sub-plan)
            sub_cond_probs = selected_sub_rows / (selected_sub_row_sums.unsqueeze(1) + 1e-12)
            # Sample local target indices (within the sub-plan)
            j_local = torch.multinomial(sub_cond_probs, num_samples=1).squeeze(1)
            
            # Map local sub-plan indices to global indices (in original x0/x1)
            global_i = source_indices_gpu[group_idx][i_local]
            global_j = target_indices_gpu[group_idx][j_local]
            
            # Assign global indices to the final sample array (using mask for correct position)
            mask = (sampled_group_indices == group_idx)
            final_i_samples[mask] = global_i.long()  # Convert to Long type for index compatibility
            final_j_samples[mask] = global_j.long()
        
        # Move final indices back to CPU and convert to numpy arrays
        i, j = final_i_samples.cpu().numpy(), final_j_samples.cpu().numpy()

    # Return empty arrays if no valid indices were sampled
    if i.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Extract sampled source and target points using the sampled indices
    return x0[i], x1[j], i, j


def get_batch_uot_fm(FM, X, t_train, batch_size, uot_plans, sampling_info_plans):
    """
    Generates training batches for a Flow Matching (FM) model using precomputed UOT plans.
    
    Args:
        FM: Flow Matching model instance (must have a 'sample_location_and_conditional_flow' method).
        X (list/np.ndarray): List of data matrices, where X[t] is data at time t_train[t].
        t_train (torch.Tensor): Time points corresponding to X (shape: (num_time_steps,)).
        batch_size (int): Number of samples per batch.
        uot_plans (list): List of UOT plans (one per consecutive time pair, from compute_uot_plans).
        sampling_info_plans (list): List of sampling metadata for UOT plans (from compute_uot_plans).
    
    Returns:
        tuple: (ts, xts, uts, gts, massts, noises)
            - ts: Sampled time points (scaled to original time, shape: (total_samples,)).
            - xts: Sampled intermediate points (from FM, shape: (total_samples, num_features)).
            - uts: Scaled conditional flows (divided by time step, shape: (total_samples, num_features)).
            - gts: Scaled target flows (divided by time step, shape: (total_samples, num_features)).
            - massts: Transport weights (from UOT plan, shape: (total_samples,)).
            - noises: Noise vectors used in FM sampling (shape: (total_samples, num_features)).
    """
    # Initialize lists to store batch components (one per time step)
    ts = []
    xts = []
    uts = []
    gts = []
    massts = []
    noises = []

    # Use GPU 0 if available, else CPU (explicitly specifies GPU 0 for multi-GPU setups)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Iterate over consecutive time steps to generate samples
    for t in range(t_train.size(0)-1):
        # Get UOT plan and sampling metadata for current time step (t -> t+1)
        uot_plan = uot_plans[t]
        sampling_info = sampling_info_plans[t]

        # Sample source (X[t]) and target (X[t+1]) pairs from the UOT plan
        x0, x1, idx_0, idx_1 = sample_from_ot_plan(uot_plan, X[t], X[t + 1], batch_size, sampling_info)

        # Uncomment the following block to visualize sampled source-target pairs (debugging)
        # plt.figure(figsize=(10, 10))
        # plt.scatter(x0[:, 0], x0[:, 1], color='red')
        # plt.scatter(x1[:, 0], x1[:, 1], color='blue')
        # # Draw lines connecting each x0 to its corresponding x1
        # for xi, xj in zip(x0, x1):
        #     plt.plot([xi[0], xj[0]], [xi[1], xj[1]], color='gray', linewidth=0.5, alpha=0.5)
        # plt.savefig(f'x0_x1_{t}.png')
        # plt.show()

        # Convert sampled points to PyTorch tensors and move to target device
        x0 = torch.from_numpy(x0).float().to(device)
        x1 = torch.from_numpy(x1).float().to(device)

        # Time difference between current and next time step (delta_t = t_{t+1} - t_t)
        delta_t = t_train[t + 1] - t_train[t]

        # Sample intermediate time points, flows, and noise from the FM model
        # Uses UOT plan and sampled indices to condition the flow
        t_sampled, xt, ut, eps, gt_samp, weights = FM.sample_location_and_conditional_flow(x0, x1, uot_plan=uot_plan, idx_0=idx_0,idx_1=idx_1, return_noise=True)

        # Scale time points to original time (t_sampled is in [0,1], so multiply by delta_t and add t_train[t])
        ts.append(t_sampled * delta_t + t_train[t])
        # Save intermediate points
        xts.append(xt)
        # Scale conditional flow by 1/delta_t (to match original time scale)
        uts.append(ut / delta_t)
        # Scale target flow by 1/delta_t (to match original time scale)
        gts.append(gt_samp / delta_t)
        # Save transport weights from UOT plan
        massts.append(weights)
        # Save noise vectors used in FM sampling
        noises.append(eps)

    # Optional: Compute mean of scaled target flows per time step (for debugging/analysis)
    # gt_means = [(gt_samp / delta_t).mean().item() for gt_samp in gts]
    # print("mean of gt_samp/delta_t per batch:", gt_means)

    # Concatenate lists into single tensors (combine samples from all time steps)
    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(gts), torch.cat(massts), torch.cat(noises)