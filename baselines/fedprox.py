"""
baselines/fedprox.py — FedProx (Li et al. 2020)
Adds proximal regularization term to handle data heterogeneity.
"""
import torch
from baselines.fedavg import FedAvg


class FedProx:
    """FedProx: FedAvg + proximal term (μ/2)||w - w_g||²."""

    def __init__(self, mu=0.01):
        self.mu = mu

    def proximal_loss(self, local_model, global_model):
        """
        Proximal regularization term.
        Add this to local training loss.
        """
        prox = 0.0
        for (_, lp), (_, gp) in zip(
                local_model.named_parameters(),
                global_model.named_parameters()):
            prox += torch.norm(lp - gp.detach(), p=2) ** 2
        return (self.mu / 2.0) * prox

    def aggregate(self, client_models, client_weights=None):
        """Same aggregation as FedAvg."""
        return FedAvg().aggregate(client_models, client_weights)
