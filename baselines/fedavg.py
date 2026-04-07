"""
baselines/fedavg.py — FedAvg (McMahan et al. 2017)
Standard federated averaging — mandatory baseline.
"""
import torch
from collections import OrderedDict


class FedAvg:
    """Standard weighted average of client model parameters."""

    def aggregate(self, client_models, client_weights=None):
        """
        Args:
            client_models: dict {client_id: state_dict}
            client_weights: dict {client_id: float} or None (uniform)
        Returns:
            OrderedDict — aggregated global model state_dict
        """
        cids = list(client_models.keys())
        if client_weights is None:
            n = len(cids)
            client_weights = {cid: 1.0 / n for cid in cids}

        # Normalize weights
        total = sum(client_weights[cid] for cid in cids)
        norm_w = {cid: client_weights[cid] / total for cid in cids}

        new_global = OrderedDict()
        ref = client_models[cids[0]]
        for key in ref.keys():
            new_global[key] = torch.zeros_like(ref[key], dtype=torch.float32)

        for cid in cids:
            w = norm_w[cid]
            for key in client_models[cid].keys():
                new_global[key] += w * client_models[cid][key].float()

        return new_global
