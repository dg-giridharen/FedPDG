"""
baselines/krum.py — Krum & Multi-Krum (Blanchard et al. 2017)
Byzantine-robust aggregation selecting client(s) closest to majority.
"""
import torch
import numpy as np
from baselines.fedavg import FedAvg


class Krum:
    """
    Krum: select the single client whose parameters are
    most similar to the majority. Multi-Krum averages top-m.
    """

    def __init__(self, num_byzantine=2, multi=False):
        self.f = num_byzantine
        self.multi = multi

    def _flatten(self, state_dict):
        """Flatten all parameters into a single vector."""
        return torch.cat([p.flatten().float()
                          for p in state_dict.values()])

    def aggregate(self, client_models):
        """
        Args:
            client_models: dict {client_id: state_dict}
        Returns:
            OrderedDict — aggregated state_dict
        """
        cids = list(client_models.keys())
        models = [client_models[cid] for cid in cids]
        n = len(models)

        if n <= 1:
            return models[0] if models else {}

        k = max(1, n - self.f - 2)
        flat = [self._flatten(m) for m in models]

        # Pairwise L2 distances
        scores = []
        for i in range(n):
            dists = sorted([
                torch.norm(flat[i] - flat[j]).item()
                for j in range(n) if j != i
            ])
            scores.append(sum(dists[:k]))

        if self.multi:
            n_select = max(1, n - self.f)
            selected_idx = np.argsort(scores)[:n_select]
        else:
            selected_idx = [int(np.argmin(scores))]

        selected_models = {cids[i]: models[i] for i in selected_idx}
        return FedAvg().aggregate(selected_models)
