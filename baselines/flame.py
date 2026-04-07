"""
baselines/flame.py — FLAME (Nguyen et al. 2022)
Clustering-based Byzantine defense using HDBSCAN.
"""
import torch
import numpy as np
from collections import OrderedDict, Counter
from baselines.fedavg import FedAvg

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from sklearn.cluster import DBSCAN as HDBSCAN


class FLAME:
    """
    FLAME: HDBSCAN clusters client updates.
    Largest cluster = benign, outliers = Byzantine.
    Adds DP noise after aggregation for privacy.
    """

    def __init__(self, noise_multiplier=0.001):
        self.noise_multiplier = noise_multiplier

    def _flatten(self, state_dict):
        """Flatten model parameters to numpy vector."""
        return torch.cat([p.flatten().float()
                          for p in state_dict.values()]).detach().cpu().numpy()

    def aggregate(self, client_models):
        """
        Args:
            client_models: dict {client_id: state_dict}
        Returns:
            OrderedDict — aggregated state_dict (with DP noise)
        """
        cids = list(client_models.keys())
        models = [client_models[cid] for cid in cids]
        n = len(models)

        if n <= 2:
            return FedAvg().aggregate(client_models)

        # Flatten all client models
        flat_models = np.array([self._flatten(m) for m in models])

        # Cluster using HDBSCAN
        try:
            clusterer = HDBSCAN(
                min_cluster_size=max(2, n // 3),
                min_samples=1,
            )
            cluster_labels = clusterer.fit_predict(flat_models)
        except Exception:
            # Fallback to all-benign assumption
            cluster_labels = np.zeros(n, dtype=int)

        # Find largest cluster (assumed benign)
        valid_labels = [l for l in cluster_labels if l >= 0]
        if len(valid_labels) == 0:
            selected_idx = list(range(n))
        else:
            largest = Counter(valid_labels).most_common(1)[0][0]
            selected_idx = [i for i, l in enumerate(cluster_labels)
                            if l == largest]

        if len(selected_idx) == 0:
            selected_idx = list(range(n))

        excluded = [cids[i] for i in range(n) if i not in selected_idx]
        if excluded:
            print(f"  FLAME: Excluded clients {excluded} (outlier clusters)")

        selected_models = {cids[i]: models[i] for i in selected_idx}
        aggregated = FedAvg().aggregate(selected_models)

        # Add differential privacy noise
        if self.noise_multiplier > 0:
            for key in aggregated:
                noise = torch.randn_like(aggregated[key]) * self.noise_multiplier
                aggregated[key] += noise

        return aggregated
