"""
components/dwa.py — Divergence-Weighted Aggregation
§III-E of the FedPDG paper.

The first semantically-aware federated aggregation method that uses
prototype-level agreement rather than raw gradient statistics.
"""
import torch
import numpy as np
from collections import OrderedDict


class DivergenceWeightedAggregation:
    """
    4-step aggregation:
        Step 1: Get trust weights from PDS evaluation
        Step 2: Zero-out Byzantine, reduce zero-day, normalize
        Step 3: Weighted average of client model parameters
        Step 4: Weighted average of client prototypes
    """

    def __init__(self, config):
        self.cfg = config
        self.round_history = []

    def aggregate(self, client_models, client_prototypes,
                  pds_results, global_state):
        """
        Main DWA aggregation.

        Args:
            client_models:     dict {client_id: state_dict}
            client_prototypes: dict {client_id: {class_id: tensor}}
            pds_results:       output of PDS.evaluate_all_clients()
            global_state:      current global model state_dict (fallback)

        Returns:
            new_global_state:      OrderedDict (aggregated model params)
            new_global_prototypes: dict {class_id: tensor}
            stats:                 dict with aggregation metadata
        """
        # ── Step 1: Collect trust weights ─────────────────────────────
        weights = {}
        excluded = []
        zeroday_clients = []

        for cid, result in pds_results.items():
            if result['status'] == 'byzantine':
                weights[cid] = 0.0
                excluded.append(cid)
            else:
                weights[cid] = result['trust_weight']
                if result['status'] == 'zeroday':
                    zeroday_clients.append(cid)

        # ── Step 2: Normalize weights ─────────────────────────────────
        total_weight = sum(weights.values())
        if total_weight < 1e-10:
            print("  DWA WARNING: All clients excluded — "
                  "returning global model unchanged.")
            return (global_state,
                    self._build_default_protos(client_prototypes),
                    {'excluded': excluded, 'zeroday': zeroday_clients,
                     'trust_weights': weights, 'n_trusted': 0})

        norm_weights = {cid: w / total_weight for cid, w in weights.items()}

        print(f"  DWA: Excluded {len(excluded)} Byzantine: {excluded}")
        print(f"  DWA: Zero-day clients (reduced weight): {zeroday_clients}")

        # ── Step 3: Weighted model aggregation ────────────────────────
        new_global = OrderedDict()

        # Get reference for parameter shapes
        ref_cid = next(cid for cid, w in norm_weights.items() if w > 0)
        ref_model = client_models[ref_cid]

        for key in ref_model.keys():
            new_global[key] = torch.zeros_like(
                ref_model[key], dtype=torch.float32)

        for cid, state_dict in client_models.items():
            w = norm_weights.get(cid, 0)
            if w <= 0:
                continue
            for key in state_dict.keys():
                new_global[key] += w * state_dict[key].float()

        # ── Step 4: Update global prototypes ──────────────────────────
        new_global_prototypes = self._aggregate_prototypes(
            client_prototypes, norm_weights)

        stats = {
            'excluded_byzantine': excluded,
            'zeroday_clients': zeroday_clients,
            'trust_weights': {k: round(v, 4) for k, v in norm_weights.items()},
            'n_trusted': sum(1 for w in norm_weights.values() if w > 0),
        }
        self.round_history.append(stats)

        return new_global, new_global_prototypes, stats

    def _aggregate_prototypes(self, client_prototypes, weights):
        """Weighted average of prototypes from trusted clients."""
        global_prototypes = {}
        class_weight_sum = {}

        for cid, protos in client_prototypes.items():
            w = weights.get(cid, 0.0)
            if w <= 0:
                continue

            for cls, proto in protos.items():
                if isinstance(proto, torch.Tensor):
                    proto = proto.cpu().float()
                else:
                    proto = torch.tensor(proto, dtype=torch.float32)

                if cls not in global_prototypes:
                    global_prototypes[cls] = torch.zeros_like(proto)
                    class_weight_sum[cls] = 0.0

                global_prototypes[cls] += w * proto
                class_weight_sum[cls] += w

        # Normalize
        for cls in global_prototypes:
            if class_weight_sum[cls] > 1e-10:
                global_prototypes[cls] /= class_weight_sum[cls]
            # L2 normalize prototype
            norm = torch.norm(global_prototypes[cls])
            if norm > 1e-8:
                global_prototypes[cls] = global_prototypes[cls] / norm

        return global_prototypes

    def _build_default_protos(self, client_prototypes):
        """Fallback: simple average of all prototypes (no weighting)."""
        return self._aggregate_prototypes(
            client_prototypes,
            {cid: 1.0 for cid in client_prototypes})
