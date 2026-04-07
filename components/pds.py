"""
components/pds.py — Prototype Disagreement Signal
§III-D of the FedPDG paper.

Computes per-client prototype divergence and uses Gini-based discrimination
to distinguish zero-day events from Byzantine attacks.
"""
import numpy as np
import torch


class PrototypeDivergenceSignal:
    """
    Three formal definitions:

    Def 1 — Per-Class Divergence:
        δ_c(k,t) = ||p_c^(k,t) - p̄_c^(t)||_2

    Def 2 — PDS Score:
        PDS_k^(t) = Σ_c δ_c(k,t)

    Def 3 — Gini Discrimination:
        G_k = 1 - Σ_c (δ_c / Σδ)²
        High Gini (spread) → Zero-day (divergence on few classes)
        Low  Gini (uniform) → Byzantine (divergence on all classes)
    """

    def __init__(self, config):
        self.cfg = config
        self.history = {}       # client_id → list of PDS scores
        self.round_stats = []   # per-round μ and σ of PDS

    def compute_pds(self, client_id, local_prototypes, global_prototypes):
        """
        Compute PDS score for a single client.

        Args:
            client_id: int
            local_prototypes:  dict {class_id: tensor(D,)}
            global_prototypes: dict {class_id: tensor(D,)}

        Returns:
            pds_score: float (aggregate divergence)
            per_class_div: dict {class_id: float}
        """
        common = set(local_prototypes.keys()) & set(global_prototypes.keys())
        if not common:
            return 0.0, {}

        per_class_div = {}
        for cls in common:
            local_p = local_prototypes[cls]
            global_p = global_prototypes[cls]

            if isinstance(local_p, torch.Tensor):
                local_p = local_p.detach().cpu().numpy()
            if isinstance(global_p, torch.Tensor):
                global_p = global_p.detach().cpu().numpy()

            div = np.linalg.norm(local_p - global_p, ord=2)
            per_class_div[cls] = float(div)

        pds_score = float(np.sum(list(per_class_div.values())))

        # Track history
        if client_id not in self.history:
            self.history[client_id] = []
        self.history[client_id].append(pds_score)

        return pds_score, per_class_div

    def compute_gini_index(self, per_class_divergences):
        """
        Gini impurity of the divergence profile.

        High Gini → divergence concentrated in FEW classes (sparse) → Zero-day
        Low  Gini → divergence spread across ALL classes (uniform) → Byzantine
        """
        if not per_class_divergences:
            return 0.0

        divs = np.array(list(per_class_divergences.values()))
        total = divs.sum()
        if total < 1e-10:
            return 0.0

        props = divs / total
        gini = 1.0 - np.sum(props ** 2)
        return float(gini)

    def classify_client(self, pds_score, gini_index, mu_pds, sigma_pds):
        """
        Classify client as normal, zero-day, or Byzantine.

        Uses adaptive z-score thresholding:
            Flagged if PDS > μ + γ·σ

        Then Gini separates:
            High Gini (sparse divergence) → zero-day
            Low  Gini (uniform divergence) → Byzantine
        """
        gamma = self.cfg.PDS_GAMMA

        if sigma_pds > 0 and pds_score > mu_pds + gamma * sigma_pds:
            # Anomalous divergence detected
            if gini_index > self.cfg.GINI_SPLIT_THRESHOLD:
                return 'zeroday'    # sparse → localized class shift
            else:
                return 'byzantine'  # uniform → all prototypes corrupted
        return 'normal'

    def evaluate_all_clients(self, all_local_prototypes, global_prototypes):
        """
        Run PDS evaluation for all clients in this round.

        Args:
            all_local_prototypes: dict {client_id: {class_id: tensor}}
            global_prototypes:    dict {class_id: tensor}

        Returns:
            dict {client_id: {pds_score, gini_index, status,
                              per_class_div, trust_weight}}
        """
        # Compute PDS for all clients
        pds_scores = {}
        per_class_divs = {}
        for cid, local_protos in all_local_prototypes.items():
            pds, per_cls = self.compute_pds(cid, local_protos, global_prototypes)
            pds_scores[cid] = pds
            per_class_divs[cid] = per_cls

        # Compute running statistics for adaptive thresholding
        all_pds = list(pds_scores.values())
        mu_pds = float(np.mean(all_pds)) if all_pds else 0.0
        sigma_pds = float(np.std(all_pds)) if all_pds else 1.0
        self.round_stats.append({'mu': mu_pds, 'sigma': sigma_pds})

        # Classify each client
        results = {}
        for cid in all_local_prototypes:
            pds = pds_scores[cid]
            per_cls = per_class_divs[cid]
            gini = self.compute_gini_index(per_cls)
            status = self.classify_client(pds, gini, mu_pds, sigma_pds)

            # Trust weight for DWA
            trust = self._compute_trust_weight(pds, status, mu_pds, sigma_pds)

            results[cid] = {
                'pds_score': pds,
                'gini_index': gini,
                'status': status,
                'per_class_div': per_cls,
                'trust_weight': trust,
            }
            print(f"  PDS Client {cid:2d}: PDS={pds:.4f} | "
                  f"Gini={gini:.4f} | Status={status:9s} | "
                  f"Trust={trust:.3f}")

        return results

    def _compute_trust_weight(self, pds_score, status, mu_pds, sigma_pds):
        """Compute trust weight for DWA aggregation."""
        if status == 'byzantine':
            return 0.0  # fully excluded
        elif status == 'zeroday':
            # Reduced but non-zero weight — novel info still valuable
            return max(0.1, 1.0 - (pds_score - mu_pds) /
                       (sigma_pds + 1e-8) * 0.2)
        else:
            # Normal: inverse PDS → lower divergence = higher trust
            # Adaptive temperature schedule (Prompt 2):
            # τ is inversely proportional to variance (σ_pds). 
            # Higher variance (Byzantine activity) → sharper softmax.
            lam_base = self.cfg.DWA_LAMBDA
            lam_adaptive = lam_base * max(1.0, sigma_pds)
            # Clamp to prevent trust collapse (User suggested floor)
            lam_clamped = min(max(lam_adaptive, 0.1), 5.0)
            return float(np.exp(-lam_clamped * pds_score))

    def get_pds_history(self):
        """Return full PDS history for visualization."""
        return self.history
