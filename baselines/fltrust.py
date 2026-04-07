"""
baselines/fltrust.py — FLTrust (Cao et al. 2022)
Server uses a small clean root dataset to compute trust scores.
"""
import torch
import torch.nn.functional as F
from collections import OrderedDict
from baselines.fedavg import FedAvg


class FLTrust:
    """
    FLTrust: Trust score = cosine similarity between
    client update and server's own update on clean data.
    """

    def __init__(self):
        pass

    def compute_server_update(self, global_model, server_data, device,
                              criterion, lr=1e-3, epochs=1):
        """
        Server trains 1 epoch on root data to get reference update.

        Args:
            global_model: nn.Module
            server_data: (X_server, y_server)
            device: torch device
            criterion: loss function
            lr: learning rate

        Returns:
            server_update: dict {key: Δw}
        """
        import copy
        model = copy.deepcopy(global_model).to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        X, y = server_data
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.long).to(device)

        for _ in range(epochs):
            logits, features = model(X_t)
            loss, _, _ = criterion(logits, features, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute update as delta from original
        old_state = global_model.state_dict()
        new_state = model.state_dict()
        server_update = OrderedDict()
        for key in old_state:
            server_update[key] = (new_state[key] - old_state[key]).float()

        return server_update

    def _cosine_sim(self, update1, update2):
        """Cosine similarity between two flattened update vectors."""
        flat1 = torch.cat([p.flatten().float() for p in update1.values()])
        flat2 = torch.cat([p.flatten().float() for p in update2.values()])
        return F.cosine_similarity(
            flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

    def aggregate(self, client_updates, server_update, global_state):
        """
        Trust-weighted aggregation.

        Args:
            client_updates: dict {cid: delta_state_dict}
            server_update:  server's reference delta
            global_state:   current global model state_dict

        Returns:
            new_global_state: OrderedDict
        """
        trust_scores = {}
        for cid, update in client_updates.items():
            ts = max(0, self._cosine_sim(update, server_update))
            trust_scores[cid] = ts

        total = sum(trust_scores.values())
        if total < 1e-10:
            print("  FLTrust: All trust scores ≈ 0, falling back to FedAvg")
            total = len(trust_scores)
            trust_scores = {cid: 1.0 for cid in trust_scores}

        # Normalize trust to sum to 1
        norm_ts = {cid: ts / total for cid, ts in trust_scores.items()}

        # Apply trust-weighted update
        new_global = OrderedDict()
        for key in global_state:
            new_global[key] = global_state[key].float().clone()

        for cid, update in client_updates.items():
            w = norm_ts[cid]
            # Scale update by server update norm for stability
            for key in update:
                server_norm = torch.norm(server_update[key].float())
                client_norm = torch.norm(update[key].float())
                scale = server_norm / (client_norm + 1e-8)
                new_global[key] += w * scale * update[key].float()

        return new_global
