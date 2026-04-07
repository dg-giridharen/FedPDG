"""
utils/partitioner.py — Non-IID Dirichlet Partitioning + Byzantine Injection
"""
import numpy as np
import torch
import os
import pickle


class DirichletPartitioner:
    """
    Partition dataset into N non-IID client splits
    using Dirichlet distribution.
    α → 0  : extremely non-IID (each client gets 1-2 classes)
    α → ∞  : IID (uniform distribution)
    """
    def __init__(self, num_clients, alpha, seed=42):
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed

    def partition(self, X, y):
        """
        Partition X, y into non-IID client splits.
        Returns: list of (X_client, y_client) tuples
        """
        np.random.seed(self.seed)
        unique_classes = np.unique(y)
        client_data = {i: {'X': [], 'y': []} for i in range(self.num_clients)}

        for cls in unique_classes:
            cls_idx = np.where(y == cls)[0]
            np.random.shuffle(cls_idx)

            # Dirichlet proportions for this class
            proportions = np.random.dirichlet(
                np.repeat(self.alpha, self.num_clients))

            # Split indices proportionally
            proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            splits = np.split(cls_idx, proportions)

            for client_id, split in enumerate(splits):
                if len(split) > 0:
                    client_data[client_id]['X'].append(X[split])
                    client_data[client_id]['y'].append(y[split])

        # Concatenate and shuffle within each client
        clients = []
        for i in range(self.num_clients):
            if client_data[i]['X']:
                Xi = np.concatenate(client_data[i]['X'], axis=0)
                yi = np.concatenate(client_data[i]['y'], axis=0)
                perm = np.random.permutation(len(Xi))
                clients.append((Xi[perm], yi[perm]))

                class_dist = dict(zip(*np.unique(yi, return_counts=True)))
                print(f"  Client {i:2d}: {len(Xi):6d} samples | "
                      f"Classes: {class_dist}")
            else:
                clients.append((np.array([]).reshape(0, X.shape[1]),
                                np.array([], dtype=y.dtype)))
                print(f"  Client {i:2d}:      0 samples | Classes: {{}}")

        return clients

    def save_partitions(self, clients, save_dir, dataset_name):
        """Save partitioned data to disk."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{dataset_name}_alpha{self.alpha}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(clients, f)
        print(f"  Saved partitions → {path}")

    def load_partitions(self, save_dir, dataset_name):
        """Load previously saved partitions."""
        path = os.path.join(save_dir, f'{dataset_name}_alpha{self.alpha}.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)


class ByzantineInjector:
    """
    Inject Byzantine behavior into selected clients.
    Supports: label_flip, sign_flip, gaussian, model_replace.
    """
    def __init__(self, attack_type='label_flip', byzantine_ratio=0.2, seed=42):
        self.attack_type = attack_type
        self.byzantine_ratio = byzantine_ratio
        self.seed = seed

    def select_byzantine_clients(self, num_clients):
        """Randomly select Byzantine clients based on ratio."""
        np.random.seed(self.seed)
        n_byz = max(1, int(num_clients * self.byzantine_ratio))
        byzantine_ids = np.random.choice(
            num_clients, n_byz, replace=False).tolist()
        print(f"  Byzantine clients ({self.attack_type}, "
              f"{self.byzantine_ratio*100:.0f}%): {byzantine_ids}")
        return byzantine_ids

    def poison_labels(self, y, num_classes):
        """Flip labels to random incorrect class."""
        np.random.seed(self.seed)
        y_poisoned = y.copy()
        for i in range(len(y_poisoned)):
            candidates = [c for c in range(num_classes) if c != y_poisoned[i]]
            y_poisoned[i] = np.random.choice(candidates)
        return y_poisoned

    def poison_gradients(self, state_dict):
        """Apply gradient-level poisoning to model parameters."""
        if self.attack_type == 'sign_flip':
            return {k: -v for k, v in state_dict.items()}
        elif self.attack_type == 'gaussian':
            return {k: v + torch.randn_like(v) * 0.1
                    for k, v in state_dict.items()}
        elif self.attack_type == 'model_replace':
            return {k: torch.randn_like(v) for k, v in state_dict.items()}
        return state_dict
