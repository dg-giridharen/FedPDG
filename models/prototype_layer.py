"""
models/prototype_layer.py — Class Prototype Management
Maintains per-class prototypes as EMA of embeddings.
Used by PDS for divergence computation and for nearest-prototype classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PrototypeLayer(nn.Module):
    """
    Maintains class prototypes as running averages of embeddings.
    Prototypes are the KEY signal for PDS, DWA, and APS.
    """
    def __init__(self, embed_dim, num_classes, momentum=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.momentum = momentum

        # Prototype memory: (num_classes, embed_dim)
        self.register_buffer('prototypes',
                             torch.zeros(num_classes, embed_dim))
        self.register_buffer('prototype_counts',
                             torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('initialized',
                             torch.zeros(num_classes, dtype=torch.bool))

    def update_prototypes(self, embeddings, labels):
        """
        EMA update of prototypes using current batch embeddings.
        Called during each local training step.
        """
        with torch.no_grad():
            for cls in labels.unique():
                cls_id = cls.item()
                if cls_id >= self.num_classes:
                    continue
                mask = (labels == cls_id)
                cls_emb = embeddings[mask].mean(dim=0)

                if not self.initialized[cls_id]:
                    self.prototypes[cls_id] = cls_emb
                    self.initialized[cls_id] = True
                else:
                    self.prototypes[cls_id] = (
                        self.momentum * self.prototypes[cls_id] +
                        (1 - self.momentum) * cls_emb
                    )
                self.prototype_counts[cls_id] += mask.sum()

    def get_prototypes(self, only_initialized=True):
        """Return dict of {class_id: prototype_vector}."""
        protos = {}
        for c in range(self.num_classes):
            if not only_initialized or self.initialized[c]:
                protos[c] = self.prototypes[c].clone()
        return protos

    def set_prototypes(self, proto_dict):
        """Set prototypes from external dict (e.g., from server)."""
        for cls_id, proto in proto_dict.items():
            if cls_id < self.num_classes:
                self.prototypes[cls_id] = proto.to(self.prototypes.device)
                self.initialized[cls_id] = True

    def predict_by_prototype(self, embeddings):
        """
        Nearest prototype classifier.
        Returns predicted class IDs and distances to nearest prototype.
        """
        active = self.initialized.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            return (torch.zeros(len(embeddings), dtype=torch.long),
                    torch.zeros(len(embeddings)))

        active_protos = self.prototypes[active]  # (C', D)

        # Cosine similarity
        emb_norm = F.normalize(embeddings, dim=1)
        proto_norm = F.normalize(active_protos, dim=1)
        sims = emb_norm @ proto_norm.T  # (B, C')

        max_sims, pred_local = sims.max(dim=1)
        pred_global = active[pred_local]

        # Distance = 1 - cosine_similarity (for open-set thresholding)
        distances = 1.0 - max_sims
        return pred_global, distances

    def compute_divergence_from_global(self, global_prototypes):
        """
        Compute per-class L2 divergence between local and global prototypes.
        Returns: dict {class_id: divergence_score}
        """
        divergences = {}
        for cls_id, global_proto in global_prototypes.items():
            if cls_id >= self.num_classes or not self.initialized[cls_id]:
                continue
            local_proto = self.prototypes[cls_id]
            div = torch.norm(
                local_proto - global_proto.to(local_proto.device), p=2)
            divergences[cls_id] = div.item()
        return divergences

    def expand(self, new_class_id, new_prototype):
        """
        Expand prototype set for a newly spawned class (APS).
        Grows internal buffers if needed.
        """
        if new_class_id >= self.num_classes:
            # Expand buffers
            n_new = new_class_id - self.num_classes + 1
            new_protos = torch.zeros(
                n_new, self.embed_dim, device=self.prototypes.device)
            new_counts = torch.zeros(
                n_new, dtype=torch.long, device=self.prototype_counts.device)
            new_init = torch.zeros(
                n_new, dtype=torch.bool, device=self.initialized.device)

            self.prototypes = torch.cat([self.prototypes, new_protos], dim=0)
            self.prototype_counts = torch.cat(
                [self.prototype_counts, new_counts], dim=0)
            self.initialized = torch.cat(
                [self.initialized, new_init], dim=0)
            self.num_classes = new_class_id + 1

        self.prototypes[new_class_id] = new_prototype.to(
            self.prototypes.device)
        self.initialized[new_class_id] = True
