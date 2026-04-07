"""
models/contrastive_loss.py — Supervised Contrastive Loss + Combined Loss
Reference: Khosla et al. (NeurIPS 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    SupConLoss: pulls same-class embeddings together,
    pushes different-class embeddings apart.
    """
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) L2-normalized embeddings
            labels:   (B,) integer class labels
        Returns:
            scalar loss
        """
        device = features.device
        B = features.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity matrix (B, B)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Positive mask: same class, excluding self
        labels = labels.contiguous().view(-1, 1)
        mask_positive = (labels == labels.T).float().to(device)
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        mask_positive.masked_fill_(mask_self, 0)

        # Log-sum-exp numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_exp = torch.exp(sim_matrix - sim_max.detach())

        # Zero out self-similarity in denominator
        sim_exp_no_self = sim_exp.masked_fill(mask_self, 0)
        log_prob = (sim_matrix - sim_max.detach()) - \
                   torch.log(sim_exp_no_self.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs only
        n_positives = mask_positive.sum(dim=1)
        valid = n_positives > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1)[valid] / \
                            n_positives[valid]

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, inputs, targets):
        # Calculate vanilla CE (no alpha) for pt calculation
        ce_loss_raw = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_raw)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Gather alpha for each target in batch
            at = self.alpha.gather(0, targets.data)
            focal_weight = focal_weight * at
            
        focal_loss = focal_weight * ce_loss_raw
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Focal Loss (classification) + SupCon (representation) loss.
    Total = L_Focal + λ * L_SupCon
    """
    def __init__(self, temperature=0.1, lambda_con=0.5, class_weights=None):
        super().__init__()
        self.ce_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        self.con_loss = SupervisedContrastiveLoss(temperature)
        self.lambda_con = lambda_con

    def forward(self, logits, features, labels):
        """
        Args:
            logits:   (B, C) classification logits
            features: (B, D) L2-normalized contrastive embeddings
            labels:   (B,) integer class labels
        Returns:
            total_loss, ce_value, con_value
        """
        l_ce = self.ce_loss(logits, labels)
        l_con = self.con_loss(features, labels)
        total = l_ce + self.lambda_con * l_con
        return total, l_ce.item(), l_con.item()
