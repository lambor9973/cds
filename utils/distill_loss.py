import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def distill_loss_only_direction(outputs, class_means, targets, normalize_offsets=False, eps=1e-8):
    s_feats = outputs["sfeatures"]
    t_feats = outputs["tfeatures"]
    if s_feats.shape[0] == 0:
        return torch.tensor(0.0, devices=s_feats.device, requires_grad=True)

    class_means = torch.from_numpy(class_means).to(device=s_feats.device, dtype=torch.float32)
    relevant_means = class_means[targets]

    offset_s = s_feats - relevant_means
    offset_t = t_feats - relevant_means

    if normalize_offsets:
        offset_s = F.normalize(offset_s, p=2, dim=-1, eps=eps)
        offset_t = F.normalize(offset_t, p=2, dim=-1, eps=eps)
        cos_sim = torch.sum(offset_s * offset_s, dim=-1)
    else:
        cos_sim = F.cosine_similarity(offset_s, offset_t, dim=-1, eps=eps)

    norm_offset_s = torch.norm(offset_s, p=2, dim=-1)
    norm_offset_t = torch.norm(offset_t, p=2, dim=-1)
    both_offsets_are_zero_mask = (norm_offset_s < eps) & (norm_offset_t < eps)
    final_cos_sim = torch.where(both_offsets_are_zero_mask, torch.ones_like(cos_sim), cos_sim)

    direction_loss = 1.0 - final_cos_sim
    return torch.mean(direction_loss)


def LobachevskyRelativeGeomLoss(outputs, old_class_means, device):
    """
    Args:
        outputs (dict): A dictionary containing:
            'tfeatures': Tensor of features from the old model
            'sfeatures': Tensor of features from the new model
            'logits': Logits form the new model
        old_class_means (np.ndarray): A numpy array of the old class means
        model (torch.nn.Module): A PyTorch model
        current_task_targets:
        device:
    """
    t_features = outputs.get('tfeatures')
    s_features = outputs.get('sfeatures')

    if old_class_means is None or old_class_means.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    try:
        old_prototypes = torch.from_numpy(old_class_means).to(device=device, dtype=torch.float32)
    except Exception as e:
        print(f'Error converting old_class_means to tensor')

    t_features_norm = F.normalize(t_features, p=2, dim=1)
    s_features_norm = F.normalize(s_features, p=2, dim=1)
    old_prototypes_norm = F.normalize(old_prototypes, p=2, dim=1)

    R_old = torch.matmul(t_features_norm, old_prototypes_norm.t())
    R_new = torch.matmul(s_features_norm, old_prototypes_norm.t())

    loss = torch.mean(torch.sum((R_new - R_old)**2, dim=1))

    return loss


def features_distill_loss(outputs):
    t_features = outputs.get('tfeatures')
    s_features = outputs.get('sfeatures')
    loss = torch.mean(torch.sum((t_features - s_features)**2, dim=1))
    return loss


def logits_distill_loss(outputs, old_class_means, model, T=4.0):
    model = model.module if isinstance(model, nn.DataParallel) else model

    s_logits = outputs.get('logits')
    t_features = outputs.get('tfeatures')
    t_logits = model.ca_forward(t_features).get('logits')

    N = old_class_means.shape[0]

    soft_teacher = F.softmax(t_logits[:,:N] / T, dim=1)
    log_soft_student = F.log_softmax(s_logits[:,:N] / T, dim=1)

    loss = F.kl_div(
        input=log_soft_student,
        target=soft_teacher,
        reduction='batchmean',
        log_target=False
    ) * (T ** 2)

    return loss


def RelativePositionLoss(outputs, old_class_means, device):
    """
    Relative position loss function focusing on displacement direction consistency.

    Args:
        outputs: dict containing 'tfeatures' and 'sfeatures'
        old_class_means: old class prototype mean matrix [K, D]
        device: computation device
    """
    t_features = outputs.get('tfeatures')
    s_features = outputs.get('sfeatures')

    if old_class_means.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    old_prototypes = torch.from_numpy(old_class_means).to(device).float()

    t_norm = F.normalize(t_features, p=2, dim=1)
    s_norm = F.normalize(s_features, p=2, dim=1)
    p_norm = F.normalize(old_prototypes, p=2, dim=1)

    t_deltas = t_norm.unsqueeze(1) - p_norm.unsqueeze(0)
    s_deltas = s_norm.unsqueeze(1) - p_norm.unsqueeze(0)

    t_deltas_norm = F.normalize(t_deltas, p=2, dim=-1)
    s_deltas_norm = F.normalize(s_deltas, p=2, dim=-1)

    cos_sim = torch.einsum('bkd,bkd->bk', s_deltas_norm, t_deltas_norm)
    loss = 1 - torch.mean(cos_sim)

    return loss


def PoincaresphereLoss(outputs, old_class_means, device):
    """
    Poincare ball model loss.

    Args:
        outputs: dict containing 'tfeatures' and 'sfeatures'
        old_class_means: old class prototype mean matrix [K, D]
        device: computation device
    """
    t_features = outputs.get('tfeatures')
    s_features = outputs.get('sfeatures')

    if old_class_means.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    old_prototypes = torch.from_numpy(old_class_means).to(device).float()

    t_norm = F.normalize(t_features, p=2, dim=1)
    s_norm = F.normalize(s_features, p=2, dim=1)
    p_norm = F.normalize(old_prototypes, p=2, dim=1)

    t_deltas = t_norm.unsqueeze(1) - p_norm.unsqueeze(0)
    s_deltas = s_norm.unsqueeze(1) - p_norm.unsqueeze(0)

    t_deltas_norm = F.normalize(t_deltas, p=2, dim=-1)
    s_deltas_norm = F.normalize(s_deltas, p=2, dim=-1)

    cos_sim = torch.einsum('bkd,bkd->bk', s_deltas_norm, t_deltas_norm)
    loss = 1 - torch.mean(cos_sim)

    return loss


class HyperbolicOps:
    """Hyperbolic geometry operations (Poincare ball model)"""

    @staticmethod
    def expmap0(v, c=1.0):
        """Map Euclidean space vectors to Poincare ball tangent space at origin"""
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)
        return torch.tanh(c ** 0.5 * v_norm) * (v / (c ** 0.5 * v_norm))

    @staticmethod
    def logmap0(y, c=1.0, eps=1e-5):
        """Inverse map from Poincare ball to tangent space at origin"""
        y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), max=(1 - eps) / c ** 0.5)
        return (1 / (c ** 0.5)) * torch.arctanh(c ** 0.5 * y_norm) * (y / y_norm)

    @staticmethod
    def mobius_add(x, y, c=1.0):
        """Mobius addition in Poincare ball model"""
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
        y2 = torch.sum(y ** 2, dim=-1, keepdim=True)

        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return numerator / torch.clamp(denominator, min=1e-8)

    @staticmethod
    def hyperbolic_distance(x, y, c=1.0):
        """Geodesic distance in Poincare ball model"""
        sqrt_c = c ** 0.5
        mobius_diff = HyperbolicOps.mobius_add(-x, y, c=c)
        norm_diff = torch.clamp(torch.norm(mobius_diff, dim=-1), min=1e-8)
        return (2 / sqrt_c) * torch.arctanh(sqrt_c * norm_diff)


class HyperbolicRelativePositionLoss(nn.Module):
    def __init__(self, curvature=1.0, eps=1e-5):
        super().__init__()
        self.c = curvature
        self.eps = eps

    def forward(self, outputs, old_prototypes):
        """
        Args:
            outputs: dict containing:
                'tfeatures': old model features (Euclidean) [B, D]
                'sfeatures': new model features (Euclidean) [B, D]
            old_prototypes: old class prototypes (Euclidean) [K, D]
        """
        t_features = outputs['tfeatures']
        s_features = outputs['sfeatures']
        batch_size, feat_dim = t_features.shape

        t_hyp = HyperbolicOps.expmap0(t_features, c=self.c)
        s_hyp = HyperbolicOps.expmap0(s_features, c=self.c)
        p_hyp = HyperbolicOps.expmap0(old_prototypes, c=self.c)

        t_deltas = HyperbolicOps.mobius_add(-p_hyp.unsqueeze(0), t_hyp.unsqueeze(1), c=self.c)
        s_deltas = HyperbolicOps.mobius_add(-p_hyp.unsqueeze(0), s_hyp.unsqueeze(1), c=self.c)

        t_deltas_tangent = HyperbolicOps.logmap0(t_deltas, c=self.c)
        s_deltas_tangent = HyperbolicOps.logmap0(s_deltas, c=self.c)

        t_deltas_norm = F.normalize(t_deltas_tangent, p=2, dim=-1)
        s_deltas_norm = F.normalize(s_deltas_tangent, p=2, dim=-1)

        cos_sim = torch.einsum('bkd,bkd->bk', s_deltas_norm, t_deltas_norm)
        loss = 1 - torch.mean(cos_sim)

        return loss
