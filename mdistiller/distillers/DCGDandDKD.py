import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def loss_fn_kd_crd(outputs, labels, teacher_outputs, alpha, temperature):
    p_s = F.log_softmax(outputs/temperature, dim=1)
    p_t = F.softmax(teacher_outputs/temperature, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (temperature**2) / outputs.shape[0] + F.cross_entropy(outputs, labels) * 0.3

    return loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DCGD_DKD(Distiller):
    """DCGD+DKD"""

    def __init__(self, student, teacher, cfg):
        super(DCGD_DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        out_multi, out_single = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(out_single, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            out_single,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        loss_multi_kd = loss_fn_kd_crd(out_multi, target, logits_teacher, 0.3, 20)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "loss_multi_kd": loss_multi_kd
        }
        return out_single, losses_dict
        # return out_multi, losses_dict
