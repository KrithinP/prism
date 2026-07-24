"""YOLOX training loss with SimOTA label assignment.

PORTED and adapted from the official Megvii YOLOX (yolox/models/yolo_head.py),
simplified for our fixed decoupled head. This is the *non-ablated* machinery: we
hold it identical across every experiment and do not vary it. Read it (textbook
Ch.6); do not tweak it per-experiment.

Head output per scale: (B, 4+1+C, H, W) = [reg(4), obj(1), cls(C)].
Targets: a length-B list of (Ni, 5) tensors, each row [cls, x1, y1, x2, y2] in pixels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bboxes_iou(a, b):
    """IoU between two sets of xyxy boxes. a:(N,4) b:(M,4) -> (N,M)."""
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1)
    inter = (br - tl).clamp(min=0).prod(2)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def cxcywh_to_xyxy(boxes):
    xy, wh = boxes[..., :2], boxes[..., 2:]
    return torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)


class YOLOXLoss(nn.Module):
    def __init__(self, num_classes=1, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    # ---- flatten head outputs and build per-anchor grid + stride ----
    def _gather(self, outputs):
        preds, grids, strides = [], [], []
        for out, stride in zip(outputs, self.strides):
            B, ch, H, W = out.shape
            out = out.flatten(2).permute(0, 2, 1)          # (B, H*W, ch)
            yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            grid = torch.stack((xv, yv), 2).reshape(1, H * W, 2).to(out.device).float()
            preds.append(out)
            grids.append(grid)
            strides.append(torch.full((1, H * W, 1), stride, device=out.device).float())
        preds = torch.cat(preds, 1)                        # (B, N, ch)
        grids = torch.cat(grids, 1)                        # (1, N, 2)
        strides = torch.cat(strides, 1)                    # (1, N, 1)
        # decode boxes to cxcywh in pixels (raw obj/cls kept as logits)
        xy = (preds[..., 0:2] + grids) * strides
        wh = torch.exp(preds[..., 2:4]) * strides
        boxes = torch.cat([xy, wh], -1)                    # (B, N, 4) cxcywh
        return preds, boxes, grids, strides

    def forward(self, outputs, targets):
        preds, boxes, grids, strides = self._gather(outputs)
        B, N, _ = preds.shape
        obj_logits = preds[..., 4]
        cls_logits = preds[..., 5:]

        total_iou = total_obj = total_cls = 0.0
        num_fg_total = 0

        for b in range(B):
            gt = targets[b].to(preds.device)
            obj_target = torch.zeros(N, device=preds.device)
            if gt.numel() == 0:
                total_obj = total_obj + self.bce(obj_logits[b], obj_target).sum()
                continue

            gt_cls = gt[:, 0].long()
            gt_xyxy = gt[:, 1:5]
            gt_cxcywh = torch.cat([(gt_xyxy[:, :2] + gt_xyxy[:, 2:]) / 2,
                                   gt_xyxy[:, 2:] - gt_xyxy[:, :2]], 1)

            fg_mask, matched_gt, matched_iou = self._assign(
                boxes[b], cls_logits[b], obj_logits[b], grids[0], strides[0, :, 0],
                gt_cxcywh, gt_cls,
            )
            num_fg = int(fg_mask.sum())
            num_fg_total += num_fg
            obj_target[fg_mask] = 1.0
            total_obj = total_obj + self.bce(obj_logits[b], obj_target).sum()

            if num_fg == 0:
                continue

            # box (IoU) loss on positives
            pred_fg = cxcywh_to_xyxy(boxes[b][fg_mask])
            gt_fg = gt_xyxy[matched_gt]
            iou = bboxes_iou(pred_fg, gt_fg).diagonal().clamp(min=1e-6)
            total_iou = total_iou + (1 - iou).sum()

            # classification loss on positives (soft target = onehot * iou)
            cls_t = F.one_hot(gt_cls[matched_gt], self.num_classes).float() * matched_iou[:, None]
            total_cls = total_cls + self.bce(cls_logits[b][fg_mask], cls_t).sum()

        denom = max(num_fg_total, 1)
        loss = (5.0 * total_iou + total_obj + total_cls) / denom
        as_f = lambda v: (float(v.detach()) if torch.is_tensor(v) else float(v)) / denom
        return loss, {
            "iou": as_f(total_iou) if num_fg_total else 0.0,
            "obj": as_f(total_obj),
            "cls": as_f(total_cls) if num_fg_total else 0.0,
            "num_fg": num_fg_total,
        }

    # ---- SimOTA for one image ----
    @torch.no_grad()
    def _assign(self, boxes, cls_logits, obj_logits, grids, strides, gt_cxcywh, gt_cls):
        N = boxes.shape[0]
        n_gt = gt_cxcywh.shape[0]
        centers = (grids + 0.5) * strides[:, None]         # anchor centres in pixels

        # 1) center prior: anchor centre inside gt box, or within radius*stride
        gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
        in_box = (
            (centers[:, None, 0] >= gt_xyxy[None, :, 0]) & (centers[:, None, 0] <= gt_xyxy[None, :, 2]) &
            (centers[:, None, 1] >= gt_xyxy[None, :, 1]) & (centers[:, None, 1] <= gt_xyxy[None, :, 3])
        )
        radius = 2.5
        gc = gt_cxcywh[:, :2]
        in_center = (
            (centers[:, None, 0] >= gc[None, :, 0] - radius * strides[:, None]) &
            (centers[:, None, 0] <= gc[None, :, 0] + radius * strides[:, None]) &
            (centers[:, None, 1] >= gc[None, :, 1] - radius * strides[:, None]) &
            (centers[:, None, 1] <= gc[None, :, 1] + radius * strides[:, None])
        )
        candidate = (in_box | in_center).any(1)            # (N,) anchors worth considering
        fg_mask = torch.zeros(N, dtype=torch.bool, device=boxes.device)
        matched_gt = torch.zeros(0, dtype=torch.long, device=boxes.device)
        matched_iou = torch.zeros(0, device=boxes.device)
        if candidate.sum() == 0:
            return fg_mask, matched_gt, matched_iou

        cand_boxes = cxcywh_to_xyxy(boxes[candidate])
        iou = bboxes_iou(cand_boxes, gt_xyxy)              # (Nc, n_gt)
        iou_cost = -torch.log(iou + 1e-8)

        # classification cost: predicted score vs gt onehot.
        # BCE on sigmoided probs is unsafe under AMP autocast, and assignment should be
        # full-precision bookkeeping anyway, so compute it in fp32 with autocast off.
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            score = (cls_logits[candidate].float().sigmoid()
                     * obj_logits[candidate].float().sigmoid()[:, None]).sqrt()
            # F.binary_cross_entropy carries a DEVICE-SIDE assert that its input is in
            # [0,1]. If training diverges the logits go NaN, clamp propagates NaN, the
            # assert fires -- and a device-side assert corrupts the CUDA context for the
            # entire process, killing every subsequent run. nan_to_num keeps divergence a
            # recoverable Python-level event (train() then detects the non-finite loss).
            score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
            gt_onehot = F.one_hot(gt_cls, self.num_classes).float()
            cls_cost = F.binary_cross_entropy(
                score[:, None, :].expand(-1, n_gt, -1).clamp(1e-6, 1 - 1e-6),
                gt_onehot[None].expand(score.shape[0], -1, -1), reduction="none",
            ).sum(-1)

        is_in = (in_box | in_center)[candidate]            # (Nc, n_gt)
        cost = cls_cost + 3.0 * iou_cost + 1e5 * (~is_in)

        # 2) dynamic-k matching
        matching = torch.zeros_like(cost)
        n_candidate_k = min(10, iou.shape[0])
        topk_iou, _ = torch.topk(iou, n_candidate_k, dim=0)
        dynamic_k = torch.clamp(topk_iou.sum(0).int(), min=1)
        for g in range(n_gt):
            _, pos = torch.topk(cost[:, g], k=int(dynamic_k[g]), largest=False)
            matching[pos, g] = 1.0
        # resolve anchors matched to >1 gt: keep lowest cost
        multi = matching.sum(1) > 1
        if multi.any():
            best = cost[multi].argmin(1)
            matching[multi] = 0.0
            matching[multi, best] = 1.0

        fg_cand = matching.sum(1) > 0
        cand_idx = candidate.nonzero(as_tuple=True)[0]
        fg_mask[cand_idx[fg_cand]] = True
        matched_gt = matching[fg_cand].argmax(1)
        matched_iou = iou[fg_cand, matched_gt]
        return fg_mask, matched_gt, matched_iou
