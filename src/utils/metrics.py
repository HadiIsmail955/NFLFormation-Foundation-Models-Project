import torch


@torch.no_grad()
def dice_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    reduce: bool = True,
):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (pred_sum + target_sum - intersection + eps)

    if reduce:
        return dice.mean().item(), iou.mean().item()
    else:
        return dice, iou


@torch.no_grad()
def precision_recall_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    reduce: bool = True,
):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    if reduce:
        return precision.mean().item(), recall.mean().item()
    else:
        return precision, recall
    

@torch.no_grad()
def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

@torch.no_grad()
def macro_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    B, P, H, W = preds.shape

    dice_list, iou_list, prec_list, rec_list = [], [], [], []

    for p in range(P):
        gt = targets[:, p]
        pr = preds[:, p]

        if gt.sum() == 0:
            continue

        inter = (pr * gt).sum(dim=(1, 2))
        pr_sum = pr.sum(dim=(1, 2))
        gt_sum = gt.sum(dim=(1, 2))
        union = pr_sum + gt_sum - inter

        dice = (2 * inter + eps) / (pr_sum + gt_sum + eps)
        iou  = (inter + eps) / (union + eps)

        precision = (inter + eps) / (pr_sum + eps)
        recall    = (inter + eps) / (gt_sum + eps)

        dice_list.append(dice.mean())
        iou_list.append(iou.mean())
        prec_list.append(precision.mean())
        rec_list.append(recall.mean())

    if len(dice_list) == 0:
        zero = torch.tensor(0.0, device=logits.device)
        return zero, zero, zero, zero

    return (
        torch.stack(dice_list).mean(),
        torch.stack(iou_list).mean(),
        torch.stack(prec_list).mean(),
        torch.stack(rec_list).mean(),
    )
