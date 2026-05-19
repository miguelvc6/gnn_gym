import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def binary_roc_auc(logits: torch.Tensor, y: torch.Tensor) -> float:
    y_true = y.detach().cpu().view(-1).numpy()
    y_score = logits.detach().cpu().view(-1).numpy()
    return float(roc_auc_score(y_true, y_score))


def average_precision(logits: torch.Tensor, y: torch.Tensor) -> float:
    y_cpu = y.detach().cpu()
    logits_cpu = logits.detach().cpu()
    mask = ~torch.isnan(y_cpu)
    if y_cpu.ndim == 1 or y_cpu.shape[-1] == 1:
        return float(average_precision_score(y_cpu[mask].numpy(), logits_cpu[mask].numpy()))

    scores: list[float] = []
    for idx in range(y_cpu.shape[-1]):
        column_mask = mask[:, idx]
        if column_mask.sum() == 0:
            continue
        labels = y_cpu[column_mask, idx].numpy()
        if len(set(labels.tolist())) < 2:
            continue
        scores.append(float(average_precision_score(labels, logits_cpu[column_mask, idx].numpy())))
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def mean_absolute_error(pred: torch.Tensor, y: torch.Tensor) -> float:
    mask = ~torch.isnan(y)
    return float(torch.abs(pred[mask] - y[mask]).mean().item())


def hits_at_k(pos_pred: torch.Tensor, neg_pred: torch.Tensor, k: int) -> float:
    kth_score = torch.topk(neg_pred.view(-1), min(k, neg_pred.numel())).values[-1]
    return float((pos_pred.view(-1) > kth_score).float().mean().item())
