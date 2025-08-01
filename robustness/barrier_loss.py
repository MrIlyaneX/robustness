import torch
import torch.nn.functional as F


def per_sample_margin_loss(logits: torch.Tensor, labels: torch.Tensor, delta: float) -> torch.Tensor:
    """Calculates the per-sample margin loss.
    Args:
        logits (torch.Tensor): Model outputs.
        labels (torch.Tensor): True labels.
        delta (float): Margin threshold.
    Returns:
        g_theta (torch.Tensor): The margin values for each sample.
    """
    correct_label_logit = logits[torch.arange(len(labels)), labels]
    highest_incorrect_logit = (
        logits.masked_fill(F.one_hot(labels, num_classes=logits.shape[1]).bool(), -1e9).max(1).values
    )
    margins = correct_label_logit - highest_incorrect_logit
    g_theta = delta - margins
    return g_theta


def logarithmic_barrier_loss(
    logits: torch.Tensor, labels: torch.Tensor, delta: float, mu: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the logarithmic barrier loss to enforce a margin.
    Args:
        logits (torch.Tensor): Model outputs.
        labels (torch.Tensor): True labels.
        delta (float): Margin threshold.
        mu (float): Barrier parameter.
    Returns:
        barrier_loss (torch.Tensor): The computed barrier loss.
        g_theta (torch.Tensor): The margin values.
    """
    g_theta = per_sample_margin_loss(logits, labels, delta)
    barrier_loss = -mu * torch.log(torch.clamp(-g_theta, min=1e-6)).mean()

    return barrier_loss, g_theta.detach()
