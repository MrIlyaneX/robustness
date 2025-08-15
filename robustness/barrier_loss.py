import torch
import torch.nn.functional as F


def per_sample_margin_loss(logits: torch.Tensor, labels: torch.Tensor, delta: float) -> torch.Tensor:
    """Calculates the per-sample margin loss violation, g(x) = delta - margin.

    Args:
        logits (torch.Tensor): Model outputs.
        labels (torch.Tensor): True labels.
        delta (float): Margin threshold.

    Returns:
        g_theta (torch.Tensor): The margin violation values for each sample. Positive means violated.
    """
    correct_label_logit = logits[torch.arange(len(labels)), labels]
    
    mask = F.one_hot(labels, num_classes=logits.shape[1]).bool()
    highest_incorrect_logit = logits.masked_fill(mask, -torch.inf).max(dim=1).values
    
    margins = correct_label_logit - highest_incorrect_logit
    g_theta = delta - margins
    return g_theta, margins


# --- MODIFIED FUNCTION ---
def augmented_lagrangian_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    delta: float,
    rho: float,
    lambda_margin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Now returns 3 items
    """
    Calculates the Augmented Lagrangian loss for the margin constraint.
    The constraint is g(x) <= 0, where g(x) = delta - margin.
    The loss is (1/2*rho) * ([max(0, lambda + rho*g(x))]^2 - lambda^2).

    Args:
        logits (torch.Tensor): Model outputs.
        labels (torch.Tensor): True labels.
        delta (float): Margin threshold.
        rho (float): Penalty parameter for the margin constraint.
        lambda_margin (torch.Tensor): Lagrange multiplier for the margin constraint.

    Returns:
        al_loss (torch.Tensor): The computed Augmented Lagrangian loss.
        g_theta (torch.Tensor): The detached margin violation values for the dual update.
        margins (torch.Tensor): The detached actual margin values for logging.
    """
    g_theta, margins = per_sample_margin_loss(logits, labels, delta)

    if lambda_margin.dim() == 0 or lambda_margin.shape[0] != g_theta.shape[0]:
        lambda_margin = torch.zeros_like(g_theta) + lambda_margin.mean()

    term_in_max = (lambda_margin + rho * g_theta).clamp_min_(0)
    
    al_penalty = (torch.pow(term_in_max, 2) - torch.pow(lambda_margin, 2)) / (2 * rho)
    al_loss = al_penalty.mean()

    return al_loss, g_theta.detach(), margins.detach()


def logarithmic_barrier_loss(
    logits: torch.Tensor, labels: torch.Tensor, delta: float, mu: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the logarithmic barrier loss to enforce a margin.
    (This function is now legacy and not used by the Augmented Lagrangian method).
    """
    g_theta, margins = per_sample_margin_loss(logits, labels, delta)
    barrier_loss = -mu * torch.log(torch.clamp(-g_theta, min=1e-8)).mean()

    return barrier_loss, margins.detach()