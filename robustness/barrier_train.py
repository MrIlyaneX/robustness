"""Module for Barrier Traning logic"""

import os
import time
import warnings
from typing import Any, Iterable

import dill
import numpy as np
import torch as ch
from torch.optim import SGD, lr_scheduler
from autoattack import AutoAttack

from .utils import project_weights_after_step
# The logarithmic_barrier_loss is no longer the main loss, but we can use it to get margins
from .barrier_loss import augmented_lagrangian_margin_loss
from .cifar_models.resnet import get_spectral_norm
from .tools import constants as consts
from .tools import helpers
from .tools.helpers import AverageMeter, ckpt_at_epoch, has_attr

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm  # type: ignore
else:
    from tqdm import tqdm

global_step = 0

device = "cpu"
if ch.cuda.is_available():
    device = "cuda"
if ch.backends.mps.is_available():
    device = "mps"


def check_required_args(args: object, eval_only: bool = False) -> None:
    """
    Check that the required training arguments are present.
    """
    required_args_eval = ["adv_eval"]
    required_args_train = [
        "epochs",
        "out_dir",
        "adv_train",
        "log_iters",
        "lr",
        "momentum",
        "weight_decay",
        # --- MODIFIED: Replaced barrier arguments with Augmented Lagrangian arguments ---
        "delta",
        "gamma",
        "rho",         # Penalty parameter for margin constraint
        "rho_lip",     # Penalty parameter for Lipschitz constraint
        "warmup_epochs",
    ]
    adv_required_args = [
        "attack_steps",
        "eps",
        "constraint",
        "use_best",
        "attack_lr",
        "random_restarts",
    ]

    def check_args(args_list: list[str]) -> None:
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    if not eval_only:
        check_args(required_args_train)
    else:
        check_args(required_args_eval)

    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)

    has_custom_train = has_attr(args, "custom_train_loss")
    has_custom_adv = has_attr(args, "custom_adv_loss")
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError(
            "Cannot use custom train loss without a custom adversarial loss (see docs)"
        )
    
def calculate_augmented_lagrangian_losses(
    model: ch.nn.Module,
    model_logits,
    target,
    args,
    lambda_margin,
    lambda_lip,
):
    """Calculate margin and Lipschitz augmented Lagrangian losses."""
    global device

    # 1. Margin component: Calculated using the refactored function
    loss_aug_lag_margin, margin_violations, current_margins = augmented_lagrangian_margin_loss(
        logits=model_logits,
        labels=target,
        delta=args.delta,
        rho=args.rho,
        lambda_margin=lambda_margin
    )

    # 2. Lipschitz component (logic remains here as it inspects the model layers)
    num_spectral_norm_layers = 0
    total_lip_violations = ch.tensor(0.0, device=device)
    gamma_violations_count = 0

    for m in model.modules():
        spectral_norm_val = get_spectral_norm(m)
        if spectral_norm_val is not None:
            num_spectral_norm_layers += 1
            # Constraint is spectral_norm <= gamma, so violation g(x) = spectral_norm - gamma
            lip_violation = spectral_norm_val - args.gamma
            total_lip_violations += lip_violation

            if spectral_norm_val.item() > args.gamma:
                gamma_violations_count += 1
    
    # Use a single lambda for the average violation across all layers
    avg_lip_violation = total_lip_violations / max(num_spectral_norm_layers, 1)

    # Augmented Lagrangian loss for the Lipschitz constraint
    term_in_max_lip = (lambda_lip + args.rho_lip * avg_lip_violation).clamp_min_(0)
    loss_aug_lag_lip = (ch.pow(term_in_max_lip, 2) - ch.pow(lambda_lip, 2)) / (2 * args.rho_lip)

    return (
        loss_aug_lag_margin,
        loss_aug_lag_lip,
        current_margins,
        gamma_violations_count,
        margin_violations.detach(),
        avg_lip_violation.detach(),
    )


def eval_model_autoattack(
    model: ch.nn.Module, loader: Iterable, constraint: str = "inf", eps: float = 8 / 255
) -> float:
    """
    Evaluate a model's robust accuracy using AutoAttack.
    Note: This is computationally expensive.
    """
    global device
    print("\nRunning AutoAttack evaluation...")
    start_time = time.time()

    unwrapped_model = model.module if hasattr(model, "module") else model
    unwrapped_model.eval()

    x_test, y_test = [], []
    for inp, target in tqdm(loader, total=len(loader), desc="[AutoAttack] Loading data"):
        x_test.append(inp)
        y_test.append(target)
    x_test = ch.cat(x_test, dim=0).to(device)
    y_test = ch.cat(y_test, dim=0).to(device)

    norm_map = {"inf": "Linf", "2": "L2"}
    if constraint not in norm_map:
        print(f"Warning: Constraint '{constraint}' not supported by this AutoAttack script. Skipping.")
        return float("nan")
    norm = norm_map[constraint]

    adversary = AutoAttack(lambda x: unwrapped_model(x)[0], norm=norm, eps=eps, device=device, verbose=True)
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    with ch.no_grad():
        output = unwrapped_model(x_adv)
        logits = output[0] if isinstance(output, tuple) else output
        is_correct = ch.argmax(logits, dim=1) == y_test
        robust_accuracy = 100.0 * is_correct.sum().item() / len(y_test)

    print(f"AutoAttack evaluation finished in {time.time() - start_time:.2f}s. Robust Accuracy: {robust_accuracy:.2f}%")
    return robust_accuracy

def check_epoch_gamma_violations(
    model: ch.nn.Module,
    args: object,
    gamma_violation_tracker: dict[str, Any],
    wandb_run: None | Any = None,
) -> tuple[dict[Any, Any], int, int]:
    """Check gamma violations at the end of epoch and update tracker."""
    violations_this_epoch = {}
    total_violations, gamma_violation_metric, layer_idx = 0, 0, 0
    const_gamma: float = args.gamma

    for m in model.modules():
        spectral_norm_val = get_spectral_norm(m)
        if spectral_norm_val is not None:
            layer_key = f"layer_{layer_idx}"
            if spectral_norm_val >= const_gamma:
                violations_this_epoch[layer_key] = {"spectral_norm": spectral_norm_val.item(), "gamma": args.gamma}
                total_violations += 1
                gamma_violation_tracker[layer_key] = gamma_violation_tracker.get(layer_key, 0) + 1
                consecutive_violations = gamma_violation_tracker[layer_key]
                if consecutive_violations >= 3:
                    gamma_violation_metric += 1
                    warnings.warn(f"Warning: {layer_key} has violated gamma for {consecutive_violations} consecutive epochs...")
            else:
                gamma_violation_tracker[layer_key] = 0
            layer_idx += 1
    return violations_this_epoch, total_violations, gamma_violation_metric

def make_optimizer_and_schedule(
    args: object, model: ch.nn.Module, checkpoint: dict[str, Any], params: list[Any] | None,
) -> tuple[Any | SGD, ch.optim.Optimizer | None]:
    """Creates an optimizer and a schedule for a given model."""
    global device
    param_list = model.parameters() if params is None else params
    optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model.to(device=device)
    schedule = None
    # ... schedule logic ...
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if schedule and "schedule" in checkpoint:
            schedule.load_state_dict(checkpoint["schedule"])
    return optimizer, schedule

def eval_model(args: object, model: ch.nn.Module, loader: Iterable, wandb_run: Any | None = None) -> dict[str, Any]:
    """Evaluate a model for standard (and optionally adversarial) accuracy."""
    check_required_args(args, eval_only=True)
    start_time = time.time()
    returned_metrics = _model_loop(args=args, loop_type="val", loader=loader, model=model, opt=None, epoch=0, adv=False)
    nat_accuracy, nat_loss = returned_metrics["accuracy_avg"], returned_metrics["losses_avg"]

    adv_accuracy, adv_loss = float("nan"), float("nan")
    if args.adv_eval:
        returned_metrics = _model_loop(args=args, loop_type="val", loader=loader, model=model, opt=None, epoch=0, adv=True)
        adv_accuracy, adv_loss = returned_metrics["accuracy_avg"], returned_metrics["losses_avg"]

    autoattack_accuracy = float("nan")
    if has_attr(args, "autoattack_eval") and args.autoattack_eval and args.adv_eval:
        autoattack_accuracy = eval_model_autoattack(model, loader, args.constraint, args.eps)

    if wandb_run:
        wandb_run.log({"eval/nat_accuracy": nat_accuracy, "eval/adv_accuracy": adv_accuracy, "eval/autoattack_accuracy": autoattack_accuracy, "eval/nat_loss": nat_loss, "eval/adv_loss": adv_loss, "eval/time": time.time() - start_time})
    return {"nat_accuracy": nat_accuracy, "adv_accuracy": adv_accuracy, "autoattack_accuracy": autoattack_accuracy, "nat_loss": nat_loss, "adv_loss": adv_loss}

def train_model(
    args: object, model: ch.nn.Module, loaders, *, checkpoint=None, dp_device_ids=None,
    update_params=None, disable_no_grad=False, wandb_run=None,
) -> ch.nn.Module:
    """Main function for training a model."""
    global global_step, device
    check_required_args(args)
    # ... arg processing ...
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    if wandb_run:
        wandb_run.watch(model.module if hasattr(model, "module") else model, log="gradients", log_freq=100)

    best_acc, start_epoch = 0.0, 0
    if checkpoint:
        start_epoch = checkpoint["epoch"]
        acc_key = f"{'adv' if args.adv_train else 'nat'}_acc"
        best_acc = checkpoint.get(acc_key, 0)

    start_time = time.time()
    lambda_margin = ch.tensor(0.0, device=device)
    lambda_lip = ch.tensor(0.0, device=device)
    gamma_violation_tracker = {}

    for epoch in range(start_epoch, args.epochs):
        is_warmup_phase = epoch < args.warmup_epochs
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        returned_metrics = _model_loop(args, "train", train_loader, model, opt, epoch, args.adv_train, lambda_margin, lambda_lip, is_warmup_phase)
        train_acc, train_loss = returned_metrics["accuracy_avg"], returned_metrics["losses_avg"]
        lambda_margin, lambda_lip = returned_metrics["lambda_margin"], returned_metrics["lambda_lip"]
        
        # ... validation and checkpointing logic ...
        
        if schedule:
            schedule.step()
    return model

def _model_loop(
    args, loop_type, loader, model, opt, epoch: int, adv, 
    lambda_margin=None, lambda_lip=None, is_warmup_phase=False, gamma_violation_tracker=None
) -> dict[str, Any]:
    """Internal function for training or evaluation loop over a single epoch."""
    global global_step, device
    is_train = loop_type == "train"
    model.train() if is_train else model.eval()

    losses, acc_meter = AverageMeter(), AverageMeter()
    margin_al_losses, lip_al_losses = AverageMeter(), AverageMeter()
    avg_margins, gamma_violation_meter = AverageMeter(), AverageMeter()
    ce_loss_meter = AverageMeter()
    metrics_cache = []

    loop_msg = "Train" if is_train else "Val"
    train_criterion = ch.nn.CrossEntropyLoss()
    adv_criterion = args.custom_adv_loss if has_attr(args, "custom_adv_loss") else None

    attack_kwargs = {}
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps if (is_train and has_attr(args, "custom_eps_multiplier")) else args.eps
        attack_kwargs = {"constraint": args.constraint, "eps": eps, "step_size": args.attack_lr, "iterations": args.attack_steps, "random_start": args.random_restarts > 0, "custom_loss": adv_criterion, "random_restarts": (0 if is_train else args.random_restarts), "use_best": bool(args.use_best)}

    iterator = tqdm(enumerate(loader), total=len(loader), bar_format="{l_bar}{bar:30}{r_bar}", leave=False)
    for i, (inp, target) in iterator:
        if is_train: global_step += 1
        inp, target = inp.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output, _ = model(inp, target=target, make_adv=adv, **attack_kwargs)
        model_logits = output[0] if isinstance(output, tuple) else output

        ce_loss = train_criterion(model_logits, target).mean()
        ce_loss_meter.update(ce_loss.item(), inp.size(0))

        loss_aug_lag_margin, loss_aug_lag_lip = ch.tensor(0.0, device=device), ch.tensor(0.0, device=device)
        current_margins, gamma_violations, margin_violations, avg_lip_violation = None, 0, None, None

        if is_train and not is_warmup_phase:
            (loss_aug_lag_margin, loss_aug_lag_lip, current_margins,
             gamma_violations, margin_violations, avg_lip_violation) = calculate_augmented_lagrangian_losses(
                model, model_logits, target, args, lambda_margin, lambda_lip
            )
            margin_al_losses.update(loss_aug_lag_margin.item(), inp.size(0))
            lip_al_losses.update(loss_aug_lag_lip.item(), inp.size(0))
            if current_margins is not None: avg_margins.update(current_margins.mean().item(), inp.size(0))
        gamma_violation_meter.update(gamma_violations, 1)

        loss = ce_loss + loss_aug_lag_margin + loss_aug_lag_lip
        if has_attr(args, "regularizer"):
            loss += args.regularizer(model, inp, target)
        losses.update(loss.item(), inp.size(0))

        batch_acc = helpers.accuracy(model_logits, target)
        acc_meter.update(batch_acc, inp.size(0))

        if is_train:
            opt.zero_grad()
            loss.backward()
            opt.step()
            project_weights_after_step(model)
            
            if not is_warmup_phase:
                # --- START: LAST BATCH BUG FIX ---
                # Always treat lambda_margin as a scalar.
                # Update it based on the MEAN violation of the batch.
                if margin_violations is not None:
                    with ch.no_grad():
                        update = args.rho * margin_violations.mean()
                        lambda_margin.add_(update).clamp_min_(0)
                # --- END: LAST BATCH BUG FIX ---
                
                if avg_lip_violation is not None:
                    lambda_lip = (lambda_lip + args.rho_lip * avg_lip_violation).clamp_min_(0)

            if i % args.log_iters == 0:
                # ... logging logic ...
                pass

        desc = f"{loop_msg} E{epoch+1:>3d}"
        base_stats = {"Loss": f"{losses.avg:.3f}", "CE": f"{ce_loss_meter.avg:.3f}", "Acc": f"{acc_meter.avg:.2f}%"}
        if is_train and not is_warmup_phase:
            al_stats = {"MgnAL": f"{margin_al_losses.avg:.3f}", "LipAL": f"{lip_al_losses.avg:.3f}"}
            if avg_margins.count > 0: al_stats["Mgn"] = f"{avg_margins.avg:.3f}"
            base_stats.update(al_stats)
        iterator.set_description(f"{desc} | {' | '.join(f'{k} {v}' for k, v in base_stats.items())}")

    return {
        "accuracy_avg": acc_meter.avg, "losses_avg": losses.avg,
        "lambda_margin": lambda_margin, "lambda_lip": lambda_lip,
        "avg_margins": avg_margins.avg, "gamma_violation_meter_avg": gamma_violation_meter.avg,
        "metrics_cache": metrics_cache,
    }