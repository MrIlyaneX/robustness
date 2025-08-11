"""Module for Barrier Traning logic"""

import os
import time
import warnings
from typing import Any, Iterable

import dill
import numpy as np
import torch as ch
from torch.optim import SGD, lr_scheduler

from .utils import project_weights_after_step
from .barrier_loss import logarithmic_barrier_loss
from .cifar_models.resnet import get_spectral_norm
from .tools import constants as consts
from .tools import helpers
from .tools.helpers import AverageMeter, ckpt_at_epoch, has_attr

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm  # type: ignore
else:
    from tqdm import tqdm

try:
    from apex import amp
except Exception:
    # warnings.warn("Could not import amp.")
    pass


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
        # Required arguments for barrier methods
        "delta",
        "gamma",
        "mu",
        "mu_lip",
        "eta",
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


def calculate_barrier_losses(
    model: ch.nn.Module,
    model_logits,
    target,
    args,
    current_mu: float,
    current_mu_lip: float,
):
    """Calculate margin and Lipschitz barrier losses."""
    global device

    loss_bar, current_margins = logarithmic_barrier_loss(model_logits, target, args.delta, current_mu)

    num_spectral_norm_layers = 0
    current_lip_bar_sum = ch.tensor(0.0, device=device)
    gamma_violations = 0

    for m in model.modules():
        spectral_norm_val = get_spectral_norm(m)
        if spectral_norm_val is not None:
            num_spectral_norm_layers += 1
            log_arg_lip = ch.clamp_min(args.gamma - spectral_norm_val, 1e-8)
            current_lip_bar_sum += -current_mu_lip * ch.log(log_arg_lip)

            if spectral_norm_val >= args.gamma:
                gamma_violations += 1

    lip_bar = current_lip_bar_sum / max(num_spectral_norm_layers, 1)

    return loss_bar, lip_bar, current_margins, gamma_violations


def check_epoch_gamma_violations(
    model: ch.nn.Module,
    args: object,
    gamma_violation_tracker: dict[str, Any],
    wandb_run: None | Any = None,
) -> tuple[dict[Any, Any], int, int]:
    """Check gamma violations at the end of epoch and update tracker."""
    violations_this_epoch = {}
    total_violations = 0
    gamma_violation_metric = 0
    layer_idx = 0
    const_gamma: float = args.gamma

    for m in model.modules():
        spectral_norm_val = get_spectral_norm(m)
        if spectral_norm_val is not None:
            layer_key = f"layer_{layer_idx}"
            if spectral_norm_val >= const_gamma:
                violations_this_epoch[layer_key] = {
                    "spectral_norm": spectral_norm_val.item(),
                    "gamma": args.gamma,
                }
                total_violations += 1
                gamma_violation_tracker[layer_key] = gamma_violation_tracker.get(layer_key, 0) + 1
                consecutive_violations = gamma_violation_tracker[layer_key]
                if consecutive_violations >= 3:
                    gamma_violation_metric += 1
                    warning_msg = f"Warning: {layer_key} has violated gamma for {consecutive_violations} consecutive epochs (spectral_norm: {spectral_norm_val:.4f}, gamma: {args.gamma:.4f})"
                    warnings.warn(warning_msg)
                    if wandb_run is not None:
                        wandb_run.log({
                            f"warnings/{layer_key}_gamma_violation": consecutive_violations,
                            f"warnings/{layer_key}_spectral_norm": spectral_norm_val.item(),
                            f"warnings/{layer_key}_gamma": args.gamma,
                            "warnings/gamma_violation_message": warning_msg,
                        }, commit=False)
            else:
                print(f"{layer_key} spectral norm: {spectral_norm_val:.4f}")
                gamma_violation_tracker[layer_key] = 0
            layer_idx += 1
    return violations_this_epoch, total_violations, gamma_violation_metric


def make_optimizer_and_schedule(
    args: object,
    model: ch.nn.Module,
    checkpoint: dict[str, Any],
    params: list[Any] | None,
) -> tuple[Any | SGD, ch.optim.Optimizer | None]:
    """Creates an optimizer and a schedule for a given model."""
    global device
    param_list = model.parameters() if params is None else params
    optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.mixed_precision:
        model.to("cuda")
        model, optimizer = amp.initialize(model, optimizer, "O1")
    else:
        model.to(device=device)

    schedule = None
    if args.custom_lr_multiplier == "cyclic":
        def lr_func(t: int) -> Any:
            eps = args.epochs
            return np.interp([t], [0, eps * 4 // 15, eps], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_multiplier:
        cs = args.custom_lr_multiplier
        periods = eval(cs) if isinstance(cs, str) else cs
        if args.lr_interpolation == "linear":
            def lr_func(t: int) -> Any:
                return np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep) -> Any:
                for milestone, lr in reversed(periods):
                    if ep >= milestone:
                        return lr
                return 1.0
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            if schedule:
                schedule.load_state_dict(checkpoint["schedule"])
        except:
            steps_to_take = checkpoint["epoch"]
            print(f"Could not load schedule. Stepping {steps_to_take} times instead...")
            for _ in range(steps_to_take):
                if schedule:
                    schedule.step()
        if "amp" in checkpoint and checkpoint["amp"] not in [None, "N/A"]:
            amp.load_state_dict(checkpoint["amp"])
        if args.mixed_precision:
            model.load_state_dict(checkpoint["model"])
    return optimizer, schedule


def eval_model(args: object, model: ch.nn.Module, loader: Iterable, wandb_run: Any | None = None) -> dict[str, Any]:
    """Evaluate a model for standard (and optionally adversarial) accuracy."""
    check_required_args(args, eval_only=True)
    start_time = time.time()
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model)

    # Natural evaluation
    returned_metrics = _model_loop(args, "val", loader, model, None, 0, False)
    nat_acc = returned_metrics["accuracy_avg"]
    nat_loss = returned_metrics["losses_avg"]

    adv_acc, adv_loss = float("nan"), float("nan")
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if has_attr(args, "eps") else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, "attack_lr") else None
        # Adversarial evaluation
        returned_metrics = _model_loop(args, "val", loader, model, None, 0, True)
        adv_acc = returned_metrics["accuracy_avg"]
        adv_loss = returned_metrics["losses_avg"]

    if wandb_run:
        wandb_run.log({
            "eval/nat_acc": nat_acc,
            "eval/adv_acc": adv_acc,
            "eval/nat_loss": nat_loss,
            "eval/adv_loss": adv_loss,
            "eval/time": time.time() - start_time,
        })
    return {
        "nat_acc": nat_acc,
        "adv_acc": adv_acc,
        "nat_loss": nat_loss,
        "adv_loss": adv_loss,
    }


def train_model(
    args: object,
    model: ch.nn.Module,
    loaders,
    *,
    checkpoint=None,
    dp_device_ids=None,
    update_params=None,
    disable_no_grad=False,
    wandb_run=None,
) -> ch.nn.Module:
    """Main function for training a model."""
    global global_step, device
    check_required_args(args)
    for p in ["eps", "attack_lr", "custom_eps_multiplier"]:
        setattr(args, p, eval(str(getattr(args, p))) if has_attr(args, p) else None)
    if args.custom_eps_multiplier is not None:
        eps_periods = args.custom_eps_multiplier
        args.custom_eps_multiplier = lambda t: np.interp([t], *zip(*eps_periods))[0]

    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    assert not hasattr(model, "module"), "model is already in DataParallel."
    if ch.cuda.is_available():
        model = ch.nn.DataParallel(model, device_ids=dp_device_ids).cuda()
    else:
        model.to(device=device)

    if wandb_run:
        watched_model = model.module if hasattr(model, "module") else model
        wandb_run.watch(watched_model, opt, log="gradients", log_freq=1, log_graph=True)

    best_acc = 0.0
    start_epoch = 0
    if checkpoint:
        start_epoch = checkpoint["epoch"]
        acc_key = f"{'adv' if args.adv_train else 'nat'}_acc"
        if acc_key in checkpoint:
            best_acc = checkpoint[acc_key]
        else:
            best_acc = _model_loop(args, "val", val_loader, model, None, start_epoch - 1, args.adv_train)["accuracy_avg"]

    start_time = time.time()
    current_mu = args.mu
    current_mu_lip = args.mu_lip
    lambda_dual = ch.tensor([0.0]).to(device=device)
    gamma_violation_tracker = {}

    for epoch in range(start_epoch, args.epochs):
        is_warmup_phase = epoch < args.warmup_epochs
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # Train for one epoch
        returned_metrics = _model_loop(args, "train", train_loader, model, opt, epoch, args.adv_train, current_mu, current_mu_lip, lambda_dual, is_warmup_phase, gamma_violation_tracker)
        train_acc = returned_metrics["accuracy_avg"]
        train_loss = returned_metrics["losses_avg"]
        train_avg_margins = returned_metrics["avg_margins"]
        train_gamma_violation_avg = returned_metrics["gamma_violation_meter_avg"]
        metrics_cache = returned_metrics["metrics_cache"]
        lambda_dual = returned_metrics["lambda_dual"]

        if wandb_run and metrics_cache:
            for log_item in metrics_cache:
                wandb_run.log(log_item, commit=True)

        # Validation
        ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()
        with ctx:
            returned_metrics = _model_loop(args, "val", val_loader, model, None, epoch, False)
            nat_acc = returned_metrics["accuracy_avg"]
            nat_loss = returned_metrics["losses_avg"]

        adv_val_acc, adv_val_loss, adv_avg_margins, gamma_violation_avg = (float("nan"),) * 4
        if args.adv_eval or args.adv_train:
            returned_metrics = _model_loop(args, "val", val_loader, model, None, epoch, True, current_mu, current_mu_lip, lambda_dual, is_warmup_phase, gamma_violation_tracker)
            adv_val_acc = returned_metrics["accuracy_avg"]
            adv_val_loss = returned_metrics["losses_avg"]
            adv_avg_margins = returned_metrics["avg_margins"]
            gamma_violation_avg = returned_metrics["gamma_violation_meter_avg"]

        _, total_violations_this_epoch, gamma_violation_metric = check_epoch_gamma_violations(model, args, gamma_violation_tracker, wandb_run)

        # Save checkpoint and update best accuracy
        our_acc = adv_val_acc if args.adv_train else nat_acc
        is_best = our_acc > best_acc
        best_acc = max(our_acc, best_acc)
        acc_key = f"{'adv' if args.adv_train else 'nat'}_acc"

        sd_info = {"model": model.state_dict(), "optimizer": opt.state_dict(), "schedule": (schedule and schedule.state_dict()), "epoch": epoch + 1, "amp": (amp.state_dict() if args.mixed_precision else None), "mu": current_mu, "mu_lip": current_mu_lip, "lambda_dual": lambda_dual.cpu().numpy(), "iter_step": global_step, "gamma_violation_tracker": gamma_violation_tracker, acc_key: our_acc}

        if wandb_run:
            global_step += 1
            wandb_log_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "val/nat_loss": nat_loss,
                "val/nat_acc": nat_acc,
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "train/epoch_lambda_dual": lambda_dual.mean().item(),
                "train/epoch_avg_margins": train_avg_margins,
                "train/epoch_gamma_violation_avg": train_gamma_violation_avg,
                "val/adv_loss": adv_val_loss,
                "val/adv_acc": adv_val_acc,
                "val/avg_margins": adv_avg_margins,
                "total_time": time.time() - start_time,
                "gamma_violations/total_this_epoch": total_violations_this_epoch,
                "gamma_violations/metric": gamma_violation_metric,
            }
            if not is_warmup_phase:
                wandb_log_dict["val/gamma_violations_avg"] = gamma_violation_avg
            wandb_run.log(wandb_log_dict, commit=True)

        last_epoch = epoch == (args.epochs - 1)
        save_its = args.save_ckpt_iters
        if (save_its > 0 and epoch % save_its == 0) or last_epoch:
            ch.save(sd_info, os.path.join(args.out_dir, ckpt_at_epoch(epoch)), pickle_module=dill)
        ch.save(sd_info, os.path.join(args.out_dir, consts.CKPT_NAME_LATEST), pickle_module=dill)
        if is_best:
            ch.save(sd_info, os.path.join(args.out_dir, consts.CKPT_NAME_BEST), pickle_module=dill)

        if schedule:
            schedule.step()
        current_mu *= 0.9
        current_mu_lip *= 0.9
    return model


def _model_loop(
    args, loop_type, loader, model, opt, epoch: int, adv, current_mu=0.0, current_mu_lip=0.0,
    lambda_dual=None, is_warmup_phase=False, gamma_violation_tracker=None
) -> dict[str, Any]:
    """Internal function for training or evaluation loop over a single epoch."""
    global global_step, device
    is_train = loop_type == "train"
    model = model.train() if is_train else model.eval()

    losses, acc_meter = AverageMeter(), AverageMeter()
    margin_barrier_losses, lip_barrier_losses = AverageMeter(), AverageMeter()
    avg_margins, gamma_violation_meter = AverageMeter(), AverageMeter()
    ce_loss_meter = AverageMeter()
    metrics_cache = []

    loop_msg = "Train" if is_train else "Val"
    train_criterion = args.custom_train_loss if has_attr(args, "custom_train_loss") else ch.nn.CrossEntropyLoss()
    adv_criterion = args.custom_adv_loss if has_attr(args, "custom_adv_loss") else None

    attack_kwargs = {}
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps if (is_train and args.custom_eps_multiplier) else args.eps
        attack_kwargs = {
            "constraint": args.constraint, "eps": eps, "step_size": args.attack_lr,
            "iterations": args.attack_steps, "random_start": args.random_start,
            "custom_loss": adv_criterion, "random_restarts": (0 if is_train else args.random_restarts),
            "use_best": bool(args.use_best),
        }

    iterator = tqdm(enumerate(loader), total=len(loader), bar_format="{l_bar}{bar:30}{r_bar}", leave=False)
    for i, (inp, target) in iterator:
        if is_train: global_step += 1
        inp, target = inp.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output, _ = model(inp, target=target, make_adv=adv, **attack_kwargs)
        model_logits = output[0] if isinstance(output, tuple) else output

        ce_loss = train_criterion(model_logits, target).mean()
        ce_loss_meter.update(ce_loss.item(), inp.size(0))

        loss_bar, lip_bar, current_margins, gamma_violations = ch.tensor(0.0, device=device), ch.tensor(0.0, device=device), None, 0

        if is_train and not is_warmup_phase:
            loss_bar, lip_bar, current_margins, gamma_violations = calculate_barrier_losses(model, model_logits, target, args, current_mu, current_mu_lip)
            margin_barrier_losses.update(loss_bar.item(), inp.size(0))
            lip_barrier_losses.update(lip_bar.item(), inp.size(0))
            if current_margins is not None: avg_margins.update(current_margins.mean().item(), inp.size(0))
        gamma_violation_meter.update(gamma_violations, 1)

        loss = ce_loss + loss_bar + lip_bar
        if has_attr(args, "regularizer"):
            loss += args.regularizer(model, inp, target)
        losses.update(loss.item(), inp.size(0))

        batch_acc = 0.0
        try:
            if has_attr(args, "custom_accuracy"):
                batch_acc = args.custom_accuracy(model_logits, target)
            else:
                batch_acc = helpers.accuracy(model_logits, target)
            acc_meter.update(batch_acc, inp.size(0))
        except Exception as e:
            warnings.warn(f"Failed to calculate accuracy: {e}")

        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()
            opt.step()
            
            project_weights_after_step(model)
            
            if not is_warmup_phase and current_margins is not None:
                if lambda_dual is not None and lambda_dual.shape[0] != inp.shape[0]:
                    lambda_dual = ch.zeros(inp.shape[0], device=device)
                lambda_dual = (lambda_dual + args.eta * current_margins.detach()).clamp_min_(0)

            iteration_log_dict = {
                "iter_step": global_step, "global_step": global_step,
                "train/iter_loss": loss.item(), "train/iter_ce_loss": ce_loss.item(),
                "train/iter_acc": batch_acc,
            }
            if not is_warmup_phase:
                iteration_log_dict.update({
                    "train/iter_margin_barrier_loss": loss_bar.item(),
                    "train/iter_lip_barrier_loss": lip_bar.item(),
                    "train/iter_gamma_violations": gamma_violations,
                })
                if current_margins is not None:
                    iteration_log_dict["train/iter_average_margin"] = current_margins.mean().item()
            metrics_cache.append(iteration_log_dict)

        desc = f"{loop_msg} E{epoch:>3d}"
        base_stats = {"Loss": f"{losses.avg:.3f}", "CE": f"{ce_loss_meter.avg:.3f}", "Acc": f"{acc_meter.avg:.2f}%"}
        if is_train and not is_warmup_phase:
            barrier_stats = {"MgnB": f"{margin_barrier_losses.avg:.3f}", "LipB": f"{lip_barrier_losses.avg:.3f}"}
            if avg_margins.count > 0: barrier_stats["Mgn"] = f"{avg_margins.avg:.3f}"
            base_stats.update(barrier_stats)
        iterator.set_description(f"{desc} | {' | '.join(f'{k} {v}' for k, v in base_stats.items())}")

    return {
        "accuracy_avg": acc_meter.avg, "losses_avg": losses.avg, "lambda_dual": lambda_dual,
        "avg_margins": avg_margins.avg, "gamma_violation_meter_avg": gamma_violation_meter.avg,
        "metrics_cache": metrics_cache,
    }