import os
import time
import warnings
from typing import Any, Iterable

import dill
import numpy as np
import torch as ch
from torch.optim import SGD, lr_scheduler

from .barrier_loss import logarithmic_barrier_loss, per_sample_margin_loss
from .cifar_models.resnet import get_spectral_norm
from .tools import constants as consts
from .tools import helpers
from .tools.helpers import AverageMeter, ckpt_at_epoch, has_attr

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

try:
    from apex import amp
except Exception as e:
    # warnings.warn("Could not import amp.")
    pass

import math

math.log(float(1e-6))

global_step = 0

device = "cpu"
if ch.cuda.is_available():
    device = "cuda"
if ch.backends.mps.is_available():
    device = "mps"


def check_required_args(args: object, eval_only: bool = False) -> None:
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
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

    # Generic function for checking all arguments in a list
    def check_args(args_list: list[str]) -> None:
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only:
        check_args(required_args_train)
    else:
        check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, "custom_train_loss")
    has_custom_adv = has_attr(args, "custom_adv_loss")
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError(
            "Cannot use custom train loss \
            without a custom adversarial loss (see docs)"
        )


def calculate_barrier_losses(
    model: ch.nn.Module,
    model_logits,
    target,
    args,
    current_mu: float,
    current_mu_lip: float,
    gamma_violation_tracker: dict[str, int],
):
    """Calculate margin and Lipschitz barrier losses."""
    global device

    loss_bar, current_margins = logarithmic_barrier_loss(model_logits, target, args.delta, current_mu)

    # Lipschitz Barrier Loss
    num_spectral_norm_layers = 0
    gamma_violations = 0
    current_lip_bar_sum = ch.tensor(0.0, device=device)

    layer_idx = 0
    for idx, m in enumerate(model.modules()):
        spectral_norm_val = get_spectral_norm(m)
        if spectral_norm_val is not None:
            num_spectral_norm_layers += 1
            log_arg_lip = ch.clamp_min(args.gamma - spectral_norm_val, 1e-8)
            current_lip_bar_sum += -current_mu_lip * ch.log(log_arg_lip)

            layer_key = f"layer_{layer_idx}"

            if spectral_norm_val >= args.gamma:
                # Violates the gamma constraint
                gamma_violation_tracker[layer_key] = gamma_violation_tracker.get(layer_key, 0) + 1
                consecutive_violations = gamma_violation_tracker[layer_key]
                if consecutive_violations >= 3:
                    warnings.warn(
                        f"Warning: {layer_key} has violated gamma {consecutive_violations} times in a row (spectral_norm: {spectral_norm_val:.4f}, gamma: {args.gamma:.4f})"
                    )
                gamma_violations += 1
            else:
                # No violations, reset counter
                gamma_violation_tracker[layer_key] = 0
            layer_idx += 1

    lip_bar = current_lip_bar_sum / max(num_spectral_norm_layers, 1)

    return loss_bar, lip_bar, current_margins, gamma_violations


def make_optimizer_and_schedule(
    args: object, model: ch.nn.Module, checkpoint: dict[str, Any], params: list[Any] | None
) -> tuple[Any | SGD, ch.optim.Optimizer | None]:
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`
        params (list|None) : a list of parameters that should be updatable, all
            other params will not update. If ``None``, update all params

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    global device

    # Make optimizer
    param_list = model.parameters() if params is None else params
    optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.mixed_precision:
        model.to("cuda")
        model, optimizer = amp.initialize(model, optimizer, "O1")
    else:
        model.to(device=device)

    # Make schedule
    schedule = None
    if args.custom_lr_multiplier == "cyclic":
        eps = args.epochs
        lr_func = lambda t: np.interp([t], [0, eps * 4 // 15, eps], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_multiplier:
        cs = args.custom_lr_multiplier
        periods = eval(cs) if type(cs) is str else cs
        if args.lr_interpolation == "linear":
            lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:

            def lr_func(ep):
                for milestone, lr in reversed(periods):
                    if ep >= milestone:
                        return lr
                return 1.0

        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            schedule.load_state_dict(checkpoint["schedule"])
        except:
            steps_to_take = checkpoint["epoch"]
            print(f"Could not load schedule (was probably LambdaLR). Stepping {steps_to_take} times instead...")
            for i in range(steps_to_take):
                schedule.step()

        if "amp" in checkpoint and checkpoint["amp"] not in [None, "N/A"]:
            amp.load_state_dict(checkpoint["amp"])

        # TODO: see if there's a smarter way to do this
        # TODO: see what's up with loading fp32 weights and then MP training
        if args.mixed_precision:
            model.load_state_dict(checkpoint["model"])

    return optimizer, schedule


def eval_model(args: object, model: ch.nn.Module, loader: Iterable, wandb_run=None) -> dict[str, Any]:
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
    """
    check_required_args(args, eval_only=True)
    start_time = time.time()

    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model)

    # Nat eval loop
    nat_prec1, nat_loss, nat_prec5, _, _, _, _ = _model_loop(
        args=args,
        loop_type="val",
        loader=loader,
        model=model,
        opt=None,
        epoch=0,
        adv=False,
        current_mu=0,
        lambda_dual=None,
    )

    adv_prec1, adv_loss, adv_prec5 = float("nan"), float("nan"), float("nan")
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if has_attr(args, "eps") else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, "attack_lr") else None
        # Adv eval loop
        adv_prec1, adv_loss, adv_prec5, _, _, _, _ = _model_loop(
            args=args,
            loop_type="val",
            loader=loader,
            model=model,
            opt=None,
            epoch=0,
            adv=True,
            current_mu=0,
            lambda_dual=None,
        )

    wandb_run.log(
        {
            "eval/nat_prec1": nat_prec1,
            "eval/nat_prec5": nat_prec5,
            "eval/adv_prec1": adv_prec1,
            "eval/adv_prec5": adv_prec5,
            "eval/nat_loss": nat_loss,
            "eval/adv_loss": adv_loss,
            "eval/time": time.time() - start_time,
        }
    )
    return {
        "nat_prec1": nat_prec1,
        "nat_prec5": nat_prec5,
        "adv_prec1": adv_prec1,
        "adv_prec5": adv_prec5,
        "nat_loss": nat_loss,
        "adv_loss": adv_loss,
    }


def train_model(
    args,
    model,
    loaders,
    *,
    checkpoint=None,
    dp_device_ids=None,
    update_params=None,
    disable_no_grad=False,
    wandb_run=None,
) -> ch.nn.Module:
    """
    Main function for training a model.
    """
    global global_step, device

    check_required_args(args)  # Argument sanity check
    for p in ["eps", "attack_lr", "custom_eps_multiplier"]:
        setattr(args, p, eval(str(getattr(args, p))) if has_attr(args, p) else None)
    if args.custom_eps_multiplier is not None:
        eps_periods = args.custom_eps_multiplier
        args.custom_eps_multiplier = lambda t: np.interp([t], *zip(*eps_periods))[0]

    # Initial setup
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    if ch.cuda.is_available():
        model = ch.nn.DataParallel(model, device_ids=dp_device_ids).cuda()
    else:
        model.to(device=device)

    best_prec1, start_epoch = (0, 0)

    if checkpoint:
        start_epoch = checkpoint["epoch"]
        prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
        best_prec1 = (
            checkpoint[prec1_key]
            if prec1_key in checkpoint
            else _model_loop(
                args=args,
                loop_type="val",
                loader=val_loader,
                model=model,
                opt=None,
                epoch=start_epoch - 1,
                adv=args.adv_train,
                current_mu=0,
                current_mu_lip=0,
                lambda_dual=None,
            )[0]
        )

    # Timestamp for training start time
    start_time = time.time()

    # Initialize mu, mu_lip, lambda_dual for the training loop
    current_mu = args.mu
    current_mu_lip = args.mu_lip
    lambda_dual = ch.tensor([0.0]).to(device=device)

    # k:v is layer_{layer_idx}: violations in a row
    gamma_violation_tracker = {}

    for epoch in range(start_epoch, args.epochs):
        is_warmup_phase = epoch < args.warmup_epochs

        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # train for one epoch
        (
            train_prec1,
            train_loss,
            train_prec5,
            updated_lambda_dual,
            train_avg_margins,
            train_gamma_violation_avg,
            metrics_cache,
        ) = _model_loop(
            args=args,
            loop_type="train",
            loader=train_loader,
            model=model,
            opt=opt,
            epoch=epoch,
            adv=args.adv_train,
            current_mu=current_mu,
            current_mu_lip=current_mu_lip,
            lambda_dual=lambda_dual,
            is_warmup_phase=is_warmup_phase,
            gamma_violation_tracker=gamma_violation_tracker,
        )
        lambda_dual = updated_lambda_dual

        if metrics_cache:
            for log_item in metrics_cache:
                wandb_run.log(log_item, commit=True)

        last_epoch = epoch == (args.epochs - 1)
        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)

        ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()
        with ctx:
            nat_prec1, nat_loss, nat_prec5, _, _, _, _ = _model_loop(args, "val", val_loader, model, None, epoch, False)

        adv_val_prec1, adv_val_prec5, adv_val_loss = float("nan"), float("nan"), float("nan")
        should_adv_eval = args.adv_eval or args.adv_train
        if should_adv_eval:
            (
                adv_val_prec1,
                adv_val_loss,
                adv_val_prec5,
                _,
                adv_avg_margins,
                gamma_violation_avg,
                _,
            ) = _model_loop(
                args=args,
                loop_type="val",
                loader=val_loader,
                model=model,
                opt=None,
                epoch=epoch,
                adv=True,
                current_mu=current_mu,
                current_mu_lip=current_mu_lip,
                lambda_dual=lambda_dual,
                is_warmup_phase=is_warmup_phase,
                gamma_violation_tracker=gamma_violation_tracker,
            )

        # remember best prec@1 and save checkpoint
        prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
        our_prec1 = adv_val_prec1 if args.adv_train else nat_prec1
        is_best = our_prec1 > best_prec1
        best_prec1 = max(our_prec1, best_prec1)

        sd_info = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "schedule": (schedule and schedule.state_dict()),
            "epoch": epoch + 1,
            "amp": amp.state_dict() if args.mixed_precision else None,
            "mu": current_mu,
            "mu_lip": current_mu_lip,
            "lambda_dual": lambda_dual.cpu().numpy(),
            "iter_step": global_step,
            "gamma_violation_tracker": gamma_violation_tracker,
            prec1_key: our_prec1,
        }

        global_step += 1
        wandb_log_dict = {
            "epoch_train": epoch,
            "epoch_val": epoch,
            "global_step": global_step,
            "val/nat_loss": nat_loss,
            "val/nat_prec1": nat_prec1,
            "val/nat_prec5": nat_prec5,
            "train/epoch_loss": train_loss,
            "train/epoch_prec1": train_prec1,
            "train/epoch_prec5": train_prec5,
            "train/epochs_lambda_dual": lambda_dual,
            "train/epochs_avg_margins": train_avg_margins,
            "train/epochs_gamma_violation_avg": train_gamma_violation_avg,
            "val/avg_margins": adv_avg_margins,
            "val/adv_loss": adv_val_loss,
            "val/adv_prec1": adv_val_prec1,
            "val/adv_prec5": adv_val_prec5,
            "total_time": time.time() - start_time,
        }

        if not is_warmup_phase:
            wandb_log_dict["val/gamma_violations_avg"] = gamma_violation_avg

        wandb_run.log(wandb_log_dict, commit=True)

        def save_checkpoint(filename: str) -> None:
            ckpt_save_path = os.path.join(args.out_dir, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        # If we are at a saving epoch (or the last epoch), save a checkpoint
        if should_save_ckpt or last_epoch:
            save_checkpoint(ckpt_at_epoch(epoch))

        # Update the latest and best checkpoints (overrides old one)
        save_checkpoint(consts.CKPT_NAME_LATEST)
        if is_best:
            save_checkpoint(consts.CKPT_NAME_BEST)

        if schedule:
            schedule.step()

        # update barrier strengths at the end of each epoch
        current_mu *= 0.9
        current_mu_lip *= 0.9

    return model


def _model_loop(
    args,
    loop_type,
    loader,
    model,
    opt,
    epoch: int,
    adv,
    current_mu=0,
    current_mu_lip=0,
    lambda_dual=None,
    is_warmup_phase=False,
    gamma_violation_tracker=None,
):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    This function contains the core training or evaluation loop over a single
    epoch.

    Args:
        args (object) : A list of arguments.
        loop_type (str) : 'train' or 'val'. Determines whether to train or
                          evaluate the model.
        loader (iterable) : Data loader for the current loop.
        model (AttackerModel) : The model to train or evaluate.
        opt (torch.optim.Optimizer, optional) : The optimizer for training. Defaults to None.
        epoch (int) : The current epoch number.
        adv (bool) : Whether to perform adversarial training/evaluation.
        current_mu (float) : Current value for the margin barrier loss parameter mu.
        current_mu_lip (float) : Current value for the Lipschitz barrier loss parameter mu_lip.
        lambda_dual (torch.Tensor) : Dual variable for the barrier loss.
        is_warmup_phase (bool) : Flag to indicate if the model is in the warmup phase.

    Returns:
        A tuple containing:
        - top1.avg (float): The average Top-1 accuracy over the loop.
        - losses.avg (float): The average total loss over the loop.
        - top5.avg (float): The average Top-5 accuracy over the loop.
        - lambda_dual (torch.Tensor): The updated dual variable.
        - avg_margins.avg (float): The average margin over the loop.
        - gamma_violation_meter.avg (float): The average number of gamma violations.
        - metrics_cache (list): A list of dictionaries containing iteration-level logs.
    """
    global global_step, device

    if not loop_type in ["train", "val"]:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = loop_type == "train"

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    margin_barrier_losses = AverageMeter()
    lip_barrier_losses = AverageMeter()
    avg_margins = AverageMeter()
    gamma_violation_meter = AverageMeter()

    metrics_cache = []  # Cache for iteration-level metrics

    prec = "NatPrec" if not adv else "AdvPrec"
    loop_msg = "Train" if loop_type == "train" else "Val"

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, "custom_train_loss")
    train_criterion = args.custom_train_loss if has_custom_train_loss else ch.nn.CrossEntropyLoss()

    has_custom_adv_loss = has_attr(args, "custom_adv_loss")
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        attack_kwargs = {
            "constraint": args.constraint,
            "eps": eps,
            "step_size": args.attack_lr,
            "iterations": args.attack_steps,
            "random_start": args.random_start,
            "custom_loss": adv_criterion,
            "random_restarts": random_restarts,
            "use_best": bool(args.use_best),
        }

    iterator = tqdm(enumerate(loader), total=len(loader), bar_format="{l_bar}{bar:30}{r_bar}", leave=False)
    for i, (inp, target) in iterator:
        global_step += 1

        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output, final_inp = model(inp, target=target, make_adv=adv, **attack_kwargs)

        model_logits = output[0] if (type(output) is tuple) else output

        # CE loss
        ce_loss = train_criterion(model_logits, target)
        ce_loss = ce_loss.mean() if len(ce_loss.shape) > 0 else ce_loss

        # Initialize barrier losses
        loss_bar = ch.tensor(0.0, device=device)
        lip_bar = ch.tensor(0.0, device=device)
        current_margins = None
        gamma_violations = 0

        if is_train and not is_warmup_phase:
            loss_bar, lip_bar, current_margins, gamma_violations = calculate_barrier_losses(
                model, model_logits, target, args, current_mu, current_mu_lip, gamma_violation_tracker
            )

            margin_barrier_losses.update(loss_bar.item(), inp.size(0))
            lip_barrier_losses.update(lip_bar.item(), inp.size(0))
            if current_margins is not None:
                avg_margins.update(current_margins.mean(), inp.size(0))

            if lambda_dual is not None and lambda_dual.shape[0] != inp.shape[0]:
                lambda_dual = ch.zeros(inp.shape[0], device=device)
        elif not is_train:
            _, _, _, gamma_violations = calculate_barrier_losses(model, model_logits, target, args, 0, 0, None)

        gamma_violation_meter.update(gamma_violations, 1)

        # Total Loss
        loss = ce_loss + loss_bar + lip_bar

        # measure accuracy and record loss
        top1_acc = float("nan")
        top5_acc = float("nan")
        try:
            maxk = min(5, model_logits.shape[-1])
            if has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, target)
            else:
                prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
                prec1, prec5 = prec1[0], prec5[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1, inp.size(0))
            top5.update(prec5, inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
        except Exception as e:
            warnings.warn(f"Failed to calculate the accuracy. Error: {e}")
            losses.update(loss.item(), inp.size(0))

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term = args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()
            opt.step()

            # Dual ascent update for Lambda
            if not is_warmup_phase and current_margins is not None:
                lambda_dual = (lambda_dual + args.eta * current_margins.detach()).clamp_min_(0)

        if is_train:
            iteration_log_dict = {
                "iter_step": global_step,
                "global_step": global_step,
                "train/iter_loss": loss.item(),
                "train/iter_ce_loss": ce_loss.item(),
                "train/iter_prec1": prec1,
                "train/iter_prec5": prec5,
            }
            if not is_warmup_phase:
                iteration_log_dict["train/iter_margin_barrier_loss"] = loss_bar.item()
                iteration_log_dict["train/iter_lip_barrier_loss"] = lip_bar.item()
                iteration_log_dict["train/iter_gamma_violations"] = gamma_violations

                if current_margins is not None:
                    iteration_log_dict["train/iter_average_margin"] = current_margins.mean().item()

            metrics_cache.append(iteration_log_dict)

        # Update the description for the progress bar
        desc = f"{loop_msg} E{epoch:>3d}"
        base_stats = {"Loss": f"{losses.avg:.3f}", "CE": f"{ce_loss.item():.3f}", "Prec1": f"{top1_acc:.3f}"}

        if not is_warmup_phase and is_train:
            barrier_stats = {"MgnB": f"{margin_barrier_losses.avg:.3f}", "LipB": f"{lip_barrier_losses.avg:.3f}"}
            if avg_margins.count > 0:
                barrier_stats["Mgn"] = f"{avg_margins.avg:.3f}"
            base_stats.update(barrier_stats)

        stats_str = " | ".join(f"{k} {v}" for k, v in base_stats.items())
        iterator.set_description(f"{desc} | {stats_str}")

    return (
        top1.avg,
        losses.avg,
        top5.avg,
        lambda_dual,
        avg_margins.avg,
        gamma_violation_meter.avg,
        metrics_cache,
    )
