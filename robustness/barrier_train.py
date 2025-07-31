import os
import time
import warnings

import dill
import numpy as np
import torch as ch
import torch.nn as nn
import wandb
from cox.utils import Parameters
from torch.optim import SGD, lr_scheduler
from torchvision.utils import make_grid

from .barrier_loss import logarithmic_barrier_loss, per_sample_margin_loss
from .cifar_models.resnet import get_spectral_norm
from .tools import constants as consts
from .tools import helpers
from .tools.buffered_logger import BufferedLogger
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


def check_required_args(args, eval_only=False):
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
    def check_args(args_list):
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


def calculate_barrier_losses(model, model_logits, target, args, current_mu, current_mu_lip, device):
    """Calculate margin and Lipschitz barrier losses."""
    loss_bar, current_margins = logarithmic_barrier_loss(model_logits, target, args.delta, current_mu)

    # Lipschitz Barrier Loss
    num_spectral_norm_layers = 0
    current_lip_bar_sum = ch.tensor(0.0, device=device)

    for m in model.modules():
        spectral_norm_val = get_spectral_norm(m)
        if spectral_norm_val is not None:
            num_spectral_norm_layers += 1
            log_arg_lip = ch.clamp_min(args.gamma - spectral_norm_val + 1e-6, 1e-8)
            current_lip_bar_sum += -current_mu_lip * ch.log(log_arg_lip)

    lip_bar = current_lip_bar_sum / max(num_spectral_norm_layers, 1)

    return loss_bar, lip_bar, current_margins


def make_optimizer_and_schedule(args, model, checkpoint, params):
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
    # Make optimizer
    param_list = model.parameters() if params is None else params
    optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.mixed_precision:
        model.to("cuda")
        model, optimizer = amp.initialize(model, optimizer, "O1")
    else:
        if ch.cuda.is_available():
            print("Using cuda")
            model.to("cuda")
        elif ch.backends.mps.is_available():
            print("Using mps")
            model.to("mps")
        else:
            print("Using cpu")
            model.to("cpu")

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


def eval_model(args, model, loader, store, wandb_run=None):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
    """
    check_required_args(args, eval_only=True)
    start_time = time.time()
    print("Using Barrier Eval Method")

    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    writer = store.tensorboard if store else None

    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model)

    eval_global_step = 0

    prec1, nat_loss, _, _ = _model_loop(
        args,
        "val",
        loader,
        model,
        None,
        0,
        False,
        writer,
        current_mu=0,
        current_mu_lip=0,
        lambda_dual=None,
        wandb_run=wandb_run,
        global_step=eval_global_step,
    )

    adv_prec1, adv_loss = float("nan"), float("nan")
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if has_attr(args, "eps") else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, "attack_lr") else None
        adv_prec1, adv_loss, _, _ = _model_loop(
            args,
            "val",
            loader,
            model,
            None,
            0,
            True,
            writer,
            current_mu=0,
            current_mu_lip=0,
            lambda_dual=None,
            wandb_run=wandb_run,
            global_step=eval_global_step,
        )
    log_info = {
        "epoch": 0,
        "nat_prec1": prec1,
        "adv_prec1": adv_prec1,
        "nat_loss": nat_loss,
        "adv_loss": adv_loss,
        "train_prec1": float("nan"),
        "train_loss": float("nan"),
        "time": time.time() - start_time,
    }

    if wandb_run:
        wandb_run.log(
            {
                "eval/epoch": 0,
                "eval/nat_prec1": prec1,
                "eval/adv_prec1": adv_prec1,
                "eval/nat_loss": nat_loss,
                "eval/adv_loss": adv_loss,
                "eval/time": time.time() - start_time,
            },
            step=0,
        )

    # Log info into the logs table
    if store:
        store[consts.LOGS_TABLE].append_row(log_info)
    return log_info


def train_model(
    args,
    model,
    loaders,
    *,
    checkpoint=None,
    dp_device_ids=None,
    store=None,
    update_params=None,
    disable_no_grad=False,
    wandb_run=None,
):
    """
    Main function for training a model.

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_lr_multplier (str)
                If given, use a custom LR schedule, formed by multiplying the
                    original ``lr`` (format: [(epoch, LR_MULTIPLIER),...])
            lr_interpolation (str)
                How to drop the learning rate, either ``step`` or ``linear``,
                    ignored unless ``custom_lr_multiplier`` is provided.
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            custom_eps_multiplier (str, *required if adv_train or adv_eval*)
                If given, then set epsilon according to a schedule by
                multiplying the given eps value by a factor at each epoch. Given
                in the same format as ``custom_lr_multiplier``, ``[(epoch,
                MULTIPLIER)..]``
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            custom_accuracy (function)
                If given, should be a function that takes in model outputs
                and model targets and outputs a top1 and top5 accuracy, will
                displayed instead of conventional accuracies
            regularizer (function, optional)
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

            delta (float, *required*)
                Margin parameter for the margin barrier loss.
            gamma (float, *required*)
                Maximum allowed spectral norm for layers.
            mu (float, *required*)
                Initial strength (weight) of the margin barrier loss.
            mu_lip (float, *required*)
                Initial strength (weight) of the Lipschitz barrier loss.
            eta (float, *required*)
                Learning rate for the dual variable update (lambda).
            warmup_epochs (int, *required*)
                Number of epochs to train only with CE loss before activating barrier losses.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)`
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        dp_device_ids (list|None) : if not ``None``, a list of device ids to
            use for DataParallel.
        store (cox.Store) : a cox store for logging training progress
        update_params (list) : list of parameters to use for training, if None
            then all parameters in the model are used (useful for transfer
            learning)
        disable_no_grad (bool) : if True, then even model evaluation will be
            run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())

        wandb_run (wandb.Run): Weights & Biases logger
    """
    print("Using Barrier Train Method")

    # Logging setup
    writer = store.tensorboard if store else None
    logger = BufferedLogger(os.path.join(args.out_dir, "progress_logs"))
    prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)

    # Reformat and read arguments
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
        device = "cuda"
        model = ch.nn.DataParallel(model, device_ids=dp_device_ids).cuda()
    else:
        if ch.backends.mps.is_available():
            model.to("mps")
            device = "mps"
        else:
            model.to("cpu")
            device = "cpu"

    best_prec1, start_epoch = (0, 0)
    global_step = 0

    if checkpoint:
        start_epoch = checkpoint["epoch"]
        best_prec1 = (
            checkpoint[prec1_key]
            if prec1_key in checkpoint
            else _model_loop(
                args,
                "val",
                val_loader,
                model,
                None,
                start_epoch - 1,
                args.adv_train,
                writer=None,
                current_mu=0,  # Barrier not active during initial eval
                current_mu_lip=0,
                lambda_dual=None,
                wandb_run=wandb_run,
                global_step=global_step,
            )[0]
        )

    # Timestamp for training start time
    start_time = time.time()

    # Initialize mu, mu_lip, lambda_dual for the training loop
    current_mu = args.mu
    current_mu_lip = args.mu_lip
    lambda_dual = ch.tensor([0.0]).to(device=device)

    for epoch in range(start_epoch, args.epochs):
        is_warmup_phase = epoch < args.warmup_epochs

        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        if is_warmup_phase:
            print(f"Warm-up Phase (Barrier Losses Inactive)")
            epoch_mu = 0.0
            epoch_mu_lip = 0.0
        else:
            print(f"Barrier Losses Active (mu: {current_mu:.6f}, mu_lip: {current_mu_lip:.6f})")
            epoch_mu = current_mu
            epoch_mu_lip = current_mu_lip

        # train for one epoch
        train_prec1, train_loss, updated_lambda_dual, _, global_step = _model_loop(
            args,
            "train",
            train_loader,
            model,
            opt,
            epoch,
            args.adv_train,
            writer,
            current_mu=epoch_mu,
            current_mu_lip=epoch_mu_lip,
            lambda_dual=lambda_dual,
            is_warmup_phase=is_warmup_phase,
            logger=logger,
            wandb_run=wandb_run,
            global_step=global_step,
        )
        lambda_dual = updated_lambda_dual

        last_epoch = epoch == (args.epochs - 1)

        # evaluate on validation set
        sd_info = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "schedule": (schedule and schedule.state_dict()),
            "epoch": epoch + 1,
            "amp": amp.state_dict() if args.mixed_precision else None,
            "mu": current_mu,
            "mu_lip": current_mu_lip,
            "lambda_dual": lambda_dual.cpu().numpy(),
            "global_step": global_step,
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = epoch % args.log_iters == 0

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()
            with ctx:
                prec1, nat_loss, _, _, _ = _model_loop(
                    args,
                    "val",
                    val_loader,
                    model,
                    None,
                    epoch,
                    False,
                    writer,
                    current_mu=0,
                    current_mu_lip=0,
                    lambda_dual=None,
                    wandb_run=wandb_run,
                    global_step=global_step,
                )

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            if should_adv_eval:
                adv_val_prec1, adv_val_loss, _, avg_margins, _ = _model_loop(
                    args,
                    "val",
                    val_loader,
                    model,
                    None,
                    epoch,
                    True,
                    writer,
                    current_mu=0,
                    current_mu_lip=0,
                    lambda_dual=None,
                    is_warmup_phase=True,
                    wandb_run=wandb_run,
                    global_step=global_step,
                )

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_val_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)
            sd_info[prec1_key] = our_prec1

            # log every checkpoint
            log_info = {
                "epoch": epoch + 1,
                "nat_prec1": prec1,
                "adv_prec1": adv_val_prec1,
                "nat_loss": nat_loss,
                "adv_loss": adv_val_loss,
                "avg_margins": avg_margins,
                "train_prec1": train_prec1,
                "train_loss": train_loss,
                "time": time.time() - start_time,
            }

            if wandb_run:
                wandb_log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/prec1": train_prec1,
                    "val/nat_loss": nat_loss,
                    "val/nat_prec1": prec1,
                    "total_time": time.time() - start_time,
                }
                if should_adv_eval:
                    wandb_log_dict.update(
                        {
                            "val/adv_loss": adv_val_loss,
                            "val/adv_prec1": adv_val_prec1,
                        }
                    )
                if not is_warmup_phase:
                    wandb_log_dict["val/avg_margins"] = avg_margins

                wandb_run.log(wandb_log_dict, step=global_step)

            # Log info into the logs table
            if store:
                store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch:
                save_checkpoint(ckpt_at_epoch(epoch))

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best:
                save_checkpoint(consts.CKPT_NAME_BEST)

        if schedule:
            schedule.step()
        if has_attr(args, "epoch_hook"):
            args.epoch_hook(model, log_info)

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
    epoch,
    adv,
    writer,
    current_mu,
    current_mu_lip,
    lambda_dual,
    is_warmup_phase=False,
    logger=None,
    wandb_run=None,
    global_step: int = 0,
):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)
        current_mu (float): current strength of the margin barrier loss.
        current_mu_lip (float): current strength of the Lipschitz barrier loss.
        lambda_dual (ch.Tensor): The dual variable tensor, updated in-place.
        is_warmup_phase (bool): True if currently in warm-up, False otherwise.

        wandb_run (wandb.Run): Weights & Biases logger

    Returns:
        The average top1 accuracy and the average loss across the epoch,
        along with the updated lambda_dual and (optionally) average margin.
    """
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
    device = "cuda" if ch.cuda.is_available() else ("mps" if ch.backends.mps.is_available() else "cpu")
    for i, (inp, target) in iterator:
        if is_train:
            global_step += 1

        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output, final_inp = model(inp, target=target, make_adv=adv, **attack_kwargs)

        model_logits = output[0] if (type(output) is tuple) else output
        # print(type(model_logits))

        # CE loss
        ce_loss = train_criterion(model_logits, target)
        ce_loss = ce_loss.mean() if len(ce_loss.shape) > 0 else ce_loss

        # Initialize barrier losses
        loss_bar = ch.tensor(0.0, device=device)
        lip_bar = ch.tensor(0.0, device=device)
        current_margins = None

        if is_train and not is_warmup_phase:
            loss_bar, lip_bar, current_margins = calculate_barrier_losses(
                model, model_logits, target, args, current_mu, current_mu_lip, device
            )

            margin_barrier_losses.update(loss_bar.item(), inp.size(0))
            lip_barrier_losses.update(lip_bar.item(), inp.size(0))
            if current_margins is not None:
                avg_margins.update(current_margins.mean(), inp.size(0))

            if lambda_dual is not None and lambda_dual.shape[0] != inp.shape[0]:
                lambda_dual = ch.zeros(inp.shape[0], device=device)

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

        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image("Nat input", nat_grid, epoch)
            writer.add_image("Adv input", adv_grid, epoch)

        # Update the description for the progress bar
        desc = f"{loop_msg} E{epoch:>3d}"
        base_stats = {"Loss": f"{losses.avg:.3f}", "CE": f"{ce_loss.item():.3f}", f"{prec}1": f"{top1_acc:.3f}"}

        if wandb_run and is_train:
            iteration_log_dict = {
                f"{loop_type}/iter_loss": loss.item(),
                f"{loop_type}/iter_ce_loss": ce_loss.item(),
                f"{loop_type}/iter_{prec.lower()}1": prec1,
            }
            if not is_warmup_phase:
                iteration_log_dict.update(
                    {
                        f"{loop_type}/iter_margin_barrier_loss": loss_bar.item(),
                        f"{loop_type}/iter_lip_barrier_loss": lip_bar.item(),
                    }
                )
                if current_margins is not None:
                    iteration_log_dict[f"{loop_type}/iter_average_margin"] = current_margins.mean().item()
            wandb_run.log(iteration_log_dict, step=global_step, commit=False)  # Commit False for iteration logs

        if not is_warmup_phase and is_train:
            barrier_stats = {"MgnB": f"{margin_barrier_losses.avg:.3f}", "LipB": f"{lip_barrier_losses.avg:.3f}"}
            if avg_margins.count > 0:
                barrier_stats["Mgn"] = f"{avg_margins.avg:.3f}"
            base_stats.update(barrier_stats)

        stats_str = " | ".join(f"{k} {v}" for k, v in base_stats.items())
        desc = f"{desc} | {stats_str}"

        if logger:
            logger.log(desc)
        iterator.set_description(desc)

    # At the end of _model_loop, flush the buffer:
    if logger:
        logger.flush()

    if writer is not None:
        prec_type = "adv" if adv else "nat"
        metrics = {
            f"{prec_type}_{loop_type}_loss": losses.avg,
            f"{prec_type}_{loop_type}_top1": top1.avg,
        }

        if is_train and not is_warmup_phase:
            metrics.update(
                {
                    f"{prec_type}/{loop_type}/margin_barrier_loss": margin_barrier_losses.avg,
                    f"{prec_type}/{loop_type}/lip_barrier_loss": lip_barrier_losses.avg,
                    f"{prec_type}/{loop_type}/average_margin": avg_margins.avg,
                }
            )

        for name, value in metrics.items():
            writer.add_scalar(name, value, epoch)

    return top1.avg, losses.avg, lambda_dual, avg_margins.avg, global_step
