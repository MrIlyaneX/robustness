"""
The main file, which exposes the robustness command-line tool, detailed in
:doc:`this walkthrough <../example_usage/cli_usage>`.
"""

import os
from argparse import ArgumentParser

import cox
import cox.utils
from dotenv import load_dotenv

import wandb

try:
    from . import __version__, defaults
    from .barrier_train import eval_model as eval_barrier_model
    from .barrier_train import train_model as train_barrier_model
    from .datasets import DATASETS
    from .defaults import check_and_fill_args
    from .model_utils import make_and_restore_model
    from .tools import constants, helpers
    from .train import eval_model as eval_standard_model
    from .train import train_model as train_standard_model
except:
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)


def main(args):
    """Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    """
    load_dotenv()

    wandb_run = wandb.init(
        project="robustness_barrier_training",
        config=args.dict,
        name=f"imp_gamma_{args.gamma}-delta_{args.delta}",
    )

    wandb.define_metric("epoch_val")
    wandb.define_metric("epoch_train")
    wandb.define_metric("iter_step")
    wandb.define_metric("val/*", step_metric="epoch_val")
    wandb.define_metric("train/epoch*", step_metric="epoch_train")
    wandb.define_metric("train/iter*", step_metric="iter_step")

    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers, args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch, dataset=dataset, resume_path=args.resume)
    if "module" in dir(model):
        model = model.module

    print(args)
    if args.eval_only:
        if args.loss_type == "margin_barrier":
            return eval_barrier_model(args, model, val_loader)
        else:  # Default or 'ce'
            return eval_standard_model(args, model, val_loader, store=None)

    if not args.resume_optimizer:
        checkpoint = None

    if args.loss_type == "margin_barrier":
        print(f"Using barrier training with loss type: {args.loss_type}")
        model = train_barrier_model(args, model, loaders, checkpoint=checkpoint, wandb_run=wandb_run)
    else:  # Default to 'ce' or any other standard training
        print(f"Using standard training with loss type: {args.loss_type}")
        model = train_standard_model(args, model, loaders, store=None, checkpoint=checkpoint)

    wandb.finish()
    return model


def setup_args(args):
    """
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    """

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only:
        assert args.resume is not None, "Must provide a resume path if only evaluating"
    return args


if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)

    final_model = main(args)
