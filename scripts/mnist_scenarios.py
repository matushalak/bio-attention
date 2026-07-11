#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
from torchvision import datasets, transforms

from prelude import get_device, load_dicts, save_dicts, save_results_to_csv, startup_folders
from src.composer import Arrow_DS, Cue_DS, IOR_DS, Popout_DS, Recognition_DS, Search_DS, Tracking_DS
from src.conductor import AttentionTrain
from src.model import AttentionModel
from src.utils import build_loaders, get_n_parameters, plot_all, plot_loss_all


TASK_COMPOSERS = {
    "IOR": IOR_DS,
    "Arrow": Arrow_DS,
    "Cue": Cue_DS,
    "Tracking": Tracking_DS,
    "Recognition": Recognition_DS,
    "Search": Search_DS,
    "Popout": Popout_DS,
}

TRAIN_LOSS_SLICES = {
    "IOR": (None, None),
    "Arrow": (None, None),
    "Cue": (None, slice(1, None)),
    "Tracking": (slice(1, None), slice(1, None)),
    "Recognition": (slice(1, None), None),
    "Search": (None, slice(1, None)),
    "Popout": (None, None),
}

EVAL_LOSS_SLICES = {name: ((-1,), (-1,)) for name in TASK_COMPOSERS}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def make_jsonable(value):
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, tuple):
        return [make_jsonable(v) for v in value]
    if isinstance(value, list):
        return [make_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: make_jsonable(v) for k, v in value.items()}
    return value


def load_mnist_config(pretrained_dir: Path, data_path: str, eval_mode: bool = False, iter_multiplier: int = 1):
    model_params = load_dicts(str(pretrained_dir), "model_params")
    train_params = load_dicts(str(pretrained_dir), "train_params")
    tasks = load_dicts(str(pretrained_dir), "tasks")

    for name, composer in TASK_COMPOSERS.items():
        tasks[name]["composer"] = composer
        tasks[name]["datasets"] = []
        tasks[name]["dataloaders"] = []
        tasks[name]["loss_s"] = EVAL_LOSS_SLICES[name] if eval_mode else TRAIN_LOSS_SLICES[name]

    tasks["Arrow"]["params"]["directory"] = data_path
    if iter_multiplier != 1:
        multiply_iterations(tasks, iter_multiplier)

    return model_params, train_params, tasks


def multiply_iterations(tasks: OrderedDict, factor: int) -> None:
    for name, task in tasks.items():
        params = task["params"]
        if "n_iter" in params:
            params["n_iter"] *= factor
        if "fix_attend" in params:
            params["fix_attend"] = [int(v) * factor for v in params["fix_attend"]]
        if name == "IOR" and "n_attend" in params:
            params["n_attend"] *= factor


def build_mnist_tasks(tasks: OrderedDict, data_path: str, batch_size: int):
    train_ds_full = datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    train_ds, valid_ds = random_split(train_ds_full, (50000, 10000))

    device, num_workers, pin_memory = get_device()
    for name in tasks:
        composer = tasks[name]["composer"]
        params = tasks[name]["params"]
        tasks[name]["datasets"].append(composer(train_ds, **params))
        tasks[name]["datasets"].append(composer(valid_ds, **params))
        tasks[name]["datasets"].append(composer(test_ds, **params))
        tasks[name]["datasets"][1].build_valid_test()
        tasks[name]["datasets"][2].build_valid_test()
        tasks[name]["dataloaders"] = build_loaders(
            tasks[name]["datasets"],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return device


def write_scores(scores, task_names, output_dir: Path, prefix: str) -> None:
    rows = []
    for task_name, score in zip(task_names, scores):
        rows.append({
            "task": task_name,
            "CEi": score[0],
            "CEe": score[1],
            "PixErr": score[2],
            "AttAcc": score[3],
            "ClsAcc": score[4],
        })

    csv_path = output_dir / f"{prefix}_metrics.csv"
    json_path = output_dir / f"{prefix}_metrics.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "CEi", "CEe", "PixErr", "AttAcc", "ClsAcc"])
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Saved metrics to {csv_path}")
    print(f"Saved metrics to {json_path}")


def set_eval_slices(tasks: OrderedDict) -> None:
    for name in tasks:
        tasks[name]["loss_s"] = EVAL_LOSS_SLICES[name]


def trainable_parameters(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def freeze_forward_encoder(model: AttentionModel) -> None:
    modules = [model.conv_blocks, model.conv_frnn, model.frnn_blocks, model.fc_out]
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False


def train_classifier_prephase(model, optimizer, scheduler, tasks, device, n_epochs, max_grad_norm, logger):
    task_names = list(tasks.keys())
    n_batches = len(tasks[task_names[0]]["dataloaders"][0])
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        epoch_start = time.time()
        loaders = [iter(tasks[name]["dataloaders"][0]) for name in task_names]
        loss_sum = {name: 0.0 for name in task_names if name != "IOR"}
        seen = {name: 0 for name in task_names if name != "IOR"}

        for _ in range(n_batches):
            for j, task_name in enumerate(task_names):
                if task_name == "IOR":
                    continue
                x, y, _, _, _ = next(loaders[j])
                x, y = x.to(device), y.to(device)
                model.initiate_forward(x.size(0))
                logits, _ = model.for_forward(x[:, -1])
                loss = cross_entropy(logits, y[:, -1] if y.ndim > 1 else y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                loss_sum[task_name] += loss.item()
                seen[task_name] += 1

        scheduler.step()
        parts = " ".join(f"{name}={loss_sum[name] / max(seen[name], 1):.4f}" for name in loss_sum)
        logger.info(f"classifier prephase epoch {epoch + 1}/{n_epochs} ({time.time() - epoch_start:.2f}s): {parts}")


def run_eval(args):
    pretrained_dir = Path(args.pretrained_dir)
    output_dir, logger = startup_folders(args.output_root, name=args.name)
    output_dir = Path(output_dir)
    model_params, train_params, tasks = load_mnist_config(pretrained_dir, args.data_path, eval_mode=True)
    train_params["batch_size"] = args.batch_size or train_params["batch_size"]
    device = build_mnist_tasks(tasks, args.data_path, train_params["batch_size"])
    model = AttentionModel(**model_params)
    model.load_state_dict(torch.load(pretrained_dir / "model.pth", map_location=device))
    conductor = AttentionTrain(model, None, None, tasks, logger, str(output_dir))
    scores = conductor.eval(device, "test", False, retrack=True)
    write_scores(scores, list(tasks.keys()), output_dir, "test")


def run_long(args):
    pretrained_dir = Path(args.pretrained_dir)
    output_dir, logger = startup_folders(args.output_root, name=args.name)
    output_dir = Path(output_dir)
    model_params, train_params, tasks = load_mnist_config(
        pretrained_dir,
        args.data_path,
        eval_mode=False,
        iter_multiplier=args.iter_multiplier,
    )
    train_params["n_epochs"] = args.epochs or train_params["n_epochs"]
    train_params["batch_size"] = args.batch_size or train_params["batch_size"]
    train_params["exase"] = args.name
    train_params["iter_multiplier"] = args.iter_multiplier
    device = build_mnist_tasks(tasks, args.data_path, train_params["batch_size"])

    logger.info(f"train_params\n{train_params}")
    logger.info(f"tasks\n{make_jsonable({k: v['params'] for k, v in tasks.items()})}")
    model = AttentionModel(**model_params)
    logger.info(f"Model has {get_n_parameters(model):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], weight_decay=train_params["l2"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])
    conductor = AttentionTrain(model, optimizer, scheduler, tasks, logger, str(output_dir), train_params["max_grad_norm"], args.save_intermediate)

    if not args.skip_pre_eval:
        plot_all(10, model, tasks, str(output_dir), "_pre", device, logger, False)
        conductor.eval(device)
    conductor.train(train_params["n_epochs"], device, args.verbose)
    plot_loss_all(conductor, str(output_dir))
    set_eval_slices(tasks)
    scores = conductor.eval(device, "test", False, retrack=True)
    write_scores(scores, list(tasks.keys()), output_dir, "test")

    save_dicts(tasks, str(output_dir), "tasks", logger)
    save_dicts(train_params, str(output_dir), "train_params", logger)
    save_dicts(model_params, str(output_dir), "model_params", logger)
    torch.save(model.state_dict(), output_dir / "model.pth")
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pth")
    for i, task in enumerate(tasks):
        save_results_to_csv(conductor.loss_records[i], output_dir / f"loss_{task}.csv", ["labels", "masks", "last_label"], logger)
        save_results_to_csv(conductor.valid_records[i], output_dir / f"valid_{task}.csv", ["CEi", "CEe", "PixErr", "AttAcc", "ClsAcc"], logger)


def run_frozen_encoder(args):
    pretrained_dir = Path(args.pretrained_dir)
    output_dir, logger = startup_folders(args.output_root, name=args.name)
    output_dir = Path(output_dir)
    model_params, train_params, tasks = load_mnist_config(pretrained_dir, args.data_path, eval_mode=False)
    train_params["n_epochs"] = args.decoder_epochs or train_params["n_epochs"]
    train_params["batch_size"] = args.batch_size or train_params["batch_size"]
    train_params["encoder_pretrain_epochs"] = args.encoder_epochs
    train_params["exase"] = args.name
    device = build_mnist_tasks(tasks, args.data_path, train_params["batch_size"])

    model = AttentionModel(**model_params)
    logger.info(f"Model has {get_n_parameters(model):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], weight_decay=train_params["l2"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])

    logger.info("Starting classifier-only encoder prephase. IOR is skipped because it has unordered multi-object labels.")
    train_classifier_prephase(
        model,
        optimizer,
        scheduler,
        tasks,
        device,
        args.encoder_epochs,
        train_params["max_grad_norm"],
        logger,
    )
    torch.save(model.state_dict(), output_dir / "encoder_pretrained_model.pth")

    freeze_forward_encoder(model)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=train_params["lr"], weight_decay=train_params["l2"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])
    conductor = AttentionTrain(model, optimizer, scheduler, tasks, logger, str(output_dir), train_params["max_grad_norm"], args.save_intermediate)
    conductor.eval(device)
    conductor.train(train_params["n_epochs"], device, args.verbose)
    plot_loss_all(conductor, str(output_dir))
    set_eval_slices(tasks)
    scores = conductor.eval(device, "test", False, retrack=True)
    write_scores(scores, list(tasks.keys()), output_dir, "test")

    save_dicts(tasks, str(output_dir), "tasks", logger)
    save_dicts(train_params, str(output_dir), "train_params", logger)
    save_dicts(model_params, str(output_dir), "model_params", logger)
    torch.save(model.state_dict(), output_dir / "model.pth")
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pth")
    for i, task in enumerate(tasks):
        save_results_to_csv(conductor.loss_records[i], output_dir / f"loss_{task}.csv", ["labels", "masks", "last_label"], logger)
        save_results_to_csv(conductor.valid_records[i], output_dir / f"valid_{task}.csv", ["CEi", "CEe", "PixErr", "AttAcc", "ClsAcc"], logger)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("eval-pretrained", "long-iterations", "frozen-encoder"))
    parser.add_argument("--pretrained-dir", default="./pretrained/mnist_v2")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--output-root", default="./results")
    parser.add_argument("--name", default="mnist_scenario")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iter-multiplier", type=int, default=10)
    parser.add_argument("--encoder-epochs", type=int, default=96)
    parser.add_argument("--decoder-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1821)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument("--skip-pre-eval", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    return args


def main():
    args = parse_args()
    if args.scenario == "eval-pretrained":
        run_eval(args)
    elif args.scenario == "long-iterations":
        run_long(args)
    elif args.scenario == "frozen-encoder":
        run_frozen_encoder(args)


if __name__ == "__main__":
    main()
