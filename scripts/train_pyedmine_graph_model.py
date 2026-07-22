from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--pyedmine-root", type=Path, required=True)
    bootstrap.add_argument("--model", choices=("RCD", "HyperCD"), required=True)
    known, _ = bootstrap.parse_known_args()
    train_dir = (
        known.pyedmine_root / "examples" / "cognitive_diagnosis" / "train"
    )
    sys.path.insert(0, str(known.pyedmine_root))
    sys.path.insert(0, str(train_dir))
    from set_params import (
        setup_clip_args,
        setup_common_args,
        setup_grad_acc_args,
        setup_scheduler_args,
    )

    parser = argparse.ArgumentParser(
        parents=[
            bootstrap,
            setup_common_args(),
            setup_scheduler_args(),
            setup_clip_args(),
            setup_grad_acc_args(),
        ]
    )
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--evaluate_batch_size", type=int, default=2048)
    parser.add_argument("--optimizer_type", default="adam")
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--save_model", default="True")
    parser.add_argument("--use_wandb", default="False")
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--dim_feature", type=int, default=512)
    parser.add_argument("--dim_emb", type=int, default=16)
    parser.add_argument("--leaky", type=float, default=0.8)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    from edmine.dataset.CognitiveDiagnosisDataset import (
        BasicCognitiveDiagnosisDataset,
    )
    from edmine.trainer.DLCognitiveDiagnosisTrainer import (
        DLCognitiveDiagnosisTrainer,
    )
    from edmine.utils.parse import str2bool
    from edmine.utils.use_torch import set_seed

    params = vars(args).copy()
    params["pyedmine_root"] = str(params["pyedmine_root"])
    for name in (
        "save_model",
        "use_wandb",
        "use_early_stop",
        "use_multi_metrics",
        "debug_mode",
        "use_cpu",
        "enable_scheduler",
        "enable_clip_grad",
    ):
        value = params[name]
        if isinstance(value, str):
            params[name] = str2bool(value)
    set_seed(params["seed"])
    if args.model == "RCD":
        from config.rcd import config_rcd
        from edmine.model.cognitive_diagnosis_model.RCD import RCD

        global_params, global_objects = config_rcd(params)
        model = RCD(global_params, global_objects)
    else:
        from config.hyper_cd import config_hyper_cd

        module = importlib.import_module(
            "edmine.model.cognitive_diagnosis_model.HyperCD"
        )
        if module.MODEL_NAME != "HyperCDF":
            raise RuntimeError(
                f"Unexpected HyperCD model key: {module.MODEL_NAME}"
            )
        module.MODEL_NAME = "HyperCD"
        from edmine.model.registry import MODEL_REGISTRY

        MODEL_REGISTRY["HyperCD"] = module.HyperCD
        global_params, global_objects = config_hyper_cd(params)
        model = module.HyperCD(global_params, global_objects)
    dataset_train = BasicCognitiveDiagnosisDataset(
        global_params["datasets_config"]["train"], global_objects
    )
    dataset_valid = BasicCognitiveDiagnosisDataset(
        global_params["datasets_config"]["valid"], global_objects
    )
    global_objects["data_loaders"] = {
        "train_loader": DataLoader(
            dataset_train, batch_size=args.train_batch_size, shuffle=True
        ),
        "valid_loader": DataLoader(
            dataset_valid, batch_size=args.evaluate_batch_size, shuffle=False
        ),
    }
    global_objects["models"] = {
        args.model: model.to(global_params["device"])
    }
    DLCognitiveDiagnosisTrainer(global_params, global_objects).train()


if __name__ == "__main__":
    run(parse_args())
