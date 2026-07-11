"""SVGCD plugin runner with validation-only training and guarded evaluation."""
import argparse
import io
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mastery_auxiliary import MasteryAuxiliaryObjective
from scripts.plugin_campaign import (
    CandidateStore,
    prepare_evaluation_resources,
    run_candidate_training,
    select_evaluation_loader,
    sha256_bytes,
    sha256_file,
    snapshot_evaluation_artifacts,
    validate_training_initialization,
    write_evaluation_artifacts,
)

from SVGCD.config import parse_args as base_parse_args
from SVGCD.dataset import CognitiveDataProcessor
from SVGCD.model import SVGCDNet
from SVGCD.trainer import Trainer, train_three_stage_batch
from SVGCD.utils import get_device, set_seed, setup_logger


class AuxSVGCD(SVGCDNet):
    def __init__(
        self,
        *args,
        aux_weight=0.0,
        aux_detach_item_difficulty=False,
        aux_warmup_fraction=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_weight = float(aux_weight)
        self.mastery_auxiliary = (
            MasteryAuxiliaryObjective(
                aux_weight=aux_weight,
                detach_item_difficulty=aux_detach_item_difficulty,
                warmup_fraction=aux_warmup_fraction,
            )
            if aux_weight > 0.0
            else None
        )

    def auxiliary_loss(
        self,
        stu_id,
        exer_id,
        Q_mat,
        label,
        *,
        epoch,
        total_epochs,
    ):
        if self.mastery_auxiliary is None:
            return label.detach().new_zeros((), dtype=torch.float32)
        stu_abit, exer_diff = self.graph_representations()
        return self.mastery_auxiliary(
            self.prednet_stu(stu_abit[stu_id]),
            self.prednet_exer(exer_diff[exer_id]),
            Q_mat[exer_id],
            label,
            epoch=epoch,
            total_epochs=total_epochs,
        )

    def mastery_matrix(self):
        with torch.no_grad():
            stu_abit, _ = self.graph_representations()
            return torch.sigmoid(self.prednet_stu(stu_abit)).float().cpu().numpy()


def parse_all():
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--plugin-mode", choices=("train", "evaluate"), required=True)
    pre.add_argument("--plugin-aux-weight", type=float, default=0.0)
    pre.add_argument("--plugin-aux-detach-item-difficulty", action="store_true")
    pre.add_argument("--plugin-aux-warmup-fraction", type=float, default=0.0)
    pre.add_argument("--plugin-init-checkpoint", type=Path)
    pre.add_argument(
        "--plugin-init-mode",
        choices=("random", "baseline-finetune"),
        default="random",
    )
    pre.add_argument(
        "--plugin-lr-multiplier", type=float, choices=(0.25, 1.0), default=1.0
    )
    pre.add_argument("--plugin-checkpoint", type=Path)
    pre.add_argument("--plugin-eval-split", choices=("valid", "test"))
    pre.add_argument("--plugin-test-ledger-dir", type=Path)
    pre.add_argument("--plugin-selection-json", type=Path)
    pre.add_argument("--plugin-frozen-config-id")
    pre.add_argument("--plugin-model-name", default="svgcd")
    pre.add_argument("--plugin-doa-seed", type=int, default=42)
    pre.add_argument("--plugin-min-responses", type=int, default=3)
    pre.add_argument("--plugin-max-pairs-per-concept", type=int, default=100_000)
    pre.add_argument("--plugin-split-seed", type=int, default=2024)
    pre.add_argument("--plugin-data-protocol", choices=("standard", "holdout"))
    pre.add_argument("--plugin-dataset-name")
    pre.add_argument(
        "--plugin-q-matrix-file", type=Path, default=Path("Q_matrix.csv")
    )
    initialization_options = (
        "--plugin-init-checkpoint",
        "--plugin-init-mode",
        "--plugin-lr-multiplier",
    )
    initialization_explicit = any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
        for option in initialization_options
    )
    pa, remaining = pre.parse_known_args()
    if pa.plugin_mode == "evaluate" and initialization_explicit:
        pre.error("plugin initialization arguments are train-only")
    if (
        pa.plugin_mode == "train"
        and pa.plugin_init_mode == "baseline-finetune"
        and pa.plugin_init_checkpoint is None
    ):
        pre.error("baseline-finetune requires --plugin-init-checkpoint")
    if (
        pa.plugin_mode == "train"
        and pa.plugin_init_mode == "baseline-finetune"
        and (not pa.plugin_data_protocol or not pa.plugin_dataset_name)
    ):
        pre.error(
            "baseline-finetune requires --plugin-data-protocol and "
            "--plugin-dataset-name"
        )
    if (
        pa.plugin_mode == "train"
        and pa.plugin_init_mode == "random"
        and pa.plugin_init_checkpoint is not None
    ):
        pre.error("random initialization cannot use --plugin-init-checkpoint")
    if pa.plugin_mode == "evaluate":
        if pa.plugin_checkpoint is None or pa.plugin_eval_split is None:
            pre.error("evaluate mode requires --plugin-checkpoint and --plugin-eval-split")
        if pa.plugin_eval_split == "test" and (
            pa.plugin_test_ledger_dir is None or pa.plugin_selection_json is None
        ):
            pre.error(
                "test evaluation requires --plugin-test-ledger-dir and "
                "--plugin-selection-json"
            )
    saved = sys.argv
    sys.argv = [saved[0]] + remaining
    try:
        args = base_parse_args()
    finally:
        sys.argv = saved
    for name, value in vars(pa).items():
        setattr(args, name, value)
    args.plugin_initialization_explicit = initialization_explicit
    return args


def plugin_config(args):
    config = {
        "aux_weight": args.plugin_aux_weight,
        "aux_detach_item_difficulty": args.plugin_aux_detach_item_difficulty,
        "aux_warmup_fraction": args.plugin_aux_warmup_fraction,
    }
    if args.plugin_mode == "train":
        checkpoint_sha256 = getattr(args, "plugin_init_checkpoint_sha256", None)
        if checkpoint_sha256 is None and args.plugin_init_checkpoint is not None:
            checkpoint_sha256 = sha256_file(Path(args.plugin_init_checkpoint))
        config.update(
            {
                "init_mode": args.plugin_init_mode,
                "baseline_checkpoint_sha256": checkpoint_sha256,
                "base_lr": getattr(args, "plugin_base_lr", args.lr),
                "lr_multiplier": args.plugin_lr_multiplier,
            }
        )
    return config


def recipe_config(args):
    recipe = {
        "aux_weight": args.plugin_aux_weight,
        "aux_detach_item_difficulty": args.plugin_aux_detach_item_difficulty,
        "aux_warmup_fraction": args.plugin_aux_warmup_fraction,
    }
    if (
        args.plugin_initialization_explicit
        or args.plugin_init_mode != "random"
        or args.plugin_lr_multiplier != 1.0
    ):
        recipe.update(
            {
                "init_mode": args.plugin_init_mode,
                "base_lr": getattr(args, "plugin_base_lr", args.lr),
                "lr_multiplier": args.plugin_lr_multiplier,
            }
        )
    return recipe


def backbone_config(args):
    return {
        field: getattr(args, field)
        for field in (
            "emb_dim",
            "dnn_units",
            "dropout_rate",
            "n_gnn_layer",
            "cl_tau",
            "cl_weight",
            "beta",
        )
    }


def resolve_plugin_q_matrix(args):
    path = Path(args.plugin_q_matrix_file)
    if not path.is_absolute():
        path = Path(args.data_dir) / path
    if not path.is_file():
        raise FileNotFoundError(f"plugin Q-matrix does not exist: {path}")
    return path


def campaign_protocol(args, q_matrix_bytes):
    protocol = {
        "split": "valid",
        "seed": args.seed,
        "doa_seed": args.plugin_doa_seed,
        "min_responses": args.plugin_min_responses,
        "max_pairs_per_concept": args.plugin_max_pairs_per_concept,
        "split_seed": args.plugin_split_seed,
        "q_matrix_sha256": sha256_bytes(q_matrix_bytes),
    }
    if bool(args.plugin_data_protocol) != bool(args.plugin_dataset_name):
        raise ValueError(
            "plugin data protocol and dataset name must be provided together"
        )
    if args.plugin_data_protocol:
        protocol.update(
            {
                "data_protocol": args.plugin_data_protocol,
                "dataset_name": args.plugin_dataset_name,
            }
        )
    return protocol


def processor_id_maps(proc):
    return {
        "stu_ids": [str(value) for value in proc.stu_ids],
        "exer_ids": [str(value) for value in proc.exer_ids],
        "cpt_ids": [str(value) for value in proc.cpt_ids],
    }


def load_evaluation_checkpoint(model, checkpoint_bytes, device):
    model.load_state_dict(
        torch.load(io.BytesIO(checkpoint_bytes), map_location=device)
    )


def load_training_initialization_checkpoint(model, checkpoint_bytes, device):
    state_dict = torch.load(io.BytesIO(checkpoint_bytes), map_location=device)
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("training initialization checkpoint must be a state dict")
    target_state = model.state_dict()
    source_keys = set(state_dict)
    target_keys = set(target_state)
    allowed_missing = {
        key
        for key in ("mastery_auxiliary.scale",)
        if key in target_keys
    }
    missing = target_keys - source_keys
    unexpected = source_keys - target_keys
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            "training initialization checkpoint schema mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    for key in sorted(source_keys):
        source_value = state_dict[key]
        target_value = target_state[key]
        if (
            not torch.is_tensor(source_value)
            or source_value.shape != target_value.shape
            or source_value.dtype != target_value.dtype
        ):
            raise RuntimeError(
                f"training initialization checkpoint tensor schema mismatch: {key}"
            )
    merged_state = dict(state_dict)
    for key in missing:
        merged_state[key] = target_state[key]
    model.load_state_dict(merged_state, strict=True)


def prepare_training_initialization(
    *,
    args,
    model,
    proc,
    device,
    q_matrix_path,
    q_matrix_bytes,
    protocol,
):
    args.plugin_base_lr = args.lr
    args.plugin_init_checkpoint_sha256 = None
    if args.plugin_init_mode == "baseline-finetune":
        if args.plugin_init_checkpoint is None:
            raise ValueError("baseline-finetune requires --plugin-init-checkpoint")
        init_snapshot = snapshot_evaluation_artifacts(
            args.plugin_init_checkpoint,
            q_matrix_path,
        )
        validate_training_initialization(
            checkpoint_path=args.plugin_init_checkpoint,
            snapshot=init_snapshot,
            q_matrix_bytes=q_matrix_bytes,
            processor_id_maps=processor_id_maps(proc),
            campaign_protocol=protocol,
        )
        load_training_initialization_checkpoint(
            model,
            init_snapshot.checkpoint_bytes,
            device,
        )
        args.plugin_init_checkpoint_sha256 = sha256_bytes(
            init_snapshot.checkpoint_bytes
        )
    trainer_args = argparse.Namespace(**vars(args))
    trainer_args.lr = args.plugin_base_lr * args.plugin_lr_multiplier
    return trainer_args


def main():
    args = parse_all()
    q_matrix_path = resolve_plugin_q_matrix(args)
    artifact_snapshot = None
    id_maps_payload = None
    evaluation_plugin_config = None
    if args.plugin_mode == "evaluate":
        artifact_snapshot = snapshot_evaluation_artifacts(
            args.plugin_checkpoint,
            q_matrix_path,
        )
        q_matrix_bytes = artifact_snapshot.q_matrix_bytes
        id_maps_payload = json.loads(artifact_snapshot.id_maps_bytes)
    else:
        q_matrix_bytes = q_matrix_path.read_bytes()
    protocol = campaign_protocol(args, q_matrix_bytes)
    logger = setup_logger(args.log_dir)
    logger.info(f"Args: {vars(args)}")
    device = get_device()
    set_seed(args.seed)

    def load_resources():
        processor = CognitiveDataProcessor(
            args,
            logger,
            split_mode=(args.plugin_eval_split if args.plugin_mode == "evaluate" else "train"),
            id_maps_payload=id_maps_payload,
            q_matrix_bytes=q_matrix_bytes,
        )
        return processor, processor.get_loaders()

    if args.plugin_mode == "evaluate":
        (
            (proc, loaders),
            test_claim_bytes,
            evaluation_plugin_config,
        ) = prepare_evaluation_resources(
            split=args.plugin_eval_split,
            load_resources=load_resources,
            checkpoint_path=args.plugin_checkpoint,
            plugin_config=plugin_config(args),
            backbone_config=backbone_config(args),
            protocol=protocol,
            ledger_dir=args.plugin_test_ledger_dir,
            selection_path=args.plugin_selection_json,
            supplied_frozen_config_id=args.plugin_frozen_config_id,
            route_root=PROJECT_ROOT,
            argv=sys.argv,
            artifact_snapshot=artifact_snapshot,
            return_test_claim_bytes=True,
            return_plugin_config=True,
        )
    else:
        proc, loaders = load_resources()
        test_claim_bytes = None
    model = AuxSVGCD(
        student_n=proc.num_students, exer_n=proc.num_exercises, knowledge_n=proc.num_concepts,
        args=args, pos_graph=proc.correct_adj, neg_graph=proc.wrong_adj, device=device,
        aux_weight=args.plugin_aux_weight,
        aux_detach_item_difficulty=args.plugin_aux_detach_item_difficulty,
        aux_warmup_fraction=args.plugin_aux_warmup_fraction,
    ).to(device)
    trainer_args = args
    if args.plugin_mode == "train":
        trainer_args = prepare_training_initialization(
            args=args,
            model=model,
            proc=proc,
            device=device,
            q_matrix_path=q_matrix_path,
            q_matrix_bytes=q_matrix_bytes,
            protocol=protocol,
        )
    trainer = Trainer(model, loaders, proc, trainer_args, logger)

    def train_epoch(epoch):
        model.train()
        t0 = time.time()
        last = 0.0
        for batch in trainer.train_loader:
            batch = trainer._move_batch(batch)

            def add_auxiliary_loss(loss_main):
                if model.mastery_auxiliary is None:
                    return loss_main
                return loss_main + model.auxiliary_loss(
                    batch["stu_id"],
                    batch["exer_id"],
                    batch["Q_mat"],
                    batch["label"],
                    epoch=epoch,
                    total_epochs=args.epochs,
                )

            loss_main, _ = train_three_stage_batch(
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                batch=batch,
                main_loss_augmenter=add_auxiliary_loss,
            )
            last = float(loss_main.item())
        logger.info(f"Epoch {epoch} | {time.time()-t0:.1f}s | Main {last:.4f}")

    if args.plugin_mode == "train":
        candidate_store = CandidateStore(
            output_dir=Path(args.log_dir),
            model_name=args.plugin_model_name,
            plugin_config=plugin_config(args),
            recipe_config=recipe_config(args),
            backbone_config=backbone_config(args),
            protocol=protocol,
        )

        def snapshot_candidate(epoch, validation):
            candidate_store.save(
                epoch=epoch,
                validation_metrics=validation,
                state_dict=model.state_dict(),
                mastery=model.mastery_matrix(),
                id_maps=processor_id_maps(proc),
            )

        summary = run_candidate_training(
            epochs=args.epochs,
            patience=args.patience,
            train_epoch=train_epoch,
            evaluate_validation=lambda: trainer.evaluate(trainer.val_loader, "Valid"),
            snapshot_candidate=snapshot_candidate,
        )
        logger.info(f"TRAIN COMPLETE {summary}")
        return

    load_evaluation_checkpoint(model, artifact_snapshot.checkpoint_bytes, device)
    evaluation_loader = select_evaluation_loader(loaders, args.plugin_eval_split)
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in evaluation_loader:
            batch = trainer._move_batch(batch)
            probability = model.forward_test(
                batch["stu_id"], batch["exer_id"], batch["Q_mat"]
            ).flatten()
            predictions.extend(probability.cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())
    prediction_array = np.asarray(predictions)
    label_array = np.asarray(labels)
    metrics = {
        "auc": trainer._safe_auc(label_array, prediction_array),
        "acc": float(np.mean(np.round(prediction_array) == label_array)),
        "rmse": float(np.sqrt(np.mean((label_array - prediction_array) ** 2))),
    }
    write_evaluation_artifacts(
        output_dir=Path(args.log_dir),
        split=args.plugin_eval_split,
        metrics=metrics,
        predictions=predictions,
        labels=labels,
        test_claim_bytes=test_claim_bytes,
        interaction_rows=proc.evaluation_rows(args.plugin_eval_split),
        mastery=model.mastery_matrix(),
        id_maps=processor_id_maps(proc),
        metadata={
            "checkpoint_sha256": sha256_bytes(artifact_snapshot.checkpoint_bytes),
            "source_id_maps_sha256": sha256_bytes(
                artifact_snapshot.id_maps_bytes
            ),
            "selection_json": (
                str(args.plugin_selection_json) if args.plugin_selection_json else None
            ),
            "plugin_config": evaluation_plugin_config,
            "backbone_config": backbone_config(args),
            "protocol": protocol,
        },
    )
    logger.info(f"EVALUATION COMPLETE {metrics}")


if __name__ == "__main__":
    main()
