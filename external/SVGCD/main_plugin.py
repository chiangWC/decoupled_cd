"""SVGCD plugin runner with validation-only training and guarded evaluation."""
import argparse
import os
import sys
import time
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
    sha256_file,
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
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--plugin-mode", choices=("train", "evaluate"), required=True)
    pre.add_argument("--plugin-aux-weight", type=float, default=0.0)
    pre.add_argument("--plugin-aux-detach-item-difficulty", action="store_true")
    pre.add_argument("--plugin-aux-warmup-fraction", type=float, default=0.0)
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
    pa, remaining = pre.parse_known_args()
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
    return args


def plugin_config(args):
    return {
        "aux_weight": args.plugin_aux_weight,
        "aux_detach_item_difficulty": args.plugin_aux_detach_item_difficulty,
        "aux_warmup_fraction": args.plugin_aux_warmup_fraction,
    }


def campaign_protocol(args):
    return {
        "split": "valid",
        "seed": args.seed,
        "doa_seed": args.plugin_doa_seed,
        "min_responses": args.plugin_min_responses,
        "max_pairs_per_concept": args.plugin_max_pairs_per_concept,
        "split_seed": args.plugin_split_seed,
    }


def processor_id_maps(proc):
    return {
        "stu_ids": [str(value) for value in proc.stu_ids],
        "exer_ids": [str(value) for value in proc.exer_ids],
        "cpt_ids": [str(value) for value in proc.cpt_ids],
    }


def main():
    args = parse_all()
    logger = setup_logger(args.log_dir)
    logger.info(f"Args: {vars(args)}")
    device = get_device()
    set_seed(args.seed)

    def load_resources():
        id_maps_path = (
            args.plugin_checkpoint.with_name("id_maps.json")
            if args.plugin_mode == "evaluate"
            else None
        )
        processor = CognitiveDataProcessor(
            args,
            logger,
            split_mode=(args.plugin_eval_split if args.plugin_mode == "evaluate" else "train"),
            id_maps_path=id_maps_path,
        )
        return processor, processor.get_loaders()

    if args.plugin_mode == "evaluate":
        proc, loaders = prepare_evaluation_resources(
            split=args.plugin_eval_split,
            load_resources=load_resources,
            checkpoint_path=args.plugin_checkpoint,
            plugin_config=plugin_config(args),
            protocol=campaign_protocol(args),
            ledger_dir=args.plugin_test_ledger_dir,
            selection_path=args.plugin_selection_json,
            supplied_frozen_config_id=args.plugin_frozen_config_id,
            route_root=PROJECT_ROOT,
            argv=sys.argv,
        )
    else:
        proc, loaders = load_resources()
    model = AuxSVGCD(
        student_n=proc.num_students, exer_n=proc.num_exercises, knowledge_n=proc.num_concepts,
        args=args, pos_graph=proc.correct_adj, neg_graph=proc.wrong_adj, device=device,
        aux_weight=args.plugin_aux_weight,
        aux_detach_item_difficulty=args.plugin_aux_detach_item_difficulty,
        aux_warmup_fraction=args.plugin_aux_warmup_fraction,
    ).to(device)
    trainer = Trainer(model, loaders, proc, args, logger)

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
            protocol=campaign_protocol(args),
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

    model.load_state_dict(torch.load(args.plugin_checkpoint, map_location=device))
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
        mastery=model.mastery_matrix(),
        id_maps=processor_id_maps(proc),
        metadata={
            "checkpoint_sha256": sha256_file(args.plugin_checkpoint),
            "selection_json": (
                str(args.plugin_selection_json) if args.plugin_selection_json else None
            ),
            "plugin_config": plugin_config(args),
        },
    )
    logger.info(f"EVALUATION COMPLETE {metrics}")


if __name__ == "__main__":
    main()
