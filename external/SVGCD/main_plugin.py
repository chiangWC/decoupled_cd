"""SVGCD + mastery-aux-BCE plugin runner. Dumps predictions + mastery for DOA."""
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mastery_auxiliary import MasteryAuxiliaryObjective

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
    pre.add_argument("--plugin-aux-weight", type=float, default=0.0)
    pre.add_argument("--plugin-aux-detach-item-difficulty", action="store_true")
    pre.add_argument("--plugin-aux-warmup-fraction", type=float, default=0.0)
    pa, remaining = pre.parse_known_args()
    saved = sys.argv
    sys.argv = [saved[0]] + remaining
    try:
        args = base_parse_args()
    finally:
        sys.argv = saved
    args.plugin_aux_weight = pa.plugin_aux_weight
    args.plugin_aux_detach_item_difficulty = (
        pa.plugin_aux_detach_item_difficulty
    )
    args.plugin_aux_warmup_fraction = pa.plugin_aux_warmup_fraction
    return args


def main():
    args = parse_all()
    logger = setup_logger(args.log_dir)
    logger.info(f"Args: {vars(args)}")
    device = get_device()
    set_seed(args.seed)
    proc = CognitiveDataProcessor(args, logger)
    loaders = proc.get_loaders()
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

    best_auc, patience = 0.0, 0
    best_path = os.path.join(args.log_dir, "best_model.pth")
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch)
        val = trainer.evaluate(trainer.val_loader, "Valid")
        if val["auc"] > best_auc:
            best_auc, patience = val["auc"], 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= args.patience:
                logger.info("Early stop."); break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test = trainer.evaluate(trainer.test_loader, "Test")
    logger.info(f"FINAL {test}")
    # dump test predictions in row order
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in trainer.test_loader:
            batch = trainer._move_batch(batch)
            p = model.forward_test(batch["stu_id"], batch["exer_id"], batch["Q_mat"]).flatten()
            preds.extend(p.cpu().numpy()); labels.extend(batch["label"].cpu().numpy())
    with open(os.path.join(args.log_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prob", "label"]); w.writerows(zip(preds, labels))
    np.save(os.path.join(args.log_dir, "mastery.npy"), model.mastery_matrix())
    with open(os.path.join(args.log_dir, "id_maps.json"), "w") as f:
        json.dump({"stu_ids": [str(x) for x in proc.stu_ids], "cpt_ids": [str(x) for x in proc.cpt_ids]}, f)
    with open(os.path.join(args.log_dir, "metrics.json"), "w") as f:
        json.dump({
            **test,
            "aux_weight": args.plugin_aux_weight,
            "aux_detach_item_difficulty": args.plugin_aux_detach_item_difficulty,
            "aux_warmup_fraction": args.plugin_aux_warmup_fraction,
        }, f)


if __name__ == "__main__":
    main()
