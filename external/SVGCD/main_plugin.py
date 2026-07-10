"""SVGCD + mastery-aux-BCE plugin runner. Dumps predictions + mastery for DOA."""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SVGCD.config import parse_args as base_parse_args
from SVGCD.dataset import CognitiveDataProcessor
from SVGCD.model import SVGCDNet
from SVGCD.trainer import Trainer
from SVGCD.utils import get_device, set_seed, setup_logger


class AuxSVGCD(SVGCDNet):
    def __init__(self, *a, aux_weight=0.0, **k):
        super().__init__(*a, **k)
        self.aux_weight = aux_weight
        if aux_weight > 0.0:
            self.aux_scale = torch.nn.Parameter(torch.tensor(2.0))

    def aux_bce(self, stu_id, exer_id, Q_mat, label):
        stu_abit, exer_diff = self.graph_representations()
        mastery = torch.sigmoid(self.prednet_stu(stu_abit[stu_id]))
        diff = torch.sigmoid(self.prednet_exer(exer_diff[exer_id]))
        q = Q_mat[exer_id]
        per = q * (mastery - diff)
        logit = F.softplus(self.aux_scale) * per.sum(1) / q.sum(1).clamp(min=1.0)
        return F.binary_cross_entropy_with_logits(logit, label.float())

    def mastery_matrix(self):
        with torch.no_grad():
            stu_abit, _ = self.graph_representations()
            return torch.sigmoid(self.prednet_stu(stu_abit)).float().cpu().numpy()


def parse_all():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--plugin-aux-weight", type=float, default=0.0)
    pa, remaining = pre.parse_known_args()
    saved = sys.argv
    sys.argv = [saved[0]] + remaining
    try:
        args = base_parse_args()
    finally:
        sys.argv = saved
    args.plugin_aux_weight = pa.plugin_aux_weight
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
    ).to(device)
    trainer = Trainer(model, loaders, proc, args, logger)

    def train_epoch(epoch):
        model.train()
        t0 = time.time()
        last = 0.0
        for batch in trainer.train_loader:
            batch = trainer._move_batch(batch)
            trainer.optimizer.zero_grad()
            loss_cl, _ = model.cal_loss_cl(**batch); loss_cl.backward(); trainer.optimizer.step()
            loss_kl, _ = model.cal_loss_kl(**batch); loss_kl.backward(); trainer.optimizer.step()
            loss_main, _ = model.cal_loss(**batch)
            if model.aux_weight > 0.0:
                loss_main = loss_main + model.aux_weight * model.aux_bce(
                    batch["stu_id"], batch["exer_id"], batch["Q_mat"], batch["label"])
            loss_main.backward(); trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            last = float(loss_main.item())
        trainer.scheduler.step()
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
        json.dump({**test, "aux_weight": args.plugin_aux_weight}, f)


if __name__ == "__main__":
    main()
