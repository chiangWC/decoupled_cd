"""ORCDF + decoupling plugin runner. Trains, evaluates, dumps predictions + mastery."""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ORCDF.config import parse_args as base_parse_args
from ORCDF.dataset import CognitiveDataProcessor
from ORCDF.plugin import (
    DecoupledORCDF,
    add_plugin_parameters_to_optimizer,
    build_concept_graph,
    build_tkc_mask,
)
from ORCDF.trainer import Trainer
from ORCDF.utils import get_device, set_seed, setup_logger


def parse_all():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--plugin-decouple", action="store_true")
    pre.add_argument("--plugin-aux-weight", type=float, default=0.0)
    pre.add_argument("--plugin-aux-detach-item-difficulty", action="store_true")
    pre.add_argument("--plugin-aux-warmup-fraction", type=float, default=0.0)
    plugin_args, remaining = pre.parse_known_args()
    saved = sys.argv
    sys.argv = [saved[0]] + remaining
    try:
        args = base_parse_args()
    finally:
        sys.argv = saved
    args.plugin_decouple = plugin_args.plugin_decouple
    args.plugin_aux_weight = plugin_args.plugin_aux_weight
    args.plugin_aux_detach_item_difficulty = (
        plugin_args.plugin_aux_detach_item_difficulty
    )
    args.plugin_aux_warmup_fraction = plugin_args.plugin_aux_warmup_fraction
    return args


def main():
    args = parse_all()
    logger = setup_logger(args.log_dir)
    logger.info(f"Args: {vars(args)}")
    device = get_device()
    set_seed(args.seed)

    proc = CognitiveDataProcessor(args, logger)
    loaders = proc.get_loaders()

    model = DecoupledORCDF(
        student_n=proc.num_students, exer_n=proc.num_exercises, knowledge_n=proc.num_concepts,
        latent_dim=args.latent_dim, gcn_layers=args.gcn_layers, keep_prob=args.keep_prob,
        if_type=args.if_type, mode=args.mode, flip_ratio=args.flip_ratio, ssl_temp=args.ssl_temp,
        ssl_weight=args.ssl_weight, prednet_len1=args.prednet_len1, prednet_len2=args.prednet_len2,
        dropout=args.dropout, device=device,
        decouple=args.plugin_decouple,
        aux_weight=args.plugin_aux_weight,
        aux_detach_item_difficulty=args.plugin_aux_detach_item_difficulty,
        aux_warmup_fraction=args.plugin_aux_warmup_fraction,
    ).to(device)
    model.get_graph_dict(proc.graph_dict)
    if args.plugin_decouple:
        dtype = model.extractor.dtype
        model.set_decouple_tensors(build_tkc_mask(proc, dtype), build_concept_graph(proc, dtype))

    trainer = Trainer(model, loaders, proc, args, logger)
    add_plugin_parameters_to_optimizer(
        trainer.optimizer,
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_func = nn.BCELoss()

    def train_epoch(epoch):
        model.train()
        total = 0.0
        t0 = time.time()
        model.get_flip_graph()
        for stu, exer, kn_emb, label in trainer.train_loader:
            stu, exer, kn_emb, label = stu.to(device), exer.to(device), kn_emb.to(device), label.to(device)
            trainer.optimizer.zero_grad()
            pred, extra = model(
                stu,
                exer,
                kn_emb,
                label=label,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            loss = loss_func(pred.clamp(1e-6, 1 - 1e-6), label) + extra
            loss.backward()
            trainer.optimizer.step()
            model.inter_func.monotonicity()
            total += loss.item()
        logger.info(f"Epoch {epoch} | {time.time()-t0:.1f}s | Loss {total/max(1,len(trainer.train_loader)):.4f}")

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
                logger.info("Early stop.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    metrics, preds, labels = trainer.predict_and_evaluate(trainer.test_loader, "Test")
    logger.info(f"FINAL {metrics}")
    with open(os.path.join(args.log_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prob", "label"])
        w.writerows(zip(np.asarray(preds).ravel().tolist(), np.asarray(labels).ravel().tolist()))
    np.save(os.path.join(args.log_dir, "mastery.npy"), model.mastery_matrix())
    with open(os.path.join(args.log_dir, "id_maps.json"), "w") as f:
        json.dump({"stu_ids": [str(x) for x in proc.stu_ids], "cpt_ids": [str(x) for x in proc.cpt_ids]}, f)
    with open(os.path.join(args.log_dir, "metrics.json"), "w") as f:
        json.dump({
            **metrics,
            "decouple": args.plugin_decouple,
            "aux_weight": args.plugin_aux_weight,
            "aux_detach_item_difficulty": args.plugin_aux_detach_item_difficulty,
            "aux_warmup_fraction": args.plugin_aux_warmup_fraction,
        }, f)


if __name__ == "__main__":
    main()
