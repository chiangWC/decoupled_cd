import os
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ORCDF.config import parse_args
    from ORCDF.dataset import CognitiveDataProcessor
    from ORCDF.model import ORCDFNet
    from ORCDF.trainer import Trainer
    from ORCDF.utils import get_device, set_seed, setup_logger
else:
    from .config import parse_args
    from .dataset import CognitiveDataProcessor
    from .model import ORCDFNet
    from .trainer import Trainer
    from .utils import get_device, set_seed, setup_logger


def main():
    args = parse_args()
    logger = setup_logger(args.log_dir)
    logger.info(f"Experiment Args: {vars(args)}")

    device = get_device()
    set_seed(args.seed)
    logger.info(f"Running on {device} with Seed {args.seed}")

    data_proc = CognitiveDataProcessor(args, logger)
    loaders = data_proc.get_loaders()

    model = ORCDFNet(
        student_n=data_proc.num_students,
        exer_n=data_proc.num_exercises,
        knowledge_n=data_proc.num_concepts,
        latent_dim=args.latent_dim,
        gcn_layers=args.gcn_layers,
        keep_prob=args.keep_prob,
        if_type=args.if_type,
        mode=args.mode,
        flip_ratio=args.flip_ratio,
        ssl_temp=args.ssl_temp,
        ssl_weight=args.ssl_weight,
        prednet_len1=args.prednet_len1,
        prednet_len2=args.prednet_len2,
        dropout=args.dropout,
        device=device,
    ).to(device)
    model.get_graph_dict(data_proc.graph_dict)

    trainer = Trainer(model, loaders, data_proc, args, logger)

    best_auc = 0.0
    patience = 0
    best_model_path = os.path.join(args.log_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(epoch)
        val_metrics = trainer.evaluate(trainer.val_loader, "Valid")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"*** New Best Valid AUC: {best_auc:.4f} ***")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("\n>>> Final Testing...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    metrics, preds, labels = trainer.predict_and_evaluate(trainer.test_loader, "Test")
    import csv as _csv
    import numpy as _np
    with open(os.path.join(args.log_dir, "test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        _writer = _csv.writer(f)
        _writer.writerow(["prob", "label"])
        _writer.writerows(zip(_np.asarray(preds).ravel().tolist(), _np.asarray(labels).ravel().tolist()))
    final_results = [{"experiment_name": args.experiment_name, "split": "Test", **metrics}]
    final_results.extend(trainer.evaluate_conflict_groups("Test", preds=preds, labels=labels))
    trainer.save_summary(final_results)


if __name__ == "__main__":
    main()
