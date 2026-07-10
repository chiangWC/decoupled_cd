import os
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from SVGCD.config import parse_args
    from SVGCD.dataset import CognitiveDataProcessor
    from SVGCD.model import SVGCDNet
    from SVGCD.trainer import Trainer
    from SVGCD.utils import get_device, set_seed, setup_logger
else:
    from .config import parse_args
    from .dataset import CognitiveDataProcessor
    from .model import SVGCDNet
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

    model = SVGCDNet(
        student_n=data_proc.num_students,
        exer_n=data_proc.num_exercises,
        knowledge_n=data_proc.num_concepts,
        args=args,
        pos_graph=data_proc.correct_adj,
        neg_graph=data_proc.wrong_adj,
        device=device,
    ).to(device)
    trainer = Trainer(model, loaders, data_proc, args, logger)

    best_auc = 0.0
    patience = 0
    best_model_path = os.path.join(args.log_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        main_loss, loss_dict = trainer.train_epoch(epoch)
        val_metrics = trainer.evaluate(trainer.val_loader, "Valid")
        logger.info(
            "Epoch %d summary | MainLoss: %.4f | %s"
            % (epoch, main_loss, " ".join(f"{k}={float(v):.4f}" for k, v in loss_dict.items()))
        )
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

    test_metrics = trainer.evaluate(trainer.test_loader, "Test")
    final_results = [
        {"experiment_name": args.experiment_name, "split": "Test", **test_metrics}
    ]

    trainer.save_summary(final_results)


if __name__ == "__main__":
    main()
