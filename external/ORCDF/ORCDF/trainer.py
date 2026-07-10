import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


class Trainer:
    def __init__(self, model, loaders, data_proc, args, logger):
        self.model = model
        self.args = args
        self.logger = logger
        self.data_proc = data_proc
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.device = next(model.parameters()).device
        self.loss_func = nn.BCELoss()
        self.optimizer = optim.Adam(
            [
                {"params": self.model.extractor.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": self.model.inter_func.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            ]
        )
        os.makedirs(args.log_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        start = time.time()
        self.model.get_flip_graph()

        for stu, exer, kn_emb, label in self.train_loader:
            self.optimizer.zero_grad()
            pred, extra_loss = self.model(stu, exer, kn_emb)
            pred_loss = self.loss_func(pred.clamp(1e-6, 1.0 - 1e-6), label)
            loss = pred_loss + extra_loss
            loss.backward()
            self.optimizer.step()
            self.model.inter_func.monotonicity()
            total_loss += loss.item()

        mean_loss = total_loss / max(1, len(self.train_loader))
        self.logger.info(f"Epoch {epoch} | T: {time.time() - start:.1f}s | Loss: {mean_loss:.4f}")
        return mean_loss

    @staticmethod
    def _safe_auc(labels, preds):
        try:
            return float(roc_auc_score(labels, preds))
        except ValueError:
            return 0.5

    def evaluate(self, loader, split_name):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for stu, exer, kn_emb, label in loader:
                pred, _ = self.model(stu, exer, kn_emb)
                preds.extend(pred.view(-1).cpu().numpy())
                labels.extend(label.cpu().numpy())

        if not labels:
            return {"auc": 0.0, "acc": 0.0, "rmse": 0.0}

        preds = np.array(preds)
        labels = np.array(labels)
        auc = self._safe_auc(labels, preds)
        acc = float(accuracy_score(labels, preds > 0.5))
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        self.logger.info(f"[{split_name}] AUC: {auc:.4f} | ACC: {acc:.4f} | RMSE: {rmse:.4f}")
        return {"auc": auc, "acc": acc, "rmse": rmse}

    def predict_and_evaluate(self, loader, split_name):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for stu, exer, kn_emb, label in loader:
                pred, _ = self.model(stu, exer, kn_emb)
                preds.extend(pred.view(-1).cpu().numpy())
                labels.extend(label.cpu().numpy())

        if not labels:
            return {"auc": 0.0, "acc": 0.0, "rmse": 0.0}, np.array([]), np.array([])

        preds = np.array(preds)
        labels = np.array(labels)
        metrics = {
            "auc": self._safe_auc(labels, preds),
            "acc": float(accuracy_score(labels, preds > 0.5)),
            "rmse": float(np.sqrt(mean_squared_error(labels, preds))),
        }
        self.logger.info(
            f"[{split_name}] AUC: {metrics['auc']:.4f} | "
            f"ACC: {metrics['acc']:.4f} | RMSE: {metrics['rmse']:.4f}"
        )
        return metrics, preds, labels

    def evaluate_conflict_groups(self, split_name, preds=None, labels=None):
        if split_name != "Test" or self.data_proc is None:
            return []
        if not hasattr(self.data_proc, "test_sample_conflict"):
            self.logger.info("[Test-SampleConflict] skipped: no SampleConflict metadata")
            return []

        if preds is None or labels is None:
            _, preds, labels = self.predict_and_evaluate(self.test_loader, split_name)

        meta_list = self.data_proc.test_sample_conflict
        eligible_indices = [
            i for i, meta in enumerate(meta_list)
            if meta.get("concept_count", 0) > 0 and meta.get("group") != "Unknown"
        ]
        excluded_samples = len(meta_list) - len(eligible_indices)
        grouped_indices = self._split_ordered_indices(
            eligible_indices,
            [meta_list[i]["score"] for i in eligible_indices],
            self.args.conflict_num_groups,
        )
        self.logger.info(
            f"[{split_name}-SampleConflict] excluded_samples={excluded_samples} "
            f"(no usable train-set concept-level conflict evidence)"
        )

        results = []
        for group_idx, indices in enumerate(grouped_indices, start=1):
            if not indices:
                continue
            group_preds = preds[indices]
            group_labels = labels[indices]
            auc = self._safe_auc(group_labels, group_preds)
            acc = float(accuracy_score(group_labels, group_preds > 0.5))
            mean_score = float(np.mean([meta_list[i]["score"] for i in indices]))
            mean_support = float(np.mean([meta_list[i]["support"] for i in indices]))
            mean_concepts = float(np.mean([meta_list[i]["concept_count"] for i in indices]))
            group_name = f"G{group_idx:02d}"
            row = {
                "experiment_name": self.args.experiment_name,
                "split": split_name,
                "analysis": "SampleConflict",
                "group": group_name,
                "samples": len(indices),
                "avg_sample_conflict": mean_score,
                "avg_kc_support": mean_support,
                "avg_known_concepts": mean_concepts,
                "auc": auc,
                "acc": acc,
            }
            results.append(row)
            self.logger.info(
                f"[{split_name}-SampleConflict-{group_name}] AUC: {auc:.4f} | "
                f"ACC: {acc:.4f} | samples={len(indices)} | "
                f"avg_sample_conflict={mean_score:.4f} | "
                f"avg_kc_support={mean_support:.2f} | avg_known_concepts={mean_concepts:.2f}"
            )

        return results

    @staticmethod
    def _split_ordered_indices(indices, scores, num_groups):
        if not indices:
            return []
        ordered_pairs = sorted(zip(indices, scores), key=lambda x: x[1])
        ordered_indices = [idx for idx, _ in ordered_pairs]
        return [
            list(group)
            for group in np.array_split(np.array(ordered_indices, dtype=int), num_groups)
            if len(group) > 0
        ]

    def save_summary(self, final_results):
        summary_path = os.path.join(self.args.log_dir, "metrics_summary.csv")
        if final_results:
            fieldnames = []
            for row in final_results:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        else:
            fieldnames = ["experiment_name", "split", "auc", "acc", "rmse"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_results)
        self.logger.info(f"Saved final metrics summary to {summary_path}")
