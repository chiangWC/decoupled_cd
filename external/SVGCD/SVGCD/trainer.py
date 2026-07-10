import csv
import os
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def _take_optimizer_step(loss, optimizer, scheduler):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()


def train_three_stage_batch(
    *,
    model,
    optimizer,
    scheduler,
    batch,
    main_loss_augmenter=None,
):
    """Run SVGCD's CL, KL, and main stages with isolated gradients."""
    loss_cl, loss_cl_dict = model.cal_loss_cl(**batch)
    _take_optimizer_step(loss_cl, optimizer, scheduler)

    loss_kl, loss_kl_dict = model.cal_loss_kl(**batch)
    _take_optimizer_step(loss_kl, optimizer, scheduler)

    loss_main, loss_main_dict = model.cal_loss(**batch)
    if main_loss_augmenter is not None:
        loss_main = main_loss_augmenter(loss_main)
    _take_optimizer_step(loss_main, optimizer, scheduler)

    return loss_main, {
        **loss_main_dict,
        **loss_cl_dict,
        **loss_kl_dict,
    }


class Trainer:
    def __init__(self, model, loaders, data_proc, args, logger):
        self.model = model
        self.args = args
        self.logger = logger
        self.data_proc = data_proc
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.device = next(model.parameters()).device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=3 * max(1, len(self.train_loader)),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        os.makedirs(args.log_dir, exist_ok=True)

    def _move_batch(self, batch):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def train_epoch(self, epoch):
        self.model.train()
        start = time.time()
        last_loss_dict = {}
        last_main = 0.0

        for batch in self.train_loader:
            batch = self._move_batch(batch)

            loss_main, last_loss_dict = train_three_stage_batch(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                batch=batch,
            )
            last_main = float(loss_main.item())

        self.logger.info(f"Epoch {epoch} | T: {time.time() - start:.1f}s | MainLoss: {last_main:.4f}")
        return last_main, last_loss_dict

    @staticmethod
    def _safe_auc(labels, preds):
        try:
            return float(roc_auc_score(labels, preds))
        except ValueError:
            return 0.5

    def _parse_longtail_edges(self):
        return [int(x.strip()) for x in self.args.longtail_bin_edges.split(",") if x.strip()]

    def _log_longtail_metrics(self, rows, split_name):
        if not rows:
            return
        self.logger.info(f"[{split_name}] Performance on Long-Tailed Students (RQ2-style)")
        for row in rows:
            max_label = f"{row['max_interactions']}" if row["max_interactions"] is not None else "+"
            self.logger.info(
                "    %s: stu=%d n=%d mean_inter=%.1f AUC=%.4f ACC=%.4f"
                % (
                    f"[{row['min_interactions']},{max_label})",
                    row["student_count"],
                    row["sample_count"],
                    row["mean_interactions"],
                    row["auc"],
                    row["acc"],
                )
            )

    def _longtail_rows(self, student_ids, labels, preds):
        edges = self._parse_longtail_edges()
        counts = self.data_proc.student_interaction_counts
        rows = []
        labels = np.array(labels)
        preds = np.array(preds)
        for i, left in enumerate(edges):
            right = edges[i + 1] if i + 1 < len(edges) else None
            mask = []
            student_set = set()
            inters = []
            for stu in student_ids:
                cnt = counts.get(int(stu), 0)
                hit = cnt >= left if right is None else left <= cnt < right
                mask.append(hit)
                if hit:
                    student_set.add(int(stu))
                    inters.append(cnt)
            mask = np.array(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "group": f"[{left},{'+' if right is None else right})",
                    "min_interactions": left,
                    "max_interactions": right,
                    "student_count": len(student_set),
                    "sample_count": int(mask.sum()),
                    "mean_interactions": float(np.mean(inters)),
                    "auc": self._safe_auc(labels[mask], preds[mask]),
                    "acc": float(accuracy_score(labels[mask], np.round(preds[mask]))),
                }
            )
        return rows

    def evaluate(self, loader, split_name):
        self.model.eval()
        preds, labels, student_ids = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch)
                pred = self.model.forward_test(
                    batch["stu_id"], batch["exer_id"], batch["Q_mat"]
                ).flatten()
                preds.extend(pred.cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())
                student_ids.extend(batch["stu_id"].cpu().numpy())

        preds = np.array(preds)
        labels = np.array(labels)
        auc = self._safe_auc(labels, preds)
        acc = float(accuracy_score(labels, np.round(preds)))
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        self.logger.info(f"[{split_name}] AUC: {auc:.4f} | ACC: {acc:.4f} | RMSE: {rmse:.4f}")

        pred_path = os.path.join(self.args.log_dir, f"{split_name.lower()}_predictions.csv")
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            pred_writer = csv.writer(f)
            pred_writer.writerow(["prob", "label"])
            pred_writer.writerows(zip(preds.tolist(), labels.tolist()))

        longtail_rows = self._longtail_rows(student_ids, labels, preds)
        self._log_longtail_metrics(longtail_rows, split_name)
        if longtail_rows:
            path = os.path.join(self.args.log_dir, f"{split_name.lower()}_longtail_students.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(longtail_rows[0].keys()))
                writer.writeheader()
                writer.writerows(longtail_rows)

        return {"auc": auc, "acc": acc, "rmse": rmse}

    def save_summary(self, final_results):
        summary_path = os.path.join(self.args.log_dir, "metrics_summary.csv")
        fieldnames = list(final_results[0].keys()) if final_results else [
            "experiment_name",
            "split",
            "auc",
            "acc",
            "rmse",
        ]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_results)
        self.logger.info(f"Saved final metrics summary to {summary_path}")
