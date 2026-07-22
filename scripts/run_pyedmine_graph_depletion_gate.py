from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pyedmine_cd_baselines import (
    build_maps,
    build_q_table,
    write_cd_file,
    write_id_map,
    write_kt_file,
)
from scripts.run_kancd_concept_depletion_gate import (
    align_and_write,
    verify_protocol_arm,
)
from utils import write_json
from data.pool_protocol import sha256_file


MODELS = ("RCD", "HyperCD")
SETTING_NAME = "gate_d_depletion"
RECIPE = {
    "RCD": {
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.0,
    },
    "HyperCD": {
        "batch_size": 256,
        "learning_rate": 1.0e-4,
        "weight_decay": 5.0e-4,
        "num_layer": 3,
        "dim_feature": 512,
        "dim_emb": 16,
        "leaky": 0.8,
    },
}
MODEL_SCRIPT = {"RCD": "rcd.py", "HyperCD": "hyper_cd.py"}
SOURCE_FILES = {
    "RCD": (
        "examples/cognitive_diagnosis/train/rcd.py",
        "examples/cognitive_diagnosis/train/config/rcd.py",
        "examples/cognitive_diagnosis/rcd/process_edge.py",
        "examples/cognitive_diagnosis/rcd/build_k_e_graph.py",
        "examples/cognitive_diagnosis/rcd/build_u_e_graph.py",
        "edmine/model/cognitive_diagnosis_model/RCD.py",
    ),
    "HyperCD": (
        "examples/cognitive_diagnosis/train/hyper_cd.py",
        "examples/cognitive_diagnosis/train/config/hyper_cd.py",
        "examples/cognitive_diagnosis/hyper_cd/construct_hyper_graph.py",
        "edmine/model/cognitive_diagnosis_model/HyperCD.py",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen PyEdmine graph-model Gate D arm."
    )
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("concept_depleted", "random_depleted"),
        required=True,
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pyedmine-root", type=Path, default=Path("/home/xph/jwc/pyedmine")
    )
    parser.add_argument(
        "--external-python", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path | None = None,
) -> str:
    print("$ " + " ".join(command), flush=True)
    if log_path is None:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(process.stdout, end="", flush=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        print(f"log={log_path}", flush=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return "" if log_path is not None else process.stdout


def source_audit(root: Path, model: str) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).splitlines()
    relevant = set(SOURCE_FILES[model])
    dirty_relevant = [
        line
        for line in status
        if line[3:].strip() in relevant
    ]
    if dirty_relevant:
        raise RuntimeError(
            f"Relevant PyEdmine source files are dirty: {dirty_relevant}"
        )
    return {
        "root": str(root.resolve()),
        "git_commit": commit,
        "working_tree_status": status,
        "relevant_files": {
            name: sha256_file(root / name) for name in SOURCE_FILES[model]
        },
        "relevant_files_clean": True,
    }


def prepare_runtime(
    *,
    arm_dir: Path,
    runtime: Path,
    dataset: str,
    arm: str,
) -> dict[str, Any]:
    frames = {
        split: pd.read_csv(arm_dir / f"{split}.csv")
        for split in ("train", "valid", "test")
    }
    q_matrix = pd.read_csv(arm_dir / "Q_matrix.csv")
    union = pd.concat(frames.values(), ignore_index=True)
    user_map, question_map, concept_map = build_maps([union], q_matrix)
    slug = "".join(character.lower() for character in dataset if character.isalnum())
    dataset_name = f"gate_d_{slug}_{arm}"
    preprocessed = (
        runtime
        / "meta_data"
        / "dataset"
        / "dataset_preprocessed"
        / dataset_name
    )
    setting = (
        runtime / "meta_data" / "dataset" / "settings" / SETTING_NAME
    )
    preprocessed.mkdir(parents=True)
    setting.mkdir(parents=True)
    (runtime / "saved_models").mkdir(parents=True)
    write_json(
        {"name": SETTING_NAME, "source": "frozen Gate D runtime"},
        setting / "setting.json",
    )
    file_names = {
        split: f"{dataset_name}_{split}.txt"
        for split in ("train", "valid", "test")
    }
    for split, frame in frames.items():
        write_cd_file(
            frame,
            setting / file_names[split],
            user_map,
            question_map,
        )
    write_kt_file(
        frames["train"],
        preprocessed / "data.txt",
        user_map,
        question_map,
    )
    np.save(
        preprocessed / "Q_table.npy",
        build_q_table(q_matrix, question_map, concept_map),
    )
    write_json(
        {
            "num_user": len(user_map),
            "num_question": len(question_map),
            "num_concept": len(concept_map),
        },
        preprocessed / "statics_preprocessed.json",
    )
    write_id_map(preprocessed / "user_id_map.csv", "raw_user_id", user_map)
    write_id_map(
        preprocessed / "question_id_map.csv",
        "raw_question_id",
        question_map,
    )
    write_id_map(
        preprocessed / "concept_id_map.csv",
        "raw_concept_id",
        concept_map,
    )
    (setting / f"{dataset_name}_statics.txt").write_text(
        f"num of user: {len(user_map)}\n", encoding="utf-8"
    )
    return {
        "dataset_name": dataset_name,
        "file_names": file_names,
        "dimensions": {
            "students": len(user_map),
            "exercises": len(question_map),
            "concepts": len(concept_map),
        },
        "rows": {split: len(frame) for split, frame in frames.items()},
        "train_only_graph_inputs": True,
    }


def preprocess(
    *,
    model: str,
    pyedmine_root: Path,
    python: Path,
    runtime: Path,
    prepared: dict[str, Any],
    env: dict[str, str],
) -> None:
    dataset_name = prepared["dataset_name"]
    train_name = prepared["file_names"]["train"]
    if model == "RCD":
        scripts = [
            (
                "examples/cognitive_diagnosis/rcd/process_edge.py",
                ["--setting_name", SETTING_NAME, "--dataset_name", dataset_name],
            ),
            (
                "examples/cognitive_diagnosis/rcd/build_k_e_graph.py",
                [
                    "--setting_name",
                    SETTING_NAME,
                    "--dataset_name",
                    dataset_name,
                    "--train_file_name",
                    train_name,
                ],
            ),
            (
                "examples/cognitive_diagnosis/rcd/build_u_e_graph.py",
                [
                    "--setting_name",
                    SETTING_NAME,
                    "--dataset_name",
                    dataset_name,
                    "--train_file_name",
                    train_name,
                ],
            ),
        ]
    else:
        scripts = [
            (
                "examples/cognitive_diagnosis/hyper_cd/construct_hyper_graph.py",
                [
                    "--setting_name",
                    SETTING_NAME,
                    "--dataset_name",
                    dataset_name,
                    "--train_file_name",
                    train_name,
                ],
            )
        ]
    for relative, extra in scripts:
        _run(
            [str(python), str(pyedmine_root / relative), *extra],
            cwd=runtime,
            env=env,
        )


def find_model_dir(runtime: Path, prefix: str) -> Path:
    candidates = [
        path
        for path in (runtime / "saved_models").iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one saved model matching {prefix}, got {candidates}."
        )
    return candidates[0]


def predict_split(
    *,
    args: argparse.Namespace,
    runtime: Path,
    env: dict[str, str],
    prepared: dict[str, Any],
    model_dir: Path,
    arm_dir: Path,
    split: str,
) -> dict[str, Any]:
    output = (args.output_dir / f"{split}_predictions.csv").resolve()
    _run(
        [
            str(args.external_python),
            str(REPO_ROOT / "scripts" / "export_pyedmine_predictions.py"),
            "--pyedmine-root",
            str(args.pyedmine_root),
            "--model-dir-name",
            model_dir.name,
            "--dataset-name",
            prepared["dataset_name"],
            "--file-name",
            prepared["file_names"][split],
            "--source-csv",
            str((arm_dir / f"{split}.csv").resolve()),
            "--output-csv",
            str(output),
            "--batch-size",
            "2048",
            *(["--cpu"] if args.cpu else []),
        ],
        cwd=runtime,
        env=env,
    )
    frame = pd.read_csv(output)
    return align_and_write(
        source=arm_dir / f"{split}.csv",
        probabilities=frame["prob"].to_numpy(float),
        labels=frame["label"].to_numpy(float),
        output=output,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 42:
        raise ValueError("Gate D fixes model seed to 42.")
    if args.cpu and args.gpu is not None:
        raise ValueError("--cpu and --gpu are mutually exclusive.")
    if not args.smoke and (args.epochs != 50 or args.patience != 5):
        raise ValueError("Formal Gate D fixes epochs=50 and patience=5.")
    if args.smoke and args.epochs != 1:
        raise ValueError("Smoke Gate D must use --epochs 1.")
    manifest, arm_dir, arm_hashes = verify_protocol_arm(
        args.protocol_dir, args.arm
    )
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    runtime = args.output_dir / "runtime"
    runtime.mkdir()
    source = source_audit(args.pyedmine_root, args.model)
    prepared = prepare_runtime(
        arm_dir=arm_dir,
        runtime=runtime,
        dataset=manifest["dataset"],
        arm=args.arm,
    )
    shim_dir = runtime / "dependency_shims"
    shim_dir.mkdir()
    (shim_dir / "wandb.py").write_text(
        "def init(*args, **kwargs):\n"
        "    raise RuntimeError('wandb is disabled for Gate D')\n\n"
        "def log(*args, **kwargs):\n"
        "    raise RuntimeError('wandb is disabled for Gate D')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(shim_dir.resolve()), str(args.pyedmine_root.resolve()))
    )
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    preprocess(
        model=args.model,
        pyedmine_root=args.pyedmine_root,
        python=args.external_python,
        runtime=runtime,
        prepared=prepared,
        env=env,
    )
    recipe = RECIPE[args.model]
    command = [
        str(args.external_python),
        str(REPO_ROOT / "scripts" / "train_pyedmine_graph_model.py"),
        "--pyedmine-root",
        str(args.pyedmine_root),
        "--model",
        args.model,
        "--setting_name",
        SETTING_NAME,
        "--dataset_name",
        prepared["dataset_name"],
        "--train_file_name",
        prepared["file_names"]["train"],
        "--valid_file_name",
        prepared["file_names"]["valid"],
        "--max_epoch",
        str(args.epochs),
        "--use_early_stop",
        "True",
        "--num_epoch_early_stop",
        str(args.patience),
        "--main_metric",
        "AUC",
        "--seed",
        str(args.seed),
        "--save_model",
        "True",
        "--use_wandb",
        "False",
        "--train_batch_size",
        str(recipe["batch_size"]),
        "--learning_rate",
        str(recipe["learning_rate"]),
        "--weight_decay",
        str(recipe["weight_decay"]),
    ]
    if args.cpu:
        command.extend(["--use_cpu", "True"])
    if args.model == "HyperCD":
        for name in ("num_layer", "dim_feature", "dim_emb", "leaky"):
            command.extend([f"--{name}", str(recipe[name])])
    _run(
        command,
        cwd=runtime,
        env=env,
        log_path=args.output_dir / "train.log",
    )
    prefix = (
        f"{args.model}@@{SETTING_NAME}@@"
        f"{Path(prepared['file_names']['train']).stem}@@seed_{args.seed}@@"
    )
    model_dir = find_model_dir(runtime, prefix)
    summaries = {
        split: predict_split(
            args=args,
            runtime=runtime,
            env=env,
            prepared=prepared,
            model_dir=model_dir,
            arm_dir=arm_dir,
            split=split,
        )
        for split in ("valid", "test")
    }
    checkpoint = model_dir / "saved.ckt"
    payload = {
        "schema_version": 1,
        "gate": "graph_family_concept_depletion_gate_d",
        "formal": not args.smoke,
        "model": args.model,
        "model_family": "graph_cd",
        "dataset": manifest["dataset"],
        "arm": args.arm,
        "seed": args.seed,
        "selection": "best early-stop validation AUC",
        "settings": {
            **recipe,
            "epochs": args.epochs,
            "patience": args.patience,
            "scheduler": False,
            "external_python": str(args.external_python.resolve()),
        },
        "prepared": prepared,
        "model_source": source,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "protocol": {
            "manifest": str((args.protocol_dir / "manifest.json").resolve()),
            "manifest_sha256": sha256_file(
                args.protocol_dir / "manifest.json"
            ),
            "arm_hashes": arm_hashes,
            "source_test_opened": False,
        },
        "splits": summaries,
    }
    write_json(payload, args.output_dir / "summary.json")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
