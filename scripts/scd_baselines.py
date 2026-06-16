from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCD_ROOT = Path("/home/xph/jwc/SCD")
DEFAULT_WORK_DIR = Path("/home/xph/jwc/research/local_data/scd_baselines")
DEFAULT_PYTHON = Path("/home/xph/anaconda3/envs/newSCD/bin/python")
DEFAULT_EPOCHS = 5

CODE_FILES = [
    "__init__.py",
    "build_graph.py",
    "build_stu.py",
    "data_loader.py",
    "gnn.py",
    "main.py",
    "model.py",
    "utils.py",
]

DATASET_SPECS = {
    "assist09_standard": "/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered",
    "assist09_holdout": "/tmp/assist09_holdout_seed2024",
    "assist17_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_source",
    "assist17_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_holdout_seed2024",
    "nips34_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_source",
    "nips34_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_holdout_seed2024",
    "junyi_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_source",
    "junyi_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_holdout_seed2024",
    "junyi_sample_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_sample_source",
    "junyi_sample_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_sample_holdout_seed2024",
}


@dataclass(frozen=True)
class PreparedDataset:
    key: str
    source_dir: str
    run_dir: str
    rows: dict[str, int]
    num_users: int
    num_exercises: int
    num_concepts: int


@dataclass
class JobState:
    key: str
    pid: int
    gpu: int
    run_dir: str
    log_path: str
    started_at: float
    status: str = "running"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and run SCD on paper CD splits.")
    parser.add_argument("--scd-root", type=Path, default=DEFAULT_SCD_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    add_dataset_selection(prepare)
    prepare.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run")
    add_dataset_selection(run)
    run.add_argument("--gpus", default="0,1")
    run.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    run.add_argument("--max-parallel", type=int, default=2)
    run.add_argument("--poll-seconds", type=float, default=20.0)
    run.add_argument("--force", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    return parser.parse_args()


def add_dataset_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset keys or 'all'.")


def selected_dataset_keys(value: str) -> list[str]:
    if value == "all":
        return list(DATASET_SPECS)
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {', '.join(unknown)}")
    return keys


def split_concepts(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_frame(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def build_maps(frames: dict[str, pd.DataFrame], q_matrix: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    interactions = pd.concat(frames.values(), axis=0, ignore_index=True)
    users = sorted(str(value) for value in interactions["stu_id"].unique())
    exercises = {str(value) for value in interactions["exer_id"].unique()}
    exercises.update(str(value) for value in q_matrix["exer_id"].unique())
    concepts: set[str] = set()
    for value in q_matrix["cpt_seq"].tolist():
        concepts.update(split_concepts(value))
    for value in interactions["cpt_seq"].tolist():
        concepts.update(split_concepts(value))
    return (
        {raw_id: index for index, raw_id in enumerate(users)},
        {raw_id: index for index, raw_id in enumerate(sorted(exercises))},
        {raw_id: index for index, raw_id in enumerate(sorted(concepts))},
    )


def copy_code(args: argparse.Namespace, run_dir: Path) -> None:
    for relative in CODE_FILES:
        shutil.copy2(args.scd_root / relative, run_dir / relative)
    patch_copied_code(run_dir)


def patch_copied_code(run_dir: Path) -> None:
    main_path = run_dir / "main.py"
    main_text = main_path.read_text(encoding="utf-8")
    main_text = main_text.replace("TrainDataLoader(local_map)", "TrainDataLoader(local_map, args)")
    main_text = main_text.replace("ValTestDataLoader(g)", "ValTestDataLoader(g, args)")
    main_text = main_text.replace("#train(args, construct_local_map(args))\n    test(args, construct_local_map(args),3)", "train(args, construct_local_map(args))\n    #test(args, construct_local_map(args),3)")
    main_path.write_text(main_text, encoding="utf-8")

    loader_path = run_dir / "data_loader.py"
    loader_text = loader_path.read_text(encoding="utf-8")
    loader_text = loader_text.replace("def __init__(self, g):", "def __init__(self, g, args):")
    loader_text = loader_text.replace("('cuda:%d' % (1))", "('cuda:%d' % (args.gpu))")
    loader_text = loader_text.replace("data_file = 'data/ASSIST/train_set.json'", "data_file = 'data/assist2009/train_set.json'")
    loader_path.write_text(loader_text, encoding="utf-8")


def to_scd_rows(frame: pd.DataFrame, user_map: dict[str, int], exercise_map: dict[str, int], concept_map: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "user_id": user_map[str(row.stu_id)] + 1,
                "exer_id": exercise_map[str(row.exer_id)] + 1,
                "score": float(row.label),
                "knowledge_code": [concept_map[item] + 1 for item in split_concepts(row.cpt_seq)],
            }
        )
    return rows


def to_grouped_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["user_id"]), []).append(
            {
                "exer_id": int(row["exer_id"]),
                "score": float(row["score"]),
                "knowledge_code": list(row["knowledge_code"]),
            }
        )
    return [
        {"user_id": user_id, "log_num": len(logs), "logs": logs}
        for user_id, logs in sorted(grouped.items())
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def prepare_dataset(args: argparse.Namespace, key: str) -> PreparedDataset:
    source_dir = Path(DATASET_SPECS[key])
    frames = {
        split: load_frame(source_dir / f"{split}.csv", ("stu_id", "exer_id", "cpt_seq", "label"))
        for split in ("train", "valid", "test")
    }
    q_matrix = load_frame(source_dir / "Q_matrix.csv", ("exer_id", "cpt_seq"))
    user_map, exercise_map, concept_map = build_maps(frames, q_matrix)

    run_dir = args.work_dir / "runs" / key
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_code(args, run_dir)
    for subdir in ("data/ASSIST/graph", "data/assist2009", "model", "result", "view"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    train_rows = to_scd_rows(frames["train"], user_map, exercise_map, concept_map)
    test_rows = to_scd_rows(frames["test"], user_map, exercise_map, concept_map)
    write_json(run_dir / "data" / "assist2009" / "train_set.json", train_rows)
    write_json(run_dir / "data" / "ASSIST" / "train_set.json", train_rows)
    write_json(run_dir / "data" / "ASSIST" / "test_set.json", to_grouped_eval(test_rows))

    graph_dir = run_dir / "data" / "ASSIST" / "graph"
    e_from_u = []
    u_from_e = []
    for row in frames["train"].itertuples(index=False):
        user_id = user_map[str(row.stu_id)]
        exercise_id = exercise_map[str(row.exer_id)]
        e_from_u.append(f"{user_id}\t{exercise_id}\n")
        u_from_e.append(f"{exercise_id}\t{user_id}\n")
    (graph_dir / "e_from_u.txt").write_text("".join(e_from_u), encoding="utf-8")
    (graph_dir / "u_from_e.txt").write_text("".join(u_from_e), encoding="utf-8")

    k_from_e = []
    e_from_k = []
    for row in q_matrix.itertuples(index=False):
        exercise_id = exercise_map[str(row.exer_id)]
        for concept in split_concepts(row.cpt_seq):
            concept_id = concept_map[concept]
            k_from_e.append(f"{exercise_id}\t{concept_id}\n")
            e_from_k.append(f"{concept_id}\t{exercise_id}\n")
    (graph_dir / "k_from_e.txt").write_text("".join(k_from_e), encoding="utf-8")
    (graph_dir / "e_from_k.txt").write_text("".join(e_from_k), encoding="utf-8")

    (run_dir / "config.txt").write_text(
        "# Number of Students, Number of Exercises, Number of Knowledge Concepts\n"
        f"{len(user_map)},{len(exercise_map)},{len(concept_map)}",
        encoding="utf-8",
    )
    return PreparedDataset(
        key=key,
        source_dir=str(source_dir),
        run_dir=str(run_dir),
        rows={split: int(len(frame)) for split, frame in frames.items()},
        num_users=len(user_map),
        num_exercises=len(exercise_map),
        num_concepts=len(concept_map),
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["datasets"]


def write_manifest(work_dir: Path, datasets: list[PreparedDataset]) -> None:
    existing = {item["key"]: item for item in load_manifest(work_dir / "manifest.json")}
    for dataset in datasets:
        existing[dataset.key] = asdict(dataset)
    write_json(work_dir / "manifest.json", {"datasets": list(existing.values())})


def prepare(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    prepared = [prepare_dataset(args, key) for key in selected_dataset_keys(args.datasets)]
    write_manifest(args.work_dir, prepared)
    for item in prepared:
        print(f"Prepared {item.key}: users={item.num_users}, exercises={item.num_exercises}, concepts={item.num_concepts}, rows={item.rows}")


def parse_gpus(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> None:
    manifest = {item["key"]: item for item in load_manifest(args.work_dir / "manifest.json")}
    states = load_states(args.work_dir)
    gpus = parse_gpus(args.gpus)
    keys = selected_dataset_keys(args.datasets)
    if not gpus:
        raise ValueError("At least one GPU must be provided.")
    max_parallel = min(args.max_parallel, len(gpus))
    if max_parallel < 1:
        raise ValueError("--max-parallel must be positive.")

    pending = []
    for key in keys:
        if key not in manifest:
            raise ValueError(f"{key} is not prepared yet")
        result_path = Path(manifest[key]["run_dir"]) / "result" / "scd_model_val.txt"
        if result_path.exists() and not args.force:
            print(f"skip existing: {key}")
            continue
        pending.append(key)

    running: dict[subprocess.Popen[str], str] = {}
    while pending or running:
        for process, key in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            states[key]["status"] = "completed" if return_code == 0 else "failed"
            states[key]["return_code"] = return_code
            states[key]["finished_at"] = time.time()
            print(f"{states[key]['status']}: {key} rc={return_code}")
            del running[process]
        write_json(args.work_dir / "status.json", {"jobs": list(states.values())})

        used_gpus = {int(states[key]["gpu"]) for key in running.values()}
        idle_gpus = [gpu for gpu in gpus if gpu not in used_gpus]
        while pending and idle_gpus and len(running) < max_parallel:
            key = pending.pop(0)
            gpu = idle_gpus.pop(0)
            process = launch_job(args, manifest[key], key, gpu)
            running[process] = key
            states[key] = asdict(
                JobState(
                    key=key,
                    pid=process.pid,
                    gpu=gpu,
                    run_dir=str(manifest[key]["run_dir"]),
                    log_path=str(args.work_dir / "logs" / f"{key}__SCD.log"),
                    started_at=time.time(),
                )
            )
            print(f"started: {key} pid={process.pid} gpu={gpu} log={states[key]['log_path']}")
        write_json(args.work_dir / "status.json", {"jobs": list(states.values())})

        if pending or running:
            time.sleep(args.poll_seconds)


def launch_job(args: argparse.Namespace, dataset: dict[str, Any], key: str, gpu: int) -> subprocess.Popen[str]:
    run_dir = Path(dataset["run_dir"])
    log_path = args.work_dir / "logs" / f"{key}__SCD.log"
    command = [
        str(args.python),
        "main.py",
        "--gpu",
        "0",
        "--epoch_n",
        str(args.epochs),
        "--student_n",
        str(dataset["num_users"]),
        "--exer_n",
        str(dataset["num_exercises"]),
        "--knowledge_n",
        str(dataset["num_concepts"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n")
        log_handle.flush()
        return subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def load_states(work_dir: Path) -> dict[str, dict[str, Any]]:
    path = work_dir / "status.json"
    if not path.exists():
        return {}
    return {item["key"]: item for item in json.loads(path.read_text(encoding="utf-8")).get("jobs", [])}


def status(args: argparse.Namespace) -> None:
    states = list(load_states(args.work_dir).values())
    manifest = {item["key"]: item for item in load_manifest(args.work_dir / "manifest.json")}
    for state in states:
        pid = int(state["pid"])
        alive = subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        result_path = Path(manifest[state["key"]]["run_dir"]) / "result" / "scd_model_val.txt"
        latest = result_path.read_text(encoding="utf-8").strip().splitlines()[-1:] if result_path.exists() else []
        state["status"] = "running" if alive else "finished"
        state["latest_metric"] = latest[0] if latest else None
    if args.json:
        print(json.dumps({"jobs": states}, indent=2, sort_keys=True))
    else:
        for state in states:
            print(f"{state['key']}: {state['status']} pid={state['pid']} gpu={state['gpu']} latest={state.get('latest_metric')}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    elif args.command == "status":
        status(args)


if __name__ == "__main__":
    main()
