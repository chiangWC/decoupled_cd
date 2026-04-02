from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize per-student TKC/UKC counts from interaction CSVs.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing split CSVs such as train.csv, valid.csv, test.csv.",
    )
    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated split names to include.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ukc_visualization",
        help="Directory to write figures and CSV summaries.",
    )
    parser.add_argument(
        "--tag",
        default="dataset",
        help="Short tag used in output filenames and figure titles.",
    )
    return parser.parse_args()


def parse_cpt_seq(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    return [token.strip() for token in str(raw_value).split(",") if token.strip()]


def load_frames(data_dir: Path, splits: Iterable[str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for split in splits:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")
        frames.append(pd.read_csv(csv_path))
    return frames


def build_student_stats(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    merged = pd.concat(frames, ignore_index=True)

    all_concepts: set[str] = set()
    student_exercises: dict[str, set[str]] = defaultdict(set)
    student_concepts: dict[str, set[str]] = defaultdict(set)

    for row in merged.itertuples(index=False):
        student_id = str(row.stu_id)
        exercise_id = str(row.exer_id)
        concepts = parse_cpt_seq(row.cpt_seq)

        student_exercises[student_id].add(exercise_id)
        student_concepts[student_id].update(concepts)
        all_concepts.update(concepts)

    total_concepts = len(all_concepts)
    rows: list[dict[str, int | str]] = []
    for student_id in sorted(student_exercises.keys(), key=lambda x: int(x) if x.isdigit() else x):
        e_u = student_exercises[student_id]
        tkc_u = student_concepts[student_id]
        ukc_u = all_concepts - tkc_u
        rows.append(
            {
                "stu_id": student_id,
                "e_count": len(e_u),
                "tkc_count": len(tkc_u),
                "ukc_count": len(ukc_u),
            }
        )

    return pd.DataFrame(rows), total_concepts


def summarize(student_stats: pd.DataFrame, total_concepts: int, tag: str, splits: list[str]) -> dict[str, object]:
    ukc = student_stats["ukc_count"]
    tkc = student_stats["tkc_count"]
    e_count = student_stats["e_count"]
    return {
        "tag": tag,
        "splits": splits,
        "num_students": int(len(student_stats)),
        "num_total_concepts_K": int(total_concepts),
        "students_with_nonempty_UKC": int((ukc > 0).sum()),
        "students_with_empty_UKC": int((ukc == 0).sum()),
        "ukc_mean": float(ukc.mean()),
        "ukc_median": float(ukc.median()),
        "ukc_min": int(ukc.min()),
        "ukc_max": int(ukc.max()),
        "ukc_p25": float(ukc.quantile(0.25)),
        "ukc_p75": float(ukc.quantile(0.75)),
        "tkc_mean": float(tkc.mean()),
        "tkc_median": float(tkc.median()),
        "e_mean": float(e_count.mean()),
        "e_median": float(e_count.median()),
    }


def plot_histogram(series: pd.Series, title: str, xlabel: str, output_path: Path, color: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=30, color=color, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_scatter(student_stats: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(
        student_stats["e_count"],
        student_stats["ukc_count"],
        s=12,
        alpha=0.45,
        c="#1d4ed8",
    )
    plt.title(title)
    plt.xlabel("|E_u|")
    plt.ylabel("|UKC_u|")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [token.strip() for token in args.splits.split(",") if token.strip()]
    frames = load_frames(data_dir, splits)
    student_stats, total_concepts = build_student_stats(frames)
    summary = summarize(student_stats, total_concepts, args.tag, splits)

    student_csv = output_dir / f"{args.tag}_student_ukc_stats.csv"
    summary_json = output_dir / f"{args.tag}_ukc_summary.json"
    ukc_hist = output_dir / f"{args.tag}_ukc_hist.png"
    tkc_hist = output_dir / f"{args.tag}_tkc_hist.png"
    scatter = output_dir / f"{args.tag}_ukc_vs_e.png"

    student_stats.to_csv(student_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_histogram(
        student_stats["ukc_count"],
        title=f"{args.tag}: Distribution of |UKC_u|",
        xlabel="|UKC_u|",
        output_path=ukc_hist,
        color="#dc2626",
    )
    plot_histogram(
        student_stats["tkc_count"],
        title=f"{args.tag}: Distribution of |TKC_u|",
        xlabel="|TKC_u|",
        output_path=tkc_hist,
        color="#059669",
    )
    plot_scatter(
        student_stats,
        title=f"{args.tag}: |E_u| vs |UKC_u|",
        output_path=scatter,
    )

    print(f"Wrote student stats: {student_csv}")
    print(f"Wrote summary: {summary_json}")
    print(f"Wrote figures: {ukc_hist}, {tkc_hist}, {scatter}")


if __name__ == "__main__":
    main()
