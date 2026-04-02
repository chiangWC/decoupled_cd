from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ASSIST09 with explicit user/order_id sorting.")
    parser.add_argument(
        "--raw-csv",
        default="/home/xph/jwc/MRCogD/data/assist-09/meta-data/skill_builder_data_corrected_collapsed.csv",
        help="Path to raw ASSIST09 csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/assist_09_ordered",
        help="Directory to write ordered processed files.",
    )
    parser.add_argument("--min-interactions", type=int, default=15)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def recode_entities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    stu_map = {stu_id: index for index, stu_id in enumerate(df["stu_id"].drop_duplicates().tolist())}
    df["stu_id"] = df["stu_id"].map(stu_map)

    exer_map = {exer_id: index for index, exer_id in enumerate(df["exer_id"].drop_duplicates().tolist())}
    df["exer_id"] = df["exer_id"].map(exer_map)

    concept_ids = list(dict.fromkeys(chain.from_iterable(df["cpt_seq"].tolist())))
    cpt_map = {concept_id: index for index, concept_id in enumerate(concept_ids)}
    df["cpt_seq"] = df["cpt_seq"].apply(lambda seq: [cpt_map[item] for item in seq])
    return df


def split_train_test(origin_df: pd.DataFrame, ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, stu_df in origin_df.groupby("stu_id", sort=False):
        stu_df = stu_df.sample(frac=1, random_state=seed)
        threshold = int(len(stu_df) * ratio)
        test_stu = stu_df.iloc[:threshold]
        train_stu = stu_df.iloc[threshold:]
        if len(train_stu) > 0:
            train_parts.append(train_stu)
        if len(test_stu) > 0:
            test_parts.append(test_stu)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def stringify_concepts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cpt_seq"] = out["cpt_seq"].apply(lambda seq: ",".join(map(str, seq)))
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cols = ["user_id", "problem_id", "skill_id", "correct", "order_id"]
    raw = pd.read_csv(args.raw_csv, encoding="latin1", low_memory=False, usecols=use_cols)

    data = raw.dropna(subset=["skill_id"]).copy()
    data["skill_id"] = data["skill_id"].astype(str)
    data = data.sort_values(["user_id", "order_id"], kind="stable").reset_index(drop=True)

    # Keep the first attempt in true per-user order after explicit sorting.
    data = (
        data.groupby("user_id", sort=False)
        .apply(lambda frame: frame.drop_duplicates(subset="problem_id", keep="first"))
        .reset_index(drop=True)
    )

    origin_data = data.rename(
        columns={
            "user_id": "stu_id",
            "problem_id": "exer_id",
            "skill_id": "cpt_seq",
            "correct": "label",
        }
    )
    origin_data["cpt_seq"] = origin_data["cpt_seq"].apply(lambda value: [int(token) for token in str(value).split("_")])
    origin_data = origin_data.groupby("stu_id", sort=False).filter(lambda frame: len(frame) >= args.min_interactions).copy()
    origin_data = recode_entities(origin_data)

    q_data = origin_data.drop_duplicates("exer_id")[["exer_id", "cpt_seq"]].copy()
    q_data = stringify_concepts(q_data)
    q_data.to_csv(output_dir / "Q_matrix.csv", index=False, encoding="utf8")

    full_data = origin_data[["stu_id", "exer_id", "cpt_seq", "label"]].copy()
    stringify_concepts(full_data).to_csv(output_dir / "data.csv", index=False, encoding="utf8")

    train_df, test_df = split_train_test(full_data, args.test_ratio, args.seed)
    train_df, valid_df = split_train_test(train_df, args.valid_ratio, args.seed)

    stringify_concepts(train_df).to_csv(output_dir / "train.csv", index=False, encoding="utf8")
    stringify_concepts(valid_df).to_csv(output_dir / "valid.csv", index=False, encoding="utf8")
    stringify_concepts(test_df).to_csv(output_dir / "test.csv", index=False, encoding="utf8")

    concept_set = set(chain.from_iterable(full_data["cpt_seq"].tolist()))
    summary = {
        "students": int(full_data["stu_id"].nunique()),
        "exercises": int(full_data["exer_id"].nunique()),
        "concepts": int(len(concept_set)),
        "interactions": int(len(full_data)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "output_dir": str(output_dir.resolve()),
    }
    print(summary)


if __name__ == "__main__":
    main()
