from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import torch

from data.pipeline import prepare_experiment_split_bundles
from data.pool_protocol import (
    assign_student_disjoint_support_query,
    canonicalize_interactions,
    derive_q_matrix,
    sha256_file,
    stable_fraction,
)
from scripts.analyze_prediction_slices import add_slice_features
from scripts.evaluate_checkpoint_average import _validate_protocol_summary
from scripts.evaluate_coverage_slice import add_target_coverage
from scripts.split_student_disjoint import (
    _read_source,
    validate_output_location,
)
from scripts.train import resolve_verified_protocol_manifest
from trainers.engine import _validate_history_visibility


def _student_role(student: str) -> str:
    fraction = stable_fraction(2024, "student-disjoint", student)
    if fraction < 0.70:
        return "train"
    if fraction < 0.80:
        return "valid"
    return "test"


class StudentDisjointProtocolTest(unittest.TestCase):
    def _source_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_rows: list[dict[str, object]] = []
        valid_rows: list[dict[str, object]] = []
        test_rows: list[dict[str, object]] = []
        valid_student: str | None = None
        test_student: str | None = None
        for index in range(200):
            student = f"s{index}"
            role = _student_role(student)
            train_exercises = range(20) if role == "train" else range(10)
            for exercise in train_exercises:
                train_rows.append(
                    {
                        "stu_id": student,
                        "exer_id": exercise,
                        "cpt_seq": exercise,
                        "label": (index + exercise) % 2,
                    }
                )
            train_rows.append(
                {
                    "stu_id": student,
                    "exer_id": 0,
                    "cpt_seq": "shared-extra",
                    "label": index % 2,
                }
            )
            if role == "valid":
                valid_student = valid_student or student
                valid_exercises = range(10, 13)
                test_exercises = range(16, 19)
            elif role == "test":
                test_student = test_student or student
                valid_exercises = range(16, 19)
                test_exercises = range(13, 16)
            else:
                valid_exercises = range(10, 13)
                test_exercises = range(13, 16)
            for exercise in valid_exercises:
                valid_rows.append(
                    {
                        "stu_id": student,
                        "exer_id": exercise,
                        "cpt_seq": exercise,
                        "label": (index + exercise) % 2,
                    }
                )
            for exercise in test_exercises:
                test_rows.append(
                    {
                        "stu_id": student,
                        "exer_id": exercise,
                        "cpt_seq": exercise,
                        "label": (index + exercise) % 2,
                    }
                )

        assert valid_student is not None and test_student is not None
        train_rows.append(
            {
                "stu_id": valid_student,
                "exer_id": 998,
                "cpt_seq": 3,
                "label": 1,
            }
        )
        valid_rows.extend(
            [
                {
                    "stu_id": valid_student,
                    "exer_id": 999,
                    "cpt_seq": 4,
                    "label": 0,
                },
                {
                    "stu_id": valid_student,
                    "exer_id": 0,
                    "cpt_seq": 0,
                    "label": 1,
                },
            ]
        )
        test_rows.append(
            {
                "stu_id": test_student,
                "exer_id": 999,
                "cpt_seq": 4,
                "label": 0,
            }
        )
        return (
            canonicalize_interactions(pd.DataFrame(train_rows))[0],
            canonicalize_interactions(pd.DataFrame(valid_rows))[0],
            canonicalize_interactions(pd.DataFrame(test_rows))[0],
        )

    def test_students_and_support_query_groups_are_disjoint(self) -> None:
        train, valid, test = self._source_frames()
        splits, audit = assign_student_disjoint_support_query(
            train_frame=train,
            valid_frame=valid,
            test_frame=test,
            q_matrix=derive_q_matrix(
                pd.concat([train, valid, test], ignore_index=True)
            )[0],
            seed=2024,
            min_support_groups=10,
            min_query_groups=3,
            min_target_label_count=1,
        )
        student_sets = {
            "train": set(splits["train"]["stu_id"]),
            "valid": set(splits["valid_query"]["stu_id"]),
            "test": set(splits["test_query"]["stu_id"]),
        }
        self.assertFalse(student_sets["train"] & student_sets["valid"])
        self.assertFalse(student_sets["train"] & student_sets["test"])
        self.assertFalse(student_sets["valid"] & student_sets["test"])
        self.assertEqual(
            audit["student_overlap"],
            {"train_valid": 0, "train_test": 0, "valid_test": 0},
        )
        self.assertEqual(
            audit["support_query_group_overlap"],
            {"valid": 0, "test": 0},
        )
        known = set(splits["train"]["exer_id"].astype(str))
        for name in ("valid_support", "valid_query", "test_support", "test_query"):
            self.assertTrue(set(splits[name]["exer_id"].astype(str)) <= known)
        self.assertGreater(
            audit["splits"]["valid"]["support_rows"],
            audit["splits"]["valid"]["support_groups"],
        )
        self.assertGreaterEqual(
            audit["splits"]["valid"]["support_group_length"]["min"],
            10,
        )
        self.assertGreater(
            audit["splits"]["valid"]["unknown_support_rows_removed"],
            0,
        )
        self.assertGreater(
            audit["splits"]["valid"]["unknown_exercise_rows_removed"],
            0,
        )
        self.assertGreater(
            audit["splits"]["valid"]["support_query_overlap_rows_removed"],
            0,
        )
        for role in ("valid", "test"):
            role_audit = audit["splits"][role]
            self.assertGreater(role_audit["label_counts"]["0"], 0)
            self.assertGreater(role_audit["label_counts"]["1"], 0)
            self.assertEqual(
                sum(
                    bucket["rows"]
                    for bucket in role_audit["coverage_buckets"].values()
                ),
                role_audit["query_rows"],
            )

    def test_output_path_rejects_equal_and_nested_directories(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            sibling = root / "output"
            self.assertEqual(
                validate_output_location(source, sibling),
                (source.resolve(), sibling.resolve()),
            )
            for unsafe in (source, source / "child", root):
                with self.assertRaises(ValueError):
                    validate_output_location(source, unsafe)

            pd.DataFrame(
                [{"stu_id": "s", "exer_id": 0, "cpt_seq": 0, "label": 1}]
            ).to_csv(source / "train.csv", index=False)
            pd.DataFrame(
                [{"stu_id": "s", "exer_id": 0, "cpt_seq": 0, "label": 0}]
            ).to_csv(source / "valid.csv", index=False)
            pd.DataFrame(
                [{"stu_id": "t", "exer_id": 1, "cpt_seq": 1, "label": 0}]
            ).to_csv(source / "test.csv", index=False)
            with self.assertRaises(RuntimeError):
                _read_source(source)

    def test_assignment_is_reproducible(self) -> None:
        train, valid, test = self._source_frames()
        first, first_audit = assign_student_disjoint_support_query(
            train_frame=train,
            valid_frame=valid,
            test_frame=test,
            q_matrix=derive_q_matrix(
                pd.concat([train, valid, test], ignore_index=True)
            )[0],
            min_target_label_count=1,
        )
        second, second_audit = assign_student_disjoint_support_query(
            train_frame=train,
            valid_frame=valid,
            test_frame=test,
            q_matrix=derive_q_matrix(
                pd.concat([train, valid, test], ignore_index=True)
            )[0],
            min_target_label_count=1,
        )
        for name in first:
            self.assertEqual(
                first[name]["source_row_id"].tolist(),
                second[name]["source_row_id"].tolist(),
            )
        self.assertEqual(first_audit, second_audit)



class ProtocolManifestTest(unittest.TestCase):
    FILENAMES = (
        "train.csv",
        "valid_support.csv",
        "valid_query.csv",
        "test_support.csv",
        "test_query.csv",
        "Q_matrix.csv",
    )

    def _write_manifest(self, root: Path) -> Path:
        for index, filename in enumerate(self.FILENAMES):
            (root / filename).write_text(
                f"column\n{index}\n",
                encoding="utf-8",
            )
        manifest = {
            "schema_version": 1,
            "protocol": "student_disjoint_support_query",
            "directory": str(root.resolve()),
            "files": {
                filename: {"sha256": sha256_file(root / filename), "rows": 1}
                for filename in self.FILENAMES
            },
            "audit": {"seed": 2024},
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    @staticmethod
    def _args(
        manifest_path: Path,
        *,
        confirmation: bool = False,
        max_rows: int | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            protocol_manifest=str(manifest_path),
            protocol_confirmation=confirmation,
            max_rows=max_rows,
            seed=7,
            evaluation_stage="confirmation",
        )

    def test_manifest_resolves_hashes_and_enforces_run_policy(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._write_manifest(root)

            args = self._args(manifest_path)
            metadata = resolve_verified_protocol_manifest(args)
            assert metadata is not None
            self.assertEqual(args.seed, 42)
            self.assertEqual(args.evaluation_stage, "validation")
            self.assertEqual(metadata["protocol_split_seed"], 2024)
            self.assertEqual(
                metadata["protocol_manifest_sha256"],
                sha256_file(manifest_path),
            )
            argument_names = {
                "train.csv": "train_interactions",
                "valid_support.csv": "valid_history_interactions",
                "valid_query.csv": "valid_interactions",
                "test_support.csv": "test_history_interactions",
                "test_query.csv": "test_interactions",
                "Q_matrix.csv": "q_matrix",
            }
            for filename, argument_name in argument_names.items():
                self.assertEqual(
                    getattr(args, argument_name),
                    str((root / filename).resolve()),
                )
                self.assertEqual(
                    metadata["protocol_files"][argument_name]["sha256"],
                    sha256_file(root / filename),
                )

            summary = {
                **metadata,
                **{
                    argument_name: getattr(args, argument_name)
                    for argument_name in argument_names.values()
                },
            }
            _validate_protocol_summary(summary)

            confirmation_args = self._args(
                manifest_path,
                confirmation=True,
            )
            resolve_verified_protocol_manifest(confirmation_args)
            self.assertEqual(confirmation_args.seed, 42)
            self.assertEqual(
                confirmation_args.evaluation_stage,
                "confirmation",
            )

            with self.assertRaisesRegex(ValueError, "--max-rows"):
                resolve_verified_protocol_manifest(
                    self._args(manifest_path, max_rows=10)
                )

            (root / "valid_support.csv").write_text(
                "column\ntampered\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_verified_protocol_manifest(
                    self._args(manifest_path)
                )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _validate_protocol_summary(summary)



class SeparateHistoryPipelineTest(unittest.TestCase):
    def _write_frame(
        self,
        root: Path,
        name: str,
        rows: list[dict[str, object]],
    ) -> str:
        path = root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_eval_student_evidence_uses_support_and_global_evidence_is_train_only(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = self._write_frame(
                root,
                "train.csv",
                [
                    {"stu_id": "train", "exer_id": 0, "cpt_seq": 0, "label": 0},
                    {"stu_id": "train", "exer_id": 1, "cpt_seq": 1, "label": 1},
                ],
            )
            valid_path = self._write_frame(
                root,
                "valid.csv",
                [
                    {"stu_id": "valid", "exer_id": 1, "cpt_seq": 1, "label": 0},
                ],
            )
            test_path = self._write_frame(
                root,
                "test.csv",
                [
                    {"stu_id": "test", "exer_id": 1, "cpt_seq": 1, "label": 1},
                ],
            )
            valid_support_path = self._write_frame(
                root,
                "valid_support.csv",
                [
                    {"stu_id": "valid", "exer_id": 0, "cpt_seq": 0, "label": 1},
                ],
            )
            test_support_path = self._write_frame(
                root,
                "test_support.csv",
                [
                    {"stu_id": "test", "exer_id": 0, "cpt_seq": 0, "label": 0},
                ],
            )
            q_path = self._write_frame(
                root,
                "Q_matrix.csv",
                [
                    {"exer_id": 0, "cpt_seq": 0},
                    {"exer_id": 1, "cpt_seq": 1},
                ],
            )

            bundles = prepare_experiment_split_bundles(
                train_interactions_path=train_path,
                valid_interactions_path=valid_path,
                test_interactions_path=test_path,
                valid_history_interactions_path=valid_support_path,
                test_history_interactions_path=test_support_path,
                q_matrix_path=q_path,
            )
            valid_bundle = bundles["valid"]
            student_map = valid_bundle.student_id_map
            exercise_map = valid_bundle.exercise_id_map
            valid_id = student_map["valid"]
            train_id = student_map["train"]
            ex0 = exercise_map["0"]

            self.assertEqual(
                valid_bundle.history_interactions["stu_id"].astype(str).tolist(),
                ["valid"],
            )
            self.assertEqual(
                float(valid_bundle.student_exercise_mask[valid_id, ex0]),
                1.0,
            )
            self.assertEqual(
                float(valid_bundle.student_exercise_mask[train_id, ex0]),
                0.0,
            )
            self.assertEqual(
                float(valid_bundle.response_matrix_tensor[valid_id, ex0]),
                1.0,
            )
            self.assertTrue(
                torch.equal(
                    bundles["train"].exercise_evidence_tensor,
                    valid_bundle.exercise_evidence_tensor,
                )
            )
            self.assertEqual(
                float(valid_bundle.exercise_evidence_tensor[ex0, 0]),
                1.0,
            )
            self.assertEqual(
                float(valid_bundle.exercise_evidence_tensor[ex0, 3]),
                0.0,
            )
            with self.assertRaises(ValueError):
                prepare_experiment_split_bundles(
                    train_interactions_path=train_path,
                    valid_interactions_path=valid_path,
                    test_interactions_path=test_path,
                    valid_history_interactions_path=valid_support_path,
                    q_matrix_path=q_path,
                )
            with self.assertRaises(ValueError):
                prepare_experiment_split_bundles(
                    train_interactions_path=train_path,
                    valid_interactions_path=valid_path,
                    test_interactions_path=test_path,
                    valid_history_interactions_path=test_support_path,
                    test_history_interactions_path=valid_support_path,
                    q_matrix_path=q_path,
                )

    def test_default_history_behavior_is_unchanged(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            common = [
                {"stu_id": "s", "exer_id": 0, "cpt_seq": 0, "label": 1}
            ]
            train_path = self._write_frame(root, "train.csv", common)
            valid_path = self._write_frame(
                root,
                "valid.csv",
                [{"stu_id": "s", "exer_id": 1, "cpt_seq": 1, "label": 0}],
            )
            test_path = self._write_frame(
                root,
                "test.csv",
                [{"stu_id": "s", "exer_id": 1, "cpt_seq": 1, "label": 1}],
            )
            q_path = self._write_frame(
                root,
                "Q_matrix.csv",
                [
                    {"exer_id": 0, "cpt_seq": 0},
                    {"exer_id": 1, "cpt_seq": 1},
                ],
            )
            bundles = prepare_experiment_split_bundles(
                train_interactions_path=train_path,
                valid_interactions_path=valid_path,
                test_interactions_path=test_path,
                q_matrix_path=q_path,
            )
            pd.testing.assert_frame_equal(
                bundles["valid"].history_interactions.reset_index(drop=True),
                bundles["train"].interactions.reset_index(drop=True),
            )
            self.assertTrue(
                torch.equal(
                    bundles["valid"].student_exercise_mask,
                    bundles["train"].student_exercise_mask,
                )
            )


    def test_engine_rejects_same_student_exercise_with_different_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = self._write_frame(
                root,
                "train.csv",
                [{"stu_id": "s", "exer_id": 0, "cpt_seq": 0, "label": 1}],
            )
            valid_path = self._write_frame(
                root,
                "valid.csv",
                [{"stu_id": "s", "exer_id": 0, "cpt_seq": 0, "label": 0}],
            )
            test_path = self._write_frame(
                root,
                "test.csv",
                [{"stu_id": "s", "exer_id": 1, "cpt_seq": 1, "label": 0}],
            )
            q_path = self._write_frame(
                root,
                "Q_matrix.csv",
                [
                    {"exer_id": 0, "cpt_seq": 0},
                    {"exer_id": 1, "cpt_seq": 1},
                ],
            )
            bundles = prepare_experiment_split_bundles(
                train_interactions_path=train_path,
                valid_interactions_path=valid_path,
                test_interactions_path=test_path,
                q_matrix_path=q_path,
            )
            with self.assertRaises(ValueError):
                _validate_history_visibility(bundles["valid"])


    def test_long_form_q_unifies_history_evidence_and_coverage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = self._write_frame(
                root,
                "train.csv",
                [
                    {"stu_id": "old", "exer_id": 0, "cpt_seq": "a", "label": 1},
                    {"stu_id": "old", "exer_id": 1, "cpt_seq": "c", "label": 0},
                    {"stu_id": "old", "exer_id": 2, "cpt_seq": "b", "label": 1},
                    {"stu_id": "old", "exer_id": 3, "cpt_seq": "a", "label": 0},
                ],
            )
            valid_path = self._write_frame(
                root,
                "valid.csv",
                [
                    {"stu_id": "new", "exer_id": 1, "cpt_seq": "c", "label": 0},
                    {"stu_id": "new", "exer_id": 2, "cpt_seq": "b", "label": 1},
                    {"stu_id": "new", "exer_id": 3, "cpt_seq": "a", "label": 0},
                ],
            )
            test_path = self._write_frame(
                root,
                "test.csv",
                [{"stu_id": "test", "exer_id": 1, "cpt_seq": "c", "label": 1}],
            )
            support_path = self._write_frame(
                root,
                "valid_support.csv",
                [{"stu_id": "new", "exer_id": 0, "cpt_seq": "a", "label": 1}],
            )
            test_support_path = self._write_frame(
                root,
                "test_support.csv",
                [{"stu_id": "test", "exer_id": 0, "cpt_seq": "a", "label": 0}],
            )
            q_path = self._write_frame(
                root,
                "Q_matrix.csv",
                [
                    {"exer_id": 0, "cpt_seq": "a"},
                    {"exer_id": 0, "cpt_seq": "b"},
                    {"exer_id": 1, "cpt_seq": "c"},
                    {"exer_id": 1, "cpt_seq": "d"},
                    {"exer_id": 2, "cpt_seq": "b"},
                    {"exer_id": 2, "cpt_seq": "c"},
                    {"exer_id": 3, "cpt_seq": "a"},
                    {"exer_id": 3, "cpt_seq": "b"},
                ],
            )
            bundles = prepare_experiment_split_bundles(
                train_interactions_path=train_path,
                valid_interactions_path=valid_path,
                test_interactions_path=test_path,
                valid_history_interactions_path=support_path,
                test_history_interactions_path=test_support_path,
                q_matrix_path=q_path,
            )
            valid_bundle = bundles["valid"]
            student = valid_bundle.student_id_map["new"]
            for concept in ("a", "b"):
                concept_index = valid_bundle.concept_id_map[concept]
                self.assertEqual(
                    float(valid_bundle.student_tkc_mask[student, concept_index]),
                    1.0,
                )
                self.assertEqual(
                    float(
                        valid_bundle.student_concept_evidence_tensor[
                            student,
                            concept_index,
                            0,
                        ]
                    ),
                    1.0,
                )

            covered = add_target_coverage(
                valid_bundle.interactions,
                student_history_frame=valid_bundle.history_interactions,
                q_matrix=valid_bundle.q_matrix,
            )
            coverage_names = covered["coverage_bucket"].tolist()
            self.assertEqual(coverage_names, ["zero", "partial", "full"])
            tensor_names: list[str] = []
            for exercise in valid_bundle.interaction_exercise_ids:
                target = valid_bundle.q_matrix_tensor[exercise].bool()
                seen = valid_bundle.student_tkc_mask[student].bool()
                seen_count = int((target & seen).sum())
                total = int(target.sum())
                tensor_names.append(
                    "zero"
                    if seen_count == 0
                    else "full"
                    if seen_count == total
                    else "partial"
                )
            self.assertEqual(tensor_names, coverage_names)

    def test_slice_features_use_support_for_students_and_train_for_items(
        self,
    ) -> None:
        query = pd.DataFrame(
            [
                {
                    "stu_id": "new",
                    "exer_id": 2,
                    "cpt_seq": "0",
                    "label": 1,
                    "prob": 0.8,
                    "confidence": 0.8,
                }
            ]
        )
        support = pd.DataFrame(
            [
                {"stu_id": "new", "exer_id": 0, "cpt_seq": "0", "label": 1},
            ]
        )
        global_train = pd.DataFrame(
            [
                {"stu_id": "old", "exer_id": 2, "cpt_seq": "0", "label": 0},
                {"stu_id": "old", "exer_id": 2, "cpt_seq": "0", "label": 1},
            ]
        )
        q_matrix = pd.DataFrame(
            [
                {"exer_id": 0, "cpt_seq": "0"},
                {"exer_id": 2, "cpt_seq": "0"},
            ]
        )
        covered = add_target_coverage(
            query,
            student_history_frame=support,
            q_matrix=q_matrix,
        )
        self.assertEqual(covered.loc[0, "coverage_bucket"], "full")
        enriched = add_slice_features(
            query,
            student_history_frame=support,
            global_train_frame=global_train,
            q_matrix=q_matrix,
        )
        self.assertEqual(int(enriched.loc[0, "student_history_count"]), 1)
        self.assertEqual(int(enriched.loc[0, "exercise_train_count"]), 2)


if __name__ == "__main__":
    unittest.main()
