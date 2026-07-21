from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_target_local_pairing_protocol.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_target_local_pairing_protocol", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _role(student: str) -> str:
    fraction = MODULE.stable_fraction(
        MODULE.SPLIT_SEED, MODULE.ROLE_NAMESPACE, student
    )
    return "optimizer" if fraction < MODULE.OPTIMIZER_FRACTION else "validation"


def _interaction(
    student: str,
    item: str,
    row: str,
    *,
    label: int,
    concept: str,
) -> dict[str, object]:
    return {
        "source_row_id": row,
        "stu_id": student,
        "exer_id": item,
        "cpt_seq": concept,
        "label": label,
    }


def _profile(
    student: str,
    items: list[str],
    *,
    fold: int = 0,
    repeated_first: bool = False,
) -> object:
    support = pd.DataFrame(
        [
            _interaction(
                student,
                "support",
                f"{student}-support",
                label=1,
                concept="support_concept",
            )
        ]
    )
    query_rows = []
    for index, item in enumerate(items):
        query_rows.append(
            _interaction(
                student,
                item,
                f"{student}-{item}-0",
                label=index % 2,
                concept="c",
            )
        )
        if repeated_first and index == 0:
            query_rows.append(
                _interaction(
                    student,
                    item,
                    f"{student}-{item}-1",
                    label=1 - index % 2,
                    concept="c",
                )
            )
    return MODULE.StudentProfile(
        student=student,
        support=support,
        query=pd.DataFrame(query_rows),
        theta=0.0,
        raw_accuracy=1.0,
        seen_concepts=frozenset({"support_concept"}),
        fold=fold,
    )


def _statistics(item_count: int, *, equal_ease: bool = True) -> object:
    ease = (
        np.full(item_count, 0.5, dtype=float)
        if equal_ease
        else np.linspace(0.1, 0.9, item_count)
    )
    return MODULE.ReferenceStatistics(
        signatures=np.zeros((item_count, 13), dtype=float),
        item_ease=ease,
        item_count=np.full(item_count, 10.0, dtype=float),
        bin_edges=np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=float),
        collapsed_bin_edges=0,
        reference_students_hash="students",
        reference_rows_hash="rows",
    )


class TestValidationOnlyProtocol(unittest.TestCase):
    def _write_source(self, root: Path) -> tuple[list[str], list[str]]:
        candidates = [f"s{index:03d}" for index in range(160)]
        optimizer = [student for student in candidates if _role(student) == "optimizer"]
        validation = [student for student in candidates if _role(student) == "validation"]
        self.assertGreater(len(optimizer), 20)
        self.assertGreater(len(validation), 5)

        support_items = [f"s_item_{index}" for index in range(15)]
        target_items = [f"t_item_{index}" for index in range(6)]
        train_rows = []
        valid_rows = []
        for student in candidates:
            role = _role(student)
            train_items = (
                support_items + target_items
                if role == "optimizer"
                else support_items
            )
            for index, item in enumerate(train_items):
                train_rows.append(
                    _interaction(
                        student,
                        item,
                        f"train-{student}-{item}",
                        label=index % 2,
                        concept="wrong_interaction_value",
                    )
                )
            for index, item in enumerate(target_items):
                valid_rows.append(
                    _interaction(
                        student,
                        item,
                        f"valid-{student}-{item}",
                        label=(index + 1) % 2,
                        concept="wrong_interaction_value",
                    )
                )

        q_rows = []
        for item in support_items + target_items:
            q_rows.append({"exer_id": item, "cpt_seq": "c0"})
        # The loader must union long-form rows and replace interaction cpt_seq.
        q_rows.append({"exer_id": target_items[0], "cpt_seq": "c1"})
        pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
        pd.DataFrame(valid_rows).to_csv(root / "valid.csv", index=False)
        pd.DataFrame(q_rows).to_csv(root / "Q_matrix.csv", index=False)
        (root / "test.csv").write_text("this,must,never,be,opened\n")
        (root / "manifest.json").write_text('{"test_metadata": "forbidden"}\n')
        return optimizer, validation

    def test_loader_opens_only_train_valid_q_and_uses_q_union(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            _, expected_validation = self._write_source(root)
            real_read_csv = pd.read_csv
            opened: list[str] = []

            def guarded_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
                name = Path(path).name
                opened.append(name)
                if name.startswith("test") or name == "manifest.json":
                    raise AssertionError(f"Forbidden file opened: {name}")
                return real_read_csv(path, *args, **kwargs)

            with mock.patch.object(MODULE.pd, "read_csv", side_effect=guarded_read_csv):
                protocol = MODULE.load_validation_only_protocol("synthetic", root)

            self.assertEqual(set(opened), set(MODULE.ALLOWED_SOURCE_FILES))
            self.assertEqual(protocol.audit["opened_source_files"], list(MODULE.ALLOWED_SOURCE_FILES))
            self.assertFalse(protocol.audit["test_files_opened"])
            self.assertFalse(protocol.audit["legacy_manifest_opened"])
            self.assertEqual(
                protocol.q_lookup["t_item_0"],
                ("c0", "c1"),
            )
            self.assertFalse(
                set(protocol.optimizer_train["stu_id"].astype(str))
                & set(protocol.validation_query["stu_id"].astype(str))
            )
            self.assertFalse(
                MODULE._group_set(protocol.validation_support)
                & MODULE._group_set(protocol.validation_query)
            )
            self.assertTrue(
                set(protocol.validation_query["stu_id"].astype(str))
                <= set(expected_validation)
            )
            self.assertTrue(
                protocol.validation_query["cpt_seq"].astype(str).str.contains("c0").all()
            )

    def test_flipping_validation_labels_cannot_change_protocol_or_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as first_raw, tempfile.TemporaryDirectory() as second_raw:
            first = Path(first_raw)
            second = Path(second_raw)
            self._write_source(first)
            pd.read_csv(first / "train.csv").to_csv(second / "train.csv", index=False)
            pd.read_csv(first / "Q_matrix.csv").to_csv(
                second / "Q_matrix.csv", index=False
            )
            flipped_valid = pd.read_csv(first / "valid.csv")
            flipped_valid["label"] = 1 - flipped_valid["label"]
            flipped_valid.to_csv(second / "valid.csv", index=False)

            one = MODULE.load_validation_only_protocol("synthetic", first)
            two = MODULE.load_validation_only_protocol("synthetic", second)
            for field in (
                "optimizer_students",
                "validation_students",
                "validation_support_rows",
                "validation_query_rows",
            ):
                self.assertEqual(one.audit["hashes"][field], two.audit["hashes"][field])
            self.assertEqual(
                one.audit["source_files"]["valid.csv"]["feature_projection_sha256"],
                two.audit["source_files"]["valid.csv"]["feature_projection_sha256"],
            )

            def mapping_hash(protocol: object) -> str:
                profiles, _ = MODULE.build_optimizer_profiles(
                    protocol.optimizer_train, q_lookup=protocol.q_lookup
                )
                items = sorted(protocol.q_lookup)
                item_index = {
                    item: index for index, item in enumerate(items)
                }
                statistics, _ = MODULE.build_reference_statistics(
                    profiles,
                    items=items,
                    item_index=item_index,
                    q_lookup=protocol.q_lookup,
                )
                validation_profiles = MODULE.build_validation_profiles(
                    protocol.validation_support,
                    protocol.validation_query,
                    q_lookup=protocol.q_lookup,
                )
                mapping = MODULE.build_donor_mapping(
                    validation_profiles,
                    statistics=statistics,
                    item_index=item_index,
                    q_lookup=protocol.q_lookup,
                    split="validation_query",
                    fold=None,
                    replicate=0,
                )
                return mapping.audit["mapping_sha256"]

            self.assertEqual(mapping_hash(one), mapping_hash(two))

    def test_validation_labels_join_by_id_and_preserve_prediction_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "valid.csv"
            source = pd.DataFrame(
                {
                    "stu_id": [1, 2, 3],
                    "exer_id": [10, 20, 30],
                    "cpt_seq": ["c0", "c1", "c2"],
                    "split_row_index": [90, 10, 50],
                    "label": [0, 1, 1],
                }
            )
            source.to_csv(path, index=False)
            features = MODULE.load_validation_feature_rows(path).iloc[[2, 0]].copy()
            predictions = features.loc[:, ["source_row_id"]].assign(
                probability=[0.8, 0.2]
            )
            order_sha256 = MODULE.validation_row_order_sha256(predictions)
            valid_sha256 = MODULE.sha256_file(path)
            labels = MODULE.load_validation_labels_for_evaluation(
                path,
                feature_rows=features,
                expected_valid_sha256=valid_sha256,
            )
            shuffled_labels = labels.sample(frac=1.0, random_state=7)
            shuffled_labels.attrs.update(labels.attrs)
            joined = MODULE.join_validation_predictions_with_labels(
                predictions,
                shuffled_labels,
                expected_prediction_order_sha256=order_sha256,
            )

            self.assertEqual(
                joined["source_row_id"].tolist(),
                predictions["source_row_id"].tolist(),
            )
            self.assertEqual(joined["label"].astype(int).tolist(), [1, 0])
            self.assertEqual(
                joined.attrs["valid_full_sha256_before"], valid_sha256
            )
            self.assertEqual(
                joined.attrs["valid_full_sha256_after"], valid_sha256
            )
            self.assertEqual(
                joined.attrs["prediction_row_order_sha256_before"],
                order_sha256,
            )
            self.assertEqual(
                joined.attrs["prediction_row_order_sha256_after"],
                order_sha256,
            )

    def test_validation_label_loader_rejects_equal_length_wrong_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "valid.csv"
            source = pd.DataFrame(
                {
                    "stu_id": [1, 2],
                    "exer_id": [10, 20],
                    "cpt_seq": ["c0", "c1"],
                    "split_row_index": [0, 1],
                    "label": [0, 1],
                }
            )
            source.to_csv(path, index=False)
            features = MODULE.load_validation_feature_rows(path)
            wrong = source.copy()
            wrong.loc[1, "exer_id"] = 999
            wrong.to_csv(path, index=False)

            with self.assertRaisesRegex(RuntimeError, "ID sets differ"):
                MODULE.load_validation_labels_for_evaluation(
                    path,
                    feature_rows=features,
                    expected_valid_sha256=MODULE.sha256_file(path),
                )

    def test_validation_label_join_rejects_duplicate_ids_and_bad_file_sha(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "valid.csv"
            pd.DataFrame(
                {
                    "stu_id": [1, 2],
                    "exer_id": [10, 20],
                    "cpt_seq": ["c0", "c1"],
                    "label": [0, 1],
                }
            ).to_csv(path, index=False)
            features = MODULE.load_validation_feature_rows(path)
            with self.assertRaisesRegex(RuntimeError, "full-file SHA-256"):
                MODULE.load_validation_labels_for_evaluation(
                    path,
                    feature_rows=features,
                    expected_valid_sha256="0" * 64,
                )

            labels = MODULE.load_validation_labels_for_evaluation(
                path,
                feature_rows=features,
                expected_valid_sha256=MODULE.sha256_file(path),
            )
            predictions = features.loc[:, ["source_row_id"]].assign(
                probability=[0.1, 0.9]
            )
            duplicate_predictions = pd.concat(
                [predictions, predictions.iloc[[0]]], ignore_index=True
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate source_row_id"):
                MODULE.join_validation_predictions_with_labels(
                    duplicate_predictions, labels
                )

    def test_retained_optimizer_items_finalize_validation_targets(self) -> None:
        common_items = [f"i{index}" for index in range(13)]
        skipped_only_item = "skipped-only"

        def rows(student: str, items: list[str], prefix: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    _interaction(
                        student,
                        item,
                        f"{prefix}-{index}",
                        label=index % 2,
                        concept="c",
                    )
                    for index, item in enumerate(items)
                ]
            )

        kept_support = rows("optimizer-kept", common_items[:10], "kept-s")
        kept_query = rows("optimizer-kept", common_items[10:], "kept-q")
        retained_profile = MODULE.StudentProfile(
            student="optimizer-kept",
            support=kept_support,
            query=kept_query,
            theta=0.0,
            raw_accuracy=0.5,
            seen_concepts=frozenset({"c"}),
            fold=0,
        )
        optimizer_train = pd.concat(
            [
                kept_support,
                kept_query,
                rows("optimizer-skipped", [skipped_only_item], "skipped"),
            ],
            ignore_index=True,
        )
        validation_support = rows("validation", common_items[:10], "valid-s")
        validation_query = rows(
            "validation",
            common_items[10:] + [skipped_only_item],
            "valid-q",
        )
        initial_query_hash = MODULE._hash_values(
            validation_query["source_row_id"].astype(str)
        )
        audit = {
            "assigned_students": {"validation_retained": 1},
            "rows": {
                "validation_support": len(validation_support),
                "validation_query": len(validation_query),
                "validation_support_unknown_item_removed": 0,
                "validation_query_unknown_item_removed": 0,
                "validation_query_support_group_overlap_removed": 0,
            },
            "known_items": {
                "provisional_optimizer_role_union": len(common_items) + 1,
                "finalized_after_optimizer_retention": False,
            },
            "hashes": {
                "validation_students": MODULE._hash_values(["validation"]),
                "validation_support_rows": MODULE._hash_values(
                    validation_support["source_row_id"].astype(str)
                ),
                "validation_query_rows": initial_query_hash,
            },
        }
        protocol = MODULE.ValidationOnlyProtocol(
            dataset="synthetic",
            source_dir=Path("."),
            optimizer_train=optimizer_train,
            validation_support=validation_support,
            validation_query=validation_query,
            q_lookup={
                item: ("c",) for item in common_items + [skipped_only_item]
            },
            audit=audit,
        )

        finalized = MODULE.finalize_protocol_for_retained_optimizer(
            protocol, [retained_profile]
        )

        self.assertNotIn(
            skipped_only_item,
            set(finalized.validation_query["exer_id"].astype(str)),
        )
        self.assertEqual(len(finalized.validation_query), 3)
        self.assertEqual(
            finalized.audit["assigned_students"]["validation_retained"], 1
        )
        self.assertEqual(
            finalized.audit["known_items"]["retained_optimizer_profile_union"],
            len(common_items),
        )
        self.assertEqual(
            finalized.audit["known_items"]["new_validation_query_rows_removed"],
            1,
        )
        self.assertTrue(
            finalized.audit["known_items"][
                "finalized_after_optimizer_retention"
            ]
        )
        self.assertNotEqual(
            finalized.audit["hashes"]["validation_query_rows"],
            initial_query_hash,
        )


class TestOptimizerAndDonorProtocol(unittest.TestCase):
    def test_optimizer_split_is_atomic_and_folds_are_deterministic(self) -> None:
        rows = []
        q_lookup = {f"i{index}": (f"c{index % 3}",) for index in range(30)}
        for student_index in range(120):
            student = f"o{student_index:03d}"
            for item_index in range(30):
                item = f"i{item_index}"
                for attempt in range(2):
                    rows.append(
                        _interaction(
                            student,
                            item,
                            f"{student}-{item}-{attempt}",
                            label=(student_index + item_index + attempt) % 2,
                            concept=q_lookup[item][0],
                        )
                    )
        frame = pd.DataFrame(rows)
        first, first_audit = MODULE.build_optimizer_profiles(
            frame, q_lookup=q_lookup
        )
        second, second_audit = MODULE.build_optimizer_profiles(
            frame.sample(frac=1.0, random_state=19),
            q_lookup=q_lookup,
        )
        self.assertEqual(
            first_audit["fold_assignment_sha256"],
            second_audit["fold_assignment_sha256"],
        )
        first_assignment = {profile.student: profile.fold for profile in first}
        second_assignment = {profile.student: profile.fold for profile in second}
        self.assertEqual(first_assignment, second_assignment)
        for profile in first:
            support_items = set(profile.support["exer_id"].astype(str))
            query_items = set(profile.query["exer_id"].astype(str))
            self.assertFalse(support_items & query_items)
            for item in support_items:
                self.assertEqual(
                    int((profile.support["exer_id"].astype(str) == item).sum()),
                    2,
                )
            for item in query_items:
                self.assertEqual(
                    int((profile.query["exer_id"].astype(str) == item).sum()),
                    2,
                )

    def test_donor_mapping_is_label_blind_group_atomic_and_has_no_fixed_point(self) -> None:
        items = ["a", "b", "c", "d"]
        item_index = {item: index for index, item in enumerate(items)}
        q_lookup = {item: ("concept",) for item in items}
        profile = _profile("student", items, repeated_first=True)
        statistics = _statistics(len(items))
        first = MODULE.build_donor_mapping(
            [profile],
            statistics=statistics,
            item_index=item_index,
            q_lookup=q_lookup,
            split="validation_query",
            fold=None,
            replicate=0,
        )

        flipped = MODULE.StudentProfile(
            student=profile.student,
            support=profile.support,
            query=profile.query.assign(label=1 - profile.query["label"]),
            theta=profile.theta,
            raw_accuracy=profile.raw_accuracy,
            seen_concepts=profile.seen_concepts,
            fold=profile.fold,
        )
        shuffled = MODULE.StudentProfile(
            student=profile.student,
            support=profile.support,
            query=flipped.query.sample(frac=1.0, random_state=7),
            theta=profile.theta,
            raw_accuracy=profile.raw_accuracy,
            seen_concepts=profile.seen_concepts,
            fold=profile.fold,
        )
        second = MODULE.build_donor_mapping(
            [shuffled],
            statistics=statistics,
            item_index=item_index,
            q_lookup=q_lookup,
            split="validation_query",
            fold=None,
            replicate=0,
        )

        self.assertEqual(first.audit["mapping_sha256"], second.audit["mapping_sha256"])
        self.assertEqual(first.audit["eligible_fixed_points"], 0)
        self.assertEqual(first.audit["cross_stratum_fallbacks"], 0)
        self.assertEqual(first.audit["label_columns_read_for_mapping"], [])
        self.assertTrue(first.rows["changed"].all())
        self.assertTrue(
            (
                first.rows["target_item"].astype(str)
                != first.rows["donor_target_item"].astype(str)
            ).all()
        )
        repeated = first.rows.loc[first.rows["target_item"] == "a"]
        self.assertEqual(repeated["donor_target_item"].nunique(), 1)
        self.assertEqual(first.audit["cells"][0]["distinct_items"], 4)

    def test_donor_never_crosses_cardinality_or_ease_stratum(self) -> None:
        items = ["a", "b", "c", "d", "e"]
        item_index = {item: index for index, item in enumerate(items)}
        q_lookup = {
            "a": ("c0",),
            "b": ("c1",),
            "c": ("c0", "c1"),
            "d": ("c1", "c2"),
            "e": ("c0", "c1", "c2"),
        }
        profile = _profile("student", items)
        statistics = _statistics(len(items), equal_ease=False)
        mapping = MODULE.build_donor_mapping(
            [profile],
            statistics=statistics,
            item_index=item_index,
            q_lookup=q_lookup,
            split="validation_query",
            fold=None,
            replicate=0,
        )
        edges = MODULE.ease_tercile_edges(statistics)
        for row in mapping.rows.itertuples(index=False):
            true_stratum = (
                MODULE.q_cardinality_bucket(q_lookup[str(row.target_item)]),
                MODULE._ease_tercile(
                    statistics.item_ease[item_index[str(row.target_item)]],
                    edges,
                ),
            )
            donor_stratum = (
                MODULE.q_cardinality_bucket(q_lookup[str(row.donor_target_item)]),
                MODULE._ease_tercile(
                    statistics.item_ease[item_index[str(row.donor_target_item)]],
                    edges,
                ),
            )
            self.assertEqual(true_stratum, donor_stratum)
        self.assertEqual(mapping.audit["eligible_fixed_points"], 0)

    def test_singleton_cells_are_not_silently_relaxed(self) -> None:
        items = ["a", "b", "c"]
        item_index = {item: index for index, item in enumerate(items)}
        q_lookup = {
            "a": ("c0",),
            "b": ("c0", "c1"),
            "c": ("c0", "c1", "c2"),
        }
        mapping = MODULE.build_donor_mapping(
            [_profile("student", items)],
            statistics=_statistics(len(items)),
            item_index=item_index,
            q_lookup=q_lookup,
            split="validation_query",
            fold=None,
            replicate=0,
        )
        self.assertFalse(mapping.rows["movable"].any())
        self.assertFalse(mapping.rows["changed"].any())
        self.assertEqual(mapping.audit["cross_stratum_fallbacks"], 0)
        self.assertFalse(mapping.audit["identified"])

    def test_fold_reference_excludes_held_students(self) -> None:
        items = ["a", "b"]
        item_index = {"a": 0, "b": 1}
        q_lookup = {"a": ("c",), "b": ("c",)}
        profiles = [
            _profile(f"s{index}", items, fold=index % MODULE.NUM_FOLDS)
            for index in range(20)
        ]
        references = MODULE.build_fold_references(
            profiles,
            items=items,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        self.assertEqual(len(references), MODULE.NUM_FOLDS)
        for reference in references:
            held = {
                profile.student
                for profile in profiles
                if profile.fold == reference.fold
            }
            self.assertFalse(held & set(reference.reference_students))
            self.assertEqual(
                len(reference.reference_students),
                len(profiles) - len(held),
            )


if __name__ == "__main__":
    unittest.main()
