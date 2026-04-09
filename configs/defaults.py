from __future__ import annotations


DATASET_DEFAULTS = {
    "assist_09": {
        "train_interactions": "data/assist_09_ordered/train.csv",
        "valid_interactions": "data/assist_09_ordered/valid.csv",
        "test_interactions": "data/assist_09_ordered/test.csv",
        "q_matrix": "data/assist_09_ordered/Q_matrix.csv",
        "concept_graph": "data/assist_09_ordered/transition_graph/propagation_graph.csv",
        "graph_mode": "single",
        "epochs": 20,
        "learning_rate": 1e-3,
        "concept_dim": 64,
        "alpha": 1.0,
        "beta": 1.0,
        "early_stop_patience": 5,
        "device": "auto",
        "gpus": None,
    },
    "assist_17": {
        "train_interactions": "../ConceptSkillCDM/data/assist_17/train.csv",
        "valid_interactions": "../ConceptSkillCDM/data/assist_17/valid.csv",
        "test_interactions": "../ConceptSkillCDM/data/assist_17/test.csv",
        "epochs": 5,
        "learning_rate": 1e-3,
        "concept_dim": 16,
        "alpha": 1.0,
        "beta": 1.0,
        "early_stop_patience": 5,
        "device": "auto",
        "gpus": None,
    },
    "junyi": {
        "train_interactions": "../ConceptSkillCDM/data/junyi/train.csv",
        "valid_interactions": "../ConceptSkillCDM/data/junyi/valid.csv",
        "test_interactions": "../ConceptSkillCDM/data/junyi/test.csv",
        "epochs": 5,
        "learning_rate": 1e-3,
        "concept_dim": 16,
        "alpha": 1.0,
        "beta": 1.0,
        "early_stop_patience": 5,
        "device": "auto",
        "gpus": None,
    },
}


def apply_dataset_defaults(args, parser=None):
    dataset = getattr(args, "dataset", None)
    if dataset not in DATASET_DEFAULTS:
        return args

    defaults = DATASET_DEFAULTS[dataset]
    for key, value in defaults.items():
        if parser is not None and parser.get_default(key) is not None:
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
        else:
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    return args
