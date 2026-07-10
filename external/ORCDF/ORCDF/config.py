import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="ORCDF adaptation for the current cognitive diagnosis dataset"
    )

    parser.add_argument("--data_dir", type=str, default="../data/assist09", help="Data directory")
    parser.add_argument("--log_dir", type=str, default="../logs/ORCDF", help="Log directory")

    parser.add_argument("--train_file", type=str, default="train_09.csv")
    parser.add_argument("--valid_file", type=str, default="valid_09.csv")
    parser.add_argument("--test_file", type=str, default="test_09.csv")

    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--gcn_layers", type=int, default=3)
    parser.add_argument("--keep_prob", type=float, default=0.9)
    parser.add_argument(
        "--if_type",
        type=str,
        default="ncd",
        choices=["ncd", "dp-linear", "dp-poly", "dp-rbf", "mirt", "kancd", "cdmfkc", "irt", "kscd"],
    )
    parser.add_argument("--mode", type=str, default="all", choices=["all", "R", "Q"])
    parser.add_argument("--flip_ratio", type=float, default=0.15)
    parser.add_argument("--ssl_temp", type=float, default=0.5)
    parser.add_argument("--ssl_weight", type=float, default=1e-3)
    parser.add_argument("--prednet_len1", type=int, default=512)
    parser.add_argument("--prednet_len2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--conflict_num_groups",
        type=int,
        default=10,
        help="Number of equal-frequency SampleConflict groups on the test split.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="ORCDF_assist09_tune3")

    return parser.parse_args()
