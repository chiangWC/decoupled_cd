import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="SVGCD adapted to the current cognitive diagnosis dataset"
    )

    parser.add_argument("--data_dir", type=str, default="../data/assist09", help="Data directory")
    parser.add_argument("--log_dir", type=str, default="../logs/SVGCD", help="Log directory")

    parser.add_argument("--train_file", type=str, default="train_09.csv")
    parser.add_argument("--valid_file", type=str, default="valid_09.csv")
    parser.add_argument("--test_file", type=str, default="test_09.csv")

    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--dnn_units", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--n_gnn_layer", type=int, default=2)
    parser.add_argument("--cl_tau", type=float, default=0.7)
    parser.add_argument("--cl_weight", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument(
        "--longtail_bin_edges",
        type=str,
        default="0,20,40,60,80,100,150,250,500",
        help="Comma-separated student interaction count bin edges for RQ2-style long-tail evaluation.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--experiment_name", type=str, default="SVGCD_full")

    return parser.parse_args()
