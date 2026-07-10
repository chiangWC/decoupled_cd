"""Extract the (S, K) mastery matrix from a trained ORCDF checkpoint."""
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ORCDF.config import parse_args
from ORCDF.dataset import CognitiveDataProcessor
from ORCDF.model import ORCDFNet
from ORCDF.utils import get_device, setup_logger


def main():
    args = parse_args()
    logger = setup_logger(args.log_dir)
    device = get_device()
    proc = CognitiveDataProcessor(args, logger)
    model = ORCDFNet(
        student_n=proc.num_students,
        exer_n=proc.num_exercises,
        knowledge_n=proc.num_concepts,
        latent_dim=args.latent_dim,
        gcn_layers=args.gcn_layers,
        keep_prob=args.keep_prob,
        if_type=args.if_type,
        mode=args.mode,
        flip_ratio=args.flip_ratio,
        ssl_temp=args.ssl_temp,
        ssl_weight=args.ssl_weight,
        prednet_len1=args.prednet_len1,
        prednet_len2=args.prednet_len2,
        dropout=args.dropout,
        device=device,
    ).to(device)
    model.get_graph_dict(proc.graph_dict)
    model.get_flip_graph()
    state = torch.load(os.path.join(args.log_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    num_students, num_concepts = proc.num_students, proc.num_concepts
    stu_ids = torch.arange(num_students, dtype=torch.long, device=device)
    exer_ids = torch.zeros(num_students, dtype=torch.long, device=device)
    param_dtype = next(model.parameters()).dtype
    q_dummy = torch.ones(num_students, num_concepts, dtype=param_dtype, device=device)
    with torch.no_grad():
        student_ts, diff_ts, disc_ts, knowledge_ts, extras = model.extractor.extract(stu_ids, exer_ids, q_dummy)
        mastery = model.inter_func.transform(student_ts, knowledge_ts)
    mastery = mastery.float().detach().cpu().numpy()
    np.save(os.path.join(args.log_dir, "mastery.npy"), mastery)
    with open(os.path.join(args.log_dir, "id_maps.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"stu_ids": [str(x) for x in proc.stu_ids], "cpt_ids": [str(x) for x in proc.cpt_ids]},
            f,
        )
    print("mastery saved:", mastery.shape)


if __name__ == "__main__":
    main()
