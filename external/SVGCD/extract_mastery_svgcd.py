"""Extract the (S, K) mastery matrix from a trained SVGCD checkpoint."""
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SVGCD.config import parse_args
from SVGCD.dataset import CognitiveDataProcessor
from SVGCD.model import SVGCDNet
from SVGCD.utils import get_device, setup_logger


def main():
    args = parse_args()
    logger = setup_logger(args.log_dir)
    device = get_device()
    proc = CognitiveDataProcessor(args, logger)
    model = SVGCDNet(
        student_n=proc.num_students,
        exer_n=proc.num_exercises,
        knowledge_n=proc.num_concepts,
        args=args,
        pos_graph=proc.correct_adj,
        neg_graph=proc.wrong_adj,
        device=device,
    ).to(device)
    state = torch.load(os.path.join(args.log_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        stu_abit, _exer_diff = model.graph_representations()
        mastery = torch.sigmoid(model.prednet_stu(stu_abit))
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
