# SVGCD Baseline

This folder now contains a closer SVGCD-style adaptation rather than the earlier loose conceptual baseline.

Implemented components:

- semantic-aware bipartite graphs for correct and wrong responses
- knowledge-integrated initialization that projects base student and exercise embeddings into concept space
- LightGCN-style semantic propagation on positive and negative graphs
- semantic-specific variational projection and graph reconstruction
- divide-and-conquer style contrastive learning on student and exercise semantics
- NeuralCD-style monotonic prediction head

Run:

```bash
cd SVGCD
python main.py
```

Note:

- This is still an adaptation to the current workspace and CSV data format, not a line-by-line reproduction of the original repository.
