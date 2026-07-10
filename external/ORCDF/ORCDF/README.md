# ORCDF

This folder contains a local runnable adaptation of ORCDF for the current
workspace data flow (`train/valid/test.csv` with `stu_id, exer_id, cpt_seq, label`).

Implemented upstream ideas:

- `right / wrong / all` student-exercise-knowledge graphs
- response-label flipping augmentation
- graph extractor + NCD-style interaction function
- self-supervised consistency between base and flipped graph views

Notes:

- This is a practical PyTorch adaptation for the local benchmark workflow.
- It does not reproduce the full upstream framework abstraction stack.
- The current local version only keeps the NCD-like interaction head.

Run:

```bash
cd ORCDF
python main.py
```
