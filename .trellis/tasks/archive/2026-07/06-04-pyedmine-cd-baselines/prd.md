# Run PyEdmine CD Baselines

## Goal

Run all PyEdmine cognitive diagnosis baselines on the datasets and splits used by experiments 116-119, so the paper can compare against published CD models instead of only prior internal project baselines.

## What I Already Know

* The user wants all PyEdmine CD models attempted, not only DINA/IRT/MIRT/NCD.
* PyEdmine CD train scripts found on `xph-pc:/home/xph/jwc/pyedmine`:
  * `dina.py`
  * `irt.py`
  * `mirt.py`
  * `ncd.py`
  * `rcd.py`
  * `hier_cdf.py`
  * `hyper_cd.py`
* Smoke tests already completed with `SCD` conda env for `DINA`, `IRT`, `MIRT`, and `NCD` on small Junyi CD samples.
* `RCD` needs DGL. `SCD` has DGL but fails loading `libcudart.so.11.0`; `hyperbolic_cd` can import DGL.
* `hyperbolic_cd` was missing PyEdmine/evaluator side-effect imports; `wandb` and `torch_geometric` were installed there for this batch.
* `HierCDF` needs a generated hierarchy file derived from RCD directed concept graph.
* `HyperCD` needs a generated user hypergraph `.npy` per train split.

## Dataset Scope

Standard/original splits:

* ASSIST09: `/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered`
* ASSIST17: `/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_source`
* NIPS34: `/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_source`
* Junyi: `/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_source`

Student-concept holdout splits:

* ASSIST09: `/tmp/assist09_holdout_seed2024`
* ASSIST17: `/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_holdout_seed2024`
* NIPS34: `/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_holdout_seed2024`
* Junyi: `/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_holdout_seed2024`

## Assumptions

* "跑一遍" means one run per model/dataset/split with seed `2024`.
* Baseline comparison should use current project train/valid/test splits, not PyEdmine's default random splits.
* It is acceptable to write generated PyEdmine-formatted datasets, graph files, logs, and result summaries under a dedicated remote working directory outside this repository's normal tracked source files.
* Full batch jobs may be long-running; the implementation should support resumable execution and log inspection rather than requiring a single blocking shell command.

## Requirements

* Convert current project CD CSV format (`train.csv`, `valid.csv`, `test.csv`, `Q_matrix.csv`) into PyEdmine CD dataset format.
* Preserve each split exactly: no re-splitting for the baseline runs.
* Run all seven PyEdmine CD models where possible:
  * DINA
  * IRT
  * MIRT
  * NCD
  * RCD
  * HierCDF
  * HyperCD
* Generate required graph/preprocessing artifacts for RCD, HierCDF, and HyperCD.
* Record stdout/stderr logs, exit codes, command lines, dataset/split/model metadata, and output locations.
* Make the runner resumable so completed runs are skipped unless explicitly forced.
* Use remote `xph-pc` GPU resources deliberately:
  * detect available GPUs and current memory usage before launch;
  * support assigning each run to a specific `CUDA_VISIBLE_DEVICES`;
  * avoid launching multiple large jobs on the same saturated GPU;
  * keep CPU mode available for tiny smoke tests and failure isolation;
  * log GPU assignment per run.
* Keep remote PyEdmine source changes minimal; prefer external helper scripts and generated artifacts under a dedicated experiment directory.

## Acceptance Criteria

* [x] A repeatable runner exists for converting all target datasets/splits to PyEdmine format.
* [x] A repeatable runner exists for launching all seven PyEdmine CD baselines.
* [x] At least one end-to-end small/smoke run validates conversion plus training for a direct CD model.
* [x] RCD/DGL environment status is explicitly verified and either fixed or recorded as the blocker.
* [x] Generated run metadata clearly identifies which model/dataset/split combinations completed, failed, or are pending.
* [x] Runner can schedule jobs across available remote GPUs without oversubscribing obvious memory constraints.

## Out of Scope

* Multi-seed expansion beyond seed `2024`.
* Changing PyEdmine model implementations unless required for compatibility with our data format.
* Adding new paper tables to LaTeX in this task.
* Running KT baselines.

## Technical Notes

* The current project's data format uses `stu_id`, `exer_id`, `cpt_seq`, and `label` in interaction CSVs, with `Q_matrix.csv` mapping exercises to concepts.
* PyEdmine CD format is line-based with interaction fields such as `user_id`, `question_id`, and `correctness`; it also expects a dataset statics file and `Q_table.npy` through `FileManager`.
* PyEdmine train scripts should be invoked with `PYTHONPATH=/home/xph/jwc/pyedmine`.
* Direct model smoke command pattern used successfully:
  `PYTHONPATH=$PWD /home/xph/anaconda3/envs/SCD/bin/python examples/cognitive_diagnosis/train/ncd.py --dataset_name junyi2015 --train_file_name /tmp/pyedmine_smoke_train.txt --valid_file_name /tmp/pyedmine_smoke_valid.txt --max_epoch 1 --use_cpu True --save_model False --use_wandb False`
* Remote runner: `/home/xph/jwc/research/local_data/pyedmine_cd_baselines/pyedmine_cd_baselines.py`
* Remote work dir: `/home/xph/jwc/research/local_data/pyedmine_cd_baselines`
* Current background scheduler PID: `1704878`
* Resume command, preserving completed jobs:
  `/home/xph/anaconda3/envs/hyperbolic_cd/bin/python pyedmine_cd_baselines.py --pyedmine-root /home/xph/jwc/pyedmine --work-dir /home/xph/jwc/research/local_data/pyedmine_cd_baselines run --datasets all --models all --max-epoch 50 --max-parallel 4 --min-free-gb 8 --poll-seconds 20`
