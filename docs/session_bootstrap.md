# Session Bootstrap

新会话默认先只读这份文档。它只保留当前项目最小必需上下文，目标是让接手成本尽量低。

## 最小规则

- 所有代码默认在本地修改，不直接改远端。
- 所有训练、评估、测试、smoke test 都在远端 `xph-pc` 上运行。
- 远端主机只作为运行环境，不作为人工代码编辑环境。
- 默认 `origin` 指向远端运行机上的项目仓库。
- 开始工作前，默认先执行 `git status` 和 `git pull --ff-only origin master`。
- `master` 只保留当前认可状态；探索性实验默认在 `exp/*` 分支进行。
- 本地改动先 `git commit`，再部署到远端运行机。

## 最常用命令

开始工作前检查并同步本地分支:

```bash
git status
git pull --ff-only origin master
```

本地部署已提交代码:

```bash
bash scripts/deploy_to_remote.sh
```

开始一个新实验分支:

```bash
git switch -c exp/<short-name>
```

在远端环境执行命令:

```bash
bash scripts/remote_exec.sh <your-command>
```

例如运行正式单次基线:

```bash
bash scripts/remote_exec.sh bash scripts/run_assist09_baseline.sh
```

## 当前主线

- 数据: `data/assist_09_ordered`
- 图: `data/assist_09_ordered/transition_graph/propagation_graph.csv`
- `learning_rate = 1e-3`
- `concept_dim = 64`
- `gs_mode = conditional`
- `graph_mode = single`
- `TKC/UKC` 结构传播参数独立
- `TKC` 行为消息默认使用正误双通道
- `TKC/UKC` 学生级融合默认使用自适应 gate
- 结构比较默认看 `300 epoch`

## 已定规则

- ordered ASSIST09 + transition graph 是当前固定主线。
- `dual graph` 只作为 legacy ablation，不是默认路径。
- `valid/test` 复用 `train` 行为历史做传播输入。
- 一次只改一个结构因素。
- 比较新结构时，默认先跑 `2-3` 个 seed。
- 结果未验证前，不要把探索性实验直接推到 `master`。

## 按需再读

- [docs/workflow.md](/home/jameschiang/work/decoupled_cd/docs/workflow.md)
  - 需要看完整协作/部署/运行规则时再读
- [docs/handoff.md](/home/jameschiang/work/decoupled_cd/docs/handoff.md)
  - 需要看当前主线细节、关键判断和关键文件时再读
- [README_spec.md](/home/jameschiang/work/decoupled_cd/README_spec.md)
  - 需要核对模型语义和 Step 1-4 定义时再读
- [docs/model_improvement_plan.md](/home/jameschiang/work/decoupled_cd/docs/model_improvement_plan.md)
  - 需要查历史实验和失败路线时再读
- [docs/transition_graph_notes.md](/home/jameschiang/work/decoupled_cd/docs/transition_graph_notes.md)
  - 需要修改构图逻辑时再读
- [docs/reuse_plan.md](/home/jameschiang/work/decoupled_cd/docs/reuse_plan.md)
  - 只有做工程复用或重构时再读

## 推荐开场

```text
先读 docs/session_bootstrap.md，并按其中约定工作。
如任务需要，再按文档里的“按需再读”继续展开。
```
