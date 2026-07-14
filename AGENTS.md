# Repository Guidelines

## 项目结构与模块组织

本项目是 Python/PyTorch 认知诊断研究代码。`models/` 包含 v1、v2、集成及基线模型；`data/` 负责读取交互记录、ID 映射、Q 矩阵、概念图和数据分割；`trainers/engine.py` 实现训练与评估循环；`configs/` 保存数据集默认参数；`utils/` 提供指标、设备、日志、随机种子和文件输出工具。可执行入口与实验分析位于 `scripts/`，研究说明位于 `docs/`。`results/`、`logs/`、检查点及 `data/assist_09_ordered/` 均为生成资产，不应提交。

## 构建、测试与开发命令

先进入 `decoupled_cd` Conda 环境，再从仓库根目录运行：

```bash
conda activate decoupled_cd
python scripts/train.py --dataset assist_09
python scripts/train.py --dataset assist_09 --epochs 2 --max-rows 5000
bash scripts/run_assist09_baseline.sh
python scripts/evaluate.py --help
```

第一条训练命令使用 `configs/defaults.py`；第二条是快速冒烟检查；基线脚本运行正式 ASSIST09 配置；评估脚本的 `--help` 展示检查点与数据参数。预处理数据可运行 `python scripts/preprocess_assist09_ordered.py --raw-csv <path>`，随后执行概念图构建脚本。

## 代码风格与命名约定

使用 4 空格缩进、PEP 8 风格和类型标注。模块、函数、变量采用 `snake_case`，类采用 `PascalCase`，常量采用 `UPPER_SNAKE_CASE`；CLI 参数使用 `--kebab-case`。优先延续 `from __future__ import annotations`、`pathlib.Path` 和 dataclass 等现有模式。仓库未配置 formatter 或 linter，提交前至少运行 `python -m compileall configs data models trainers utils scripts` 并检查 `git diff --check`。

## 验证与回归规范

当前没有独立测试套件或覆盖率门槛；验证依赖小规模训练、相关评估脚本及指标对比。修改模型或数据管线后至少运行冒烟命令，并在可行时复跑受影响的基线。验证集和测试集只能使用训练历史，禁止将目标交互写入传播历史。新增实验能力应由默认关闭的参数控制，并尽量零初始化以保持旧结果可复现。新增训练参数必须同步到 argparse、模型构造、输出 JSON 和 `results/experiment_results.csv` 汇总字段。

## 提交与 Pull Request 规范

历史提交采用 `feat:`、`fix:`、`docs:` 前缀；保持主题简短、使用祈使语气，并让一次提交只表达一个实验或工程目的。PR 应说明动机、影响范围、精确验证命令、数据集、seed 及前后指标；关联已有 issue。不要提交数据、日志或检查点。仅当可视化输出发生变化时附截图，否则使用指标表或结果摘要。

## 远端实验与配置

GPU 主机的 `/home/xph/jwc/research/decoupled_cd_codex` 是本仓库的实验工作区；实验前确认其分支和 commit 与本地一致，并在远端激活 `decoupled_cd`。`/home/xph/jwc/research/decoupled_cd_v2` 是 Claude 实验参考目录，只用于接收结果和比较候选改动，不是 Codex 代码来源。`scripts/remote_exec.sh` 目前仍指向旧路径，更新前不要用于 Codex 实验。不得提交凭据、机器专用路径改写或私有数据。

正式实验代码的开发、提交和运行统一在上述 xph 工作区完成。本地仓库只用于只读审查和同步已提交分支，不为实验轮次创建额外 worktree。启动训练前要求 `git status --short` 为空、HEAD 已推送到远端；禁止通过 `rsync`、`scp` 或其他文件级方式部署未提交源码。xph 同时只保留一个活动 worktree，失败轮次的结果先集中归档，再删除其工作树。
