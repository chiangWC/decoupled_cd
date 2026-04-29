# Workflow

这份文档只描述协作与运行流程，不重复项目背景和实验结论。

新会话默认先读 [docs/session_bootstrap.md](./session_bootstrap.md)；只有需要展开具体流程时再读这份文档。

这个项目采用固定的“本地改代码并提交，推送到远端仓库，远端运行所有项目命令”流程。

## 路径与环境

- 本地项目目录: `/home/jameschiang/research/decoupled_cd`
- 远端主机别名: `xph-pc`
- 远端项目目录: `/home/xph/jwc/research/decoupled_cd`
- conda 环境: `decoupled_cd`

## 工作规则

- 默认所有代码修改都先在本地完成。
- 默认不要直接改远端代码，除非用户明确要求。
- 远端主机只作为运行环境，不作为代码协作来源。
- 所有项目命令都在远端执行，包括训练、评估、测试和 smoke test。
- 本地不要直接跑项目代码；如需验证，先把本地提交推到远端，再通过 SSH 执行命令。
- 开始工作前，默认先执行 `git status` 和 `git pull --ff-only origin master`。
- `master` 只保留当前认可状态；探索性实验默认在 `exp/*` 分支进行。
- 远端执行任何项目命令前，先激活 `decoupled_cd` 环境。
- 远端目录保留 `git` 工作树，仅用于接收已推送代码并直接运行。
- 远端执行跟随当前本地分支；未推送的本地改动不会被远端看到。

## 常用命令

本地仅用于进入仓库或做非运行类操作。

如果只是本地查看环境，可以用:

```bash
bash scripts/enter_env.sh
```

将当前分支推到远端运行机仓库:

```bash
git push origin "$(git branch --show-current)"
```

开始一个新实验分支:

```bash
git switch -c exp/<short-name>
```

在远端项目环境里执行任意命令:

```bash
bash scripts/remote_exec.sh python scripts/train.py
```

远端运行前默认要求:

- 本地工作树干净，且需要运行的修改已经 `git commit`
- `origin` 已配置到远端运行机仓库
- 本地与远端仓库历史兼容，可做 fast-forward / 正常 push

## 分支约定

- `master` 只保留当前认可状态，不把探索性实验直接堆到主线。
- 新实验默认从最新 `master` 切出 `exp/<short-name>` 分支。
- 实验效果不好时，保留或删除对应 `exp/*` 分支即可，不需要用回退提交污染 `master`。
- 实验效果成立后，再整理提交并合回 `master`。
- `git push origin "$(git branch --show-current)"` 会把当前本地分支推到远端同名分支。
- `bash scripts/remote_exec.sh ...` 会先确认远端同名分支已更新到当前本地提交，再在远端切到该分支执行命令。

## 实验节奏

- 探索性结构改动默认先开 `exp/<short-name>` 分支。
- 新结构默认先跑单次。
- 单次值得继续时，再补 `2-3` 个 seed。
- 实验结束时要明确代码是否合入主线；若已经形成有效/无效判断，即使代码不合入 `master`，也要用 doc-only 提交把结论同步回 `master` 台账。

## 文档分工

- [docs/session_bootstrap.md](./session_bootstrap.md) 只保留新会话必须先知道的最小规则、当前主线口径和按需再读入口。
- [docs/workflow.md](./workflow.md) 只保留协作、分支、远端运行和文档维护流程，不记录具体实验结论。
- [docs/handoff.md](./handoff.md) 只保留当前主线状态、已经会影响后续行动的稳定判断、当前优先分支和关键文件。
- [docs/model_improvement_plan.md](./model_improvement_plan.md) 是唯一的实验台账主档，负责记录实验号、做法、结果、切片观察和结论。

## 文档更新规则

- 新实验结束后，默认先只更新 `model_improvement_plan`。
- 只有当实验改变了当前主线、默认判断、默认优先级或当前推荐分支时，才额外更新 `handoff`。
- 只有当协作方式、远端运行流程、分支约定或文档维护流程本身发生变化时，才更新 `workflow`。
- 只有当新会话第一入口必须知道的最小规则发生变化时，才更新 `session_bootstrap`。
- 失败实验、局部 slice 信号、单 seed follow-up，如果没有改变默认判断，通常不进入 `handoff`。
- 文档更新后，应检查 `handoff` 和 `model_improvement_plan` 是否出现重复结论或重复指标；若有重复，保留 `model_improvement_plan` 的详细版本，`handoff` 只写压缩判断。

## 实验记录模板

写入 `model_improvement_plan` 的新实验条目默认按最小必要信息记录，通常包含:

- 实验号 / 分支名
- 做法: 改了什么，作用在什么模块，是否 zero-init / targeted / sidecar
- 对比基线: 相对哪个实验或当前主线
- 结果: 单 seed 或多 seed 的 overall 指标
- 切片: 可选；只在它真正影响判断、解释 overall 结果或支撑继续/暂停决策时记录
- 结论: 是否继续、是否扩 seed、是否合入主线、若再访优先改什么

同步到 `handoff` 时默认压缩成:

- 一句判断
- 是否已吸收到 `master` / 是否暂停
- 需要时附一条到 `model_improvement_plan` 的引用
