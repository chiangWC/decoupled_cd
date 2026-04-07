# Workflow

这个项目采用固定的“本地改代码，远端运行所有项目命令”流程。

## 路径与环境

- 本地项目目录: `/home/jameschiang/work/decoupled_cd`
- 远端主机别名: `xph-pc`
- 远端项目目录: `/home/xph/jwc/research/decoupled_cd`
- conda 环境: `decoupled_cd`

## 工作规则

- 默认所有代码修改都先在本地完成。
- 默认不要直接改远端代码，除非用户明确要求。
- 所有项目命令都在远端执行，包括训练、评估、测试和 smoke test。
- 本地不要直接跑项目代码；如需验证，先把本地代码同步到远端，再通过 SSH 执行命令。
- 远端执行任何项目命令前，先激活 `decoupled_cd` 环境。
- 本地使用 `git` 管理版本；远端作为唯一运行环境。

## 常用命令

本地仅用于进入仓库或做非运行类操作。

如果只是本地查看环境，可以用:

```bash
bash scripts/enter_env.sh
```

推送本地代码到远端:

```bash
bash scripts/sync_to_remote.sh
```

在远端项目环境里执行任意命令:

```bash
bash scripts/remote_exec.sh python scripts/train.py
```

从远端拉回代码到本地:

```bash
bash scripts/sync_from_remote.sh
```

远端运行示例:

```bash
bash scripts/remote_exec.sh python scripts/train.py
```

默认同步时排除以下目录:

- `__pycache__/`
- `.venv/`
- `logs/`
- `results/`

## 新会话最小开场

新开会话时，默认先读:

1. `docs/workflow.md`
2. `docs/handoff.md`

如果第一条消息要写得尽量短，可以直接说:

```text
先读 docs/workflow.md 和 docs/handoff.md，并按其中约定工作。
所有项目代码都在远端主机上运行；先同步并激活 decoupled_cd 环境。
```
