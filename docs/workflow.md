# Workflow

这个项目采用固定的“本地开发，远端运行”流程。

## 路径与环境

- 本地项目目录: `/home/jameschiang/work/decoupled_cd`
- 远端主机别名: `xph-pc`
- 远端项目目录: `/home/xph/jwc/research/decoupled_cd`
- conda 环境: `decoupled_cd`

## 工作规则

- 默认所有代码修改都先在本地完成。
- 默认不要直接改远端代码，除非用户明确要求。
- 如需去远端运行，先把本地代码同步到远端，再通过 SSH 执行命令。
- 远端执行任何项目命令前，先激活 `decoupled_cd` 环境。
- 本地使用 `git` 管理版本；远端主要作为运行环境。

## 常用命令

进入本地环境:

```bash
bash scripts/enter_env.sh
```

激活环境后直接执行命令:

```bash
bash scripts/enter_env.sh python scripts/train.py
```

推送本地代码到远端:

```bash
bash scripts/sync_to_remote.sh
```

从远端拉回代码到本地:

```bash
bash scripts/sync_from_remote.sh
```

远端运行示例:

```bash
ssh xph-pc 'cd ~/jwc/research/decoupled_cd && conda activate decoupled_cd && python scripts/train.py'
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
如需去远端跑代码，先同步并激活 decoupled_cd 环境。
```
