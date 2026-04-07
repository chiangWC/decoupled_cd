# Workflow

这个项目采用“本地协作，远端只负责运行代码”的固定流程。

## 目录约定

- 本地项目目录: `/home/jameschiang/work/decoupled_cd`
- 远端主机别名: `xph-pc`
- 远端项目目录: `/home/xph/jwc/research/decoupled_cd`

## 工作原则

- 所有代码修改默认先在本地完成。
- 默认不要直接修改远端代码，除非用户明确要求。
- 需要在远端运行时，先把本地代码同步到远端，再通过 SSH 到远端执行命令。
- 远端执行任何项目命令前，先激活 `decoupled_cd` 环境。
- 本地使用 `git` 管理版本；远端主要作为运行环境。

## 同步命令

推送本地代码到远端:

```bash
bash scripts/sync_to_remote.sh
```

从远端拉回代码到本地:

```bash
bash scripts/sync_from_remote.sh
```

默认同步时排除以下目录:

- `__pycache__/`
- `.venv/`
- `logs/`
- `results/`

## 推荐的会话开场

新开一个本地会话时，先在本地项目目录打开工作区。

然后在第一条消息中说明:

```text
先读 docs/workflow.md，并按其中流程工作。
如需去远端跑代码，记得先激活 decoupled_cd 环境。
```

## 远端运行示例

```bash
ssh xph-pc 'cd ~/jwc/research/decoupled_cd && conda activate decoupled_cd && python scripts/train.py'
```
