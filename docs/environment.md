# Environment

这个项目默认使用 conda 环境 `decoupled_cd`。本地负责开发与协作，远端只负责运行代码。

完整约定见 `docs/workflow.md`。

## 本地环境

建议先进入本地项目目录:

```bash
cd /home/jameschiang/work/decoupled_cd
```

如果希望自动激活 conda 环境，并直接进入项目目录，也可以使用下面这个脚本:

```bash
bash scripts/enter_env.sh
```

这个脚本会做两件事:

1. 激活 `decoupled_cd`
2. 切换到当前本地仓库根目录

如果你想在激活环境后直接执行命令，也可以直接追加:

```bash
bash scripts/enter_env.sh python scripts/train.py
```

## 手动进入环境

```bash
conda activate decoupled_cd
cd /home/jameschiang/work/decoupled_cd
```

## 同步到远端运行

本地改完代码后，先同步到远端:

```bash
bash scripts/sync_to_remote.sh
```

然后再登录远端执行命令:

```bash
ssh xph-pc 'cd ~/jwc/research/decoupled_cd && conda activate decoupled_cd && <your-command>'
```
