# Environment

这是环境命令速查页；协作流程以 `docs/workflow.md` 为准。

## 本地进入环境

```bash
bash scripts/enter_env.sh
```

如果希望激活环境后立刻执行命令:

```bash
bash scripts/enter_env.sh python scripts/train.py
```

手动进入:

```bash
conda activate decoupled_cd
cd /home/jameschiang/work/decoupled_cd
```

## 远端运行

先同步:

```bash
bash scripts/sync_to_remote.sh
```

再在远端激活环境并执行:

```bash
ssh xph-pc 'cd ~/jwc/research/decoupled_cd && conda activate decoupled_cd && <your-command>'
```
