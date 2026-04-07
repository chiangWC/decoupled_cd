# Environment

这是环境命令速查页；协作流程以 `docs/workflow.md` 为准。

默认约定:

- 本地只改代码，不运行项目命令。
- 训练、评估、测试、smoke test 一律在远端 `xph-pc` 上执行。

## 本地进入环境

```bash
bash scripts/enter_env.sh
```

本地如果只是查看环境或做静态检查，可以使用上面的命令。

运行当前正式单次基线:

```bash
ssh xph-pc 'cd ~/jwc/research/decoupled_cd && conda activate decoupled_cd && bash scripts/run_assist09_baseline.sh'
```

运行当前正式多 seed 基线:

```bash
ssh xph-pc 'cd ~/jwc/research/decoupled_cd && conda activate decoupled_cd && bash scripts/run_assist09_multiseed.sh'
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

再在远端项目环境里执行:

```bash
bash scripts/remote_exec.sh <your-command>
```

例如远端运行当前正式单次基线:

```bash
bash scripts/remote_exec.sh bash scripts/run_assist09_baseline.sh
```

例如远端做 smoke test:

```bash
bash scripts/remote_exec.sh bash scripts/run_assist09_baseline.sh --epochs 1 --max-rows 128 --device cpu --output results/smoke_assist09_baseline.json
```
