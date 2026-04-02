# Environment

这个项目默认使用 conda 环境 `decoupled_cd`。

## 启动 Codex CLI

建议从项目内使用下面这个脚本启动:

```bash
bash scripts/start_codex.sh
```

这个脚本会做两件事:

1. 激活 `decoupled_cd`
2. 切换到项目根目录并启动 `codex`

如果你想把参数继续传给 `codex`，也可以直接追加:

```bash
bash scripts/start_codex.sh <codex-args>
```

## 手动进入环境

```bash
conda activate decoupled_cd
cd /home/xph/jwc/research/decoupled_cd
```
