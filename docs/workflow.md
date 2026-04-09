# Workflow

这份文档只描述协作、部署与运行流程，不重复项目背景和实验结论。

新会话默认先读 [docs/session_bootstrap.md](/home/jameschiang/work/decoupled_cd/docs/session_bootstrap.md)；只有需要展开具体流程时再读这份文档。

这个项目采用固定的“本地改代码并提交，部署到远端运行机，远端运行所有项目命令”流程。

## 路径与环境

- 本地项目目录: `/home/jameschiang/work/decoupled_cd`
- 远端主机别名: `xph-pc`
- 远端项目目录: `/home/xph/jwc/research/decoupled_cd`
- conda 环境: `decoupled_cd`

## 工作规则

- 默认所有代码修改都先在本地完成。
- 默认不要直接改远端代码，除非用户明确要求。
- 远端主机只作为运行环境，不作为代码协作来源。
- 所有项目命令都在远端执行，包括训练、评估、测试和 smoke test。
- 本地不要直接跑项目代码；如需验证，先把本地提交部署到远端，再通过 SSH 执行命令。
- 如果两台开发机轮换使用，开始工作前先执行 `git pull --ff-only origin master`。
- 远端执行任何项目命令前，先激活 `decoupled_cd` 环境。
- 远端目录保留 `git` 工作树，仅用于接收部署后的代码并直接运行。
- 部署只会带上已经提交的改动；本地有未提交改动时，默认不部署。
- 默认约定是 `origin` 指向远端运行机仓库，部署时将本地当前 `HEAD` 推到 `origin` 默认分支，再在远端直接运行。

## 常用命令

本地仅用于进入仓库或做非运行类操作。

如果只是本地查看环境，可以用:

```bash
bash scripts/enter_env.sh
```

将本地已提交代码部署到远端:

```bash
bash scripts/deploy_to_remote.sh
```

在远端项目环境里执行任意命令:

```bash
bash scripts/remote_exec.sh python scripts/train.py
```

部署前默认要求:

- 本地工作树干净，且需要部署的修改已经 `git commit`
- `origin` 已配置到远端运行机仓库
- 本地与远端仓库历史兼容，可做 fast-forward / 正常 push

## 新会话说明

默认入口改为 [docs/session_bootstrap.md](/home/jameschiang/work/decoupled_cd/docs/session_bootstrap.md)。
