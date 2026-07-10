# 外部插件代码来源 / Vendored Plugin Provenance

本目录中的 ORCDF 与 SVGCD 是从 Claude 实验参考工作区机械复制的内部适配版，
并非上游官方仓库的逐行镜像：

- `ORCDF/` 来源：`/home/xph/jwc/research/decoupled_cd_v2/external/ORCDF`
- `SVGCD/` 来源：`/home/xph/jwc/research/decoupled_cd_v2/external/SVGCD`

复制时排除了 `__pycache__/`、`*.pyc`、`logs*/`、`cache*/`、日志、checkpoint、
NumPy 数组、预测与运行指标等缓存或运行产物。2026-07-10 的审计复合 SHA-256 为：

- ORCDF: `09291bbc560cc4898baf63c64205195e180e1a040a37d0821c7c60988e5c299f`
- SVGCD: `28a9ec06ea6823e4dd0797d1947c8014ed3367accb38e133d359221b8300a44f`

复核口径是在来源目录的父目录运行以下命令。文件相对路径会进入第一层
`sha256sum` 输出，因此既校验内容也校验路径；清单按 NUL 分隔排序后再做一次
SHA-256：

```bash
cd /home/xph/jwc/research/decoupled_cd_v2/external
find ORCDF -type f ! -path '*/__pycache__/*' ! -name '*.pyc' \
  ! -path '*/logs*/*' ! -path '*/cache*/*' -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
find SVGCD -type f ! -path '*/__pycache__/*' ! -name '*.pyc' \
  ! -path '*/logs*/*' ! -path '*/cache*/*' -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

两个来源树均未包含 `LICENSE` 文件。These copies are restricted to remote,
internal experiments. Do not push them, publish them, or redistribute them until
the upstream licensing and attribution requirements have been resolved.
