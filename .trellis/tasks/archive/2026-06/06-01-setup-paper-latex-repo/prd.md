# setup paper latex repo

## Goal

Set up a remote LaTeX writing environment for a Chinese-first paper draft, and create a sibling paper repository that records its relationship to this project.

## What I already know

* The paper is based on `decoupled_cd_trellis`.
* The first draft will be written in Chinese and may later be translated into English.
* The paper should live outside the current engineering repository.
* The current remote initially had no `latex`, `pdflatex`, `xelatex`, `lualatex`, `latexmk`, `bibtex`, `biber`, `tlmgr`, `kpsewhich`, `tectonic`, or `pandoc`.
* The remote is Ubuntu 24.04.4 LTS with enough disk space.
* System package installation via `sudo apt` requires a password, so this task should use a user-local LaTeX installation.

## Requirements

* Install a usable LaTeX environment on the remote without requiring interactive sudo.
* Support Chinese drafting with `xelatex`.
* Create a sibling repository at `../decoupled_cd_paper`.
* Add repository instructions so future Codex sessions know the compile environment is remote and the related code repository is `../decoupled_cd_trellis`.
* Record the current code repository commit as the paper's initial code reference.

## Acceptance Criteria

* [x] `xelatex`, `latexmk`, `tlmgr`, and bibliography tooling are available from the shell.
* [x] A minimal Chinese LaTeX document compiles successfully in the paper repository.
* [x] `../decoupled_cd_paper` is initialized as a git repository.
* [x] The paper repository contains `AGENTS.md`, `README.md`, `.gitignore`, `Makefile`, source directories, and `experiments/code-ref.txt`.

## Out of Scope

* Creating a GitHub remote repository.
* Writing substantive paper content beyond a compile smoke test.
* Modifying project code in `decoupled_cd_trellis`.

## Technical Notes

* Current code commit at task creation: `7166c04a7e42c35666c87656dfcbef3ae7bc582e`.
* Prefer user-local TinyTeX because `sudo -n true` reports that a password is required.
