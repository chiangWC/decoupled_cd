# Remote R Plotting For Gate Diagnostic

## Goal

Create visual plots for experiment 118 gate diagnostics using an R environment
on the remote execution host.

## Requirements

* Check whether R is already available on the remote host.
* If missing, install R into the remote conda environment without using system
  package manager assumptions.
* Use existing gate diagnostic artifacts under `results/gate_diagnostic_exp118/`.
* Generate publication-friendly plot files, at minimum PNG and preferably PDF.
* Do not push to GitHub.

## Acceptance Criteria

* [ ] Remote R or Rscript command is available.
* [ ] Gate diagnostic plot artifacts are generated remotely.
* [ ] Final response reports paths and whether installation was needed.

## Out Of Scope

* Re-running model training.
* Changing experiment conclusions.
* Installing system-wide packages with sudo.
