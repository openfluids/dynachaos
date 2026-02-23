# dynachaos



Ricardo Frantz

## Objectives

This project reproduces and extends the complete arc of Kunihiko Kaneko's early work
on chaos (1982--1993), covering:

1. Circle maps and devil's staircases (period-adding, Arnold tongues)
2. Coupled logistic map: torus-to-chaos transition with symmetry breaking
3. Torus doubling in 3D and 4D delayed logistic maps
4. Oscillation and doubling of torus (delayed logistic map)
5. Fates of the three-torus (coupled delayed logistic, double devil's staircase)
6. Fractalization of torus (Grassberger--Procaccia correlation dimension)
7. Spatiotemporal intermittency in coupled map lattices (CML)
8. Pattern dynamics in CML (five phases, defect turbulence)
9. Globally coupled maps and violation of the law of large numbers
10. Modern diagnostics (0-1 test, SALI, permutation entropy, RQA) applied to all maps

**Framing**: Original research paper with review elements. The modern diagnostics
(Section 11) are the paper's original methodological contribution, with Sections 2--10
serving as benchmark systems reproduced at resolutions 100--1000x finer than Kaneko's
originals. New interpretations and connections to recent developments (toric chaos,
cross-feeding dynamics, data-driven methods) extend beyond pure review.

All code, data, and figures are openly available and reproducible with `uv run`.








**Subject classification**: A31 (cellular-automata and coupled map lattices),
cross-list with A33 (classical chaos), A54 (pattern formation)



- **Author Guidelines**: https://academic.oup.com/ptep/pages/General_Instructions
- **Manuscript Preparation**: https://academic.oup.com/ptep/pages/Manuscript_Preparation_Guidelines
- **Templates & Style Files**: https://academic.oup.com/ptep/pages/templates_and_style_files

- **Submission Portal**: https://publication.jps.jp/cgi-bin/ptep/submission/submission.cgi



- Document class: `\documentclass[preprint,pteplogo]{ptephy_v2}`
- References: numbered in order of appearance, in parentheses
- One reference per numbered entry
- First initials + last name only; article titles NOT required
- APC: 130,000 JPY (~850 CHF), but covered by SCOAP3 (effectively free OA)
- BibTeX style: `ptep-stl.bst` (provided with template)

## Repository Structure

```
paper/          LaTeX source (main.tex, references.bib)
src/
  dynachaos/
    maps/       Map definitions and figure scripts
    cml/        Coupled map lattice models
    diagnostics/ Modern chaos diagnostics
    utils/      Plotting style, helpers
figures/        Generated figures (.npz data + .png)
```

## Quick Start

```bash
uv sync                          # install dependencies
PYTHONPATH=src uv run python -m dynachaos.maps.coupled_logistic    # example: regenerate one figure
cd paper && make                 # compile paper
```

Pipeline CLI:

```bash
PYTHONPATH=src uv run python -m dynachaos.cli list
PYTHONPATH=src uv run python -m dynachaos.cli run sec02_circle_map --profile smoke --output-root figures
PYTHONPATH=src uv run python -m dynachaos.cli run all --profile paper --output-root figures
```

Installed package CLI (after `pip install dynachaos`):

```bash
dynachaos list
dynachaos run sec02_circle_map --profile smoke --output-root figures
```

## Unified Tooling

The project is organized so figure generation and visual style are concentrated
in a few central tools instead of ad-hoc script-level settings:

- `dynachaos.cli`:
  - `dynachaos list` for available paper sections
  - `dynachaos run <section|all>` for reproducible pipeline execution
  - `dynachaos style list|preview` for style exploration
- `dynachaos.config`:
  - single source of truth for default paper theme (`DEFAULT_FIGURE_THEME`)
  - runtime override via `DYNACHAOS_THEME`
- `dynachaos.utils.style`:
  - global style setup (`setup`)
  - deterministic color/marker mapping (`color_for`, `marker_for`, `series_style`)
  - layout classes for harmonized typography (`figure_spec`)
  - shared axis/legend polish (`apply_axes_polish`, `finalize_legend`)
- `dynachaos.pipelines`:
  - section registry + runner used by CLI and tests

Figure scripts follow a compute/plot pattern:
- First run computes and caches data as `.npz`
- Subsequent runs load cached data and re-plot
- Delete `.npz` to force recomputation

## Plot Style

All figures use a shared Swiss-inspired style system in
`dynachaos.utils.style`:

- `setup()` applies typography, grid, and color defaults
- `series_style(i)` provides a common color + marker combination for series
- `color_for(i)` / `marker_for(i)` expose deterministic cycling
- `figure_spec("single"|"double"|"grid")` defines class-level sizing + typography
- `apply_axes_polish(ax, kind=...)` and `finalize_legend(ax, kind=...)`
  enforce a consistent text hierarchy across figures
- `available_themes()` lists curated visual directions

Current themes:
- `editorial-grid`: classic Swiss editorial (neutral + red signal accent)
- `zurich-transit`: wayfinding-focused, high-contrast legibility
- `alpine-modern`: softer alpine palette with strong hierarchy
- `bauhaus-pop`: bold poster contrast for exploratory figures

Single-source theme selection for all figure scripts:
- edit `src/dynachaos/config.py` (`DEFAULT_FIGURE_THEME`)
- or override per run with env var `DYNACHAOS_THEME`

CLI helpers:
```bash
dynachaos style list
dynachaos style preview                  # render all previews to figures/style/themes
dynachaos style preview --theme editorial-grid
DYNACHAOS_THEME=zurich-transit dynachaos run sec02_circle_map --profile smoke
```

Harmonized figure regeneration workflow (from cached `.npz`):
```bash
PYTHONPATH=src uv run python -m dynachaos.cli run sec03_transition --profile smoke --output-root figures
PYTHONPATH=src uv run python -m dynachaos.cli run sec04_doubling --profile smoke --output-root figures
PYTHONPATH=src uv run python -m dynachaos.cli run sec05_oscillation --profile smoke --output-root figures
PYTHONPATH=src uv run python -m dynachaos.cli run all --profile smoke --output-root figures
```
