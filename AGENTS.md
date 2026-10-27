# Repository Guidelines

## Project Structure & Module Organization

This repository contains a PLOS ONE LaTeX submission for “Coevolutionary balance of resting-state brain networks in autism.” It is a manuscript project; no analysis source code or automated test suite is included.

- `manuscript.tex`: main article, tables, figure captions, and supporting information.
- `Bibliography.bib`: BibTeX reference database.
- `plos2025.bst`: journal bibliography style.
- `manuscript.bbl`: generated bibliography; regenerate after citation changes.
- `Images/`: main and supplementary figures in PDF and PNG formats.
- Root-level PDFs: submission correspondence, including the cover letter and fee-waiver request.

## Build, Test, and Development Commands

Run commands from the repository root with a TeX distribution providing the packages declared in `manuscript.tex`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

These steps compile the article, regenerate references, and resolve citations and cross-references. If installed, `latexmk -pdf manuscript.tex` automates the build sequence. No build wrapper or lint configuration is provided.

## Coding Style & Naming Conventions

Preserve the existing PLOS template, macros, and section organization. Match surrounding whitespace; use two spaces when indenting new multiline LaTeX blocks. Keep prose wrapped into readable source lines and avoid unrelated reformatting.

Use `\label` and `\ref` for cross-references and existing BibTeX keys for citations. Match figure paths exactly, including capitalization and the existing `GSR`/`noGSR` naming distinctions. Add references to `Bibliography.bib` rather than hand-editing generated bibliography entries.

## Testing Guidelines

Validation consists of a successful LaTeX build and visual review of `manuscript.pdf`. Check the log for undefined citations, unresolved references, missing assets, and overfull boxes. Inspect changed tables, captions, equations, figure placement, and supplementary material. No automated coverage requirement is configured.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout, so no established commit convention can be verified. Use concise, imperative messages such as `Fix supplementary figure references`. Keep manuscript, bibliography, and figure updates together when they depend on one another.

Pull requests should describe the affected sections, explain substantive scientific changes, and report build validation. Include PDF excerpts or screenshots for layout changes and link related issues when available. Avoid committing temporary LaTeX build files.
