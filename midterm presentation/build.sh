#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v tectonic >/dev/null 2>&1; then
  tectonic midterm_draft.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error midterm_draft.tex
else
  echo 'Install Tectonic or a TeX distribution with latexmk and Beamer.' >&2
  exit 1
fi
