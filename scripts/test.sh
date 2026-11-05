#!/usr/bin/env zsh
# Build the test container and run the full test suite (lint, type checks,
# README-claims tests including end-to-end CBC simulations).
#
# Requires the Git LFS grid data and the RTS-GMLC submodule to be present:
#   git lfs pull && git submodule update --init
#
# Usage: scripts/test.sh [pytest args...]   e.g. scripts/test.sh -k cli
set -euo pipefail

cd "${0:A:h}/.."

if head -c 40 vatic/data/grids/Texas-7k/TX_Data/SourceData/gen.csv \
        | grep -q "git-lfs"; then
  print -u2 "Grid data are Git LFS pointers; run 'git lfs pull' first."
  exit 1
fi
if [[ ! -d vatic/data/grids/RTS-GMLC/RTS_Data ]]; then
  print -u2 "RTS-GMLC submodule missing; run 'git submodule update --init'."
  exit 1
fi

docker-compose -f docker-compose.test.yml build
docker-compose -f docker-compose.test.yml run --rm tests sh -c '
  set -e
  cd /src
  ruff check tests
  mypy --ignore-missing-imports --follow-imports=skip --python-version 3.8 tests
  cd /work
  exec python -m pytest /src/tests -v "$@"
' sh "$@"
