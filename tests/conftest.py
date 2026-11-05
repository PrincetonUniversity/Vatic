"""Shared fixtures for the README-claims test suite.

These tests check that what README.md tells users about installing and
running Vatic is actually true of the code and data in this repository.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-e2e", action="store_true", default=False,
        help="explicitly skip end-to-end simulations that need a MILP solver",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "e2e: runs real vatic-det simulations with the CBC solver")


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    skip_requested = (config.getoption("--skip-e2e")
                      or os.environ.get("VATIC_SKIP_E2E") == "1")

    if skip_requested:
        marker = pytest.mark.skip(reason="end-to-end tests explicitly skipped")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(marker)


@pytest.fixture(scope="session")
def readme_text() -> str:
    text = README.read_text(encoding="utf-8")
    assert text.strip(), "README.md is empty"
    return text


@pytest.fixture(scope="session")
def cbc_available() -> None:
    """End-to-end tests need CBC; a missing solver is a failure, not a skip.

    Use `--skip-e2e` or VATIC_SKIP_E2E=1 to opt out deliberately.
    """
    if shutil.which("cbc") is None:
        pytest.fail("the `cbc` solver is not on PATH; install it (e.g. "
                    "`apt-get install coinor-cbc` or `conda install "
                    "-c conda-forge coincbc`) or pass --skip-e2e")


def readme_option_defaults(readme_text: str) -> dict[str, str]:
    """Map each documented `--option` to the default value the README states.

    Matches phrases such as "The default value is 4" or "the default value is
    0.05" inside each option's bullet point.
    """
    defaults = {}
    bullets = re.split(r"\n - ", readme_text)

    for bullet in bullets:
        opt_match = re.match(r"`(--[a-z-]+)", bullet)
        if not opt_match:
            continue

        default_match = re.search(
            r"[Tt]he default(?: value)? is\s+[\[`]?([0-9][0-9.]*|[a-z]+)",
            bullet,
        )
        if default_match:
            defaults[opt_match.group(1)] = default_match.group(1).rstrip(".")

    return defaults
