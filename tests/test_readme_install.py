"""README claims about installing Vatic (the "Installing Vatic" section)."""

from __future__ import annotations

import ast
import configparser
import re
import sys
from importlib import metadata

import pytest

from conftest import REPO_ROOT


def _setup_kwargs() -> dict:
    """Statically read the keyword arguments given to setuptools.setup()."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "setup"):
            kwargs = {}
            for kw in node.keywords:
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:  # e.g. packages=find_packages(...)
                    pass
            return kwargs

    raise AssertionError("could not find a setuptools.setup() call in setup.py")


def _readme_python_range(readme_text: str) -> tuple[int, int]:
    match = re.search(r"Python version within 3\.(\d+) through 3\.(\d+)",
                      readme_text)
    assert match, "README no longer states a supported Python version range"
    return int(match.group(1)), int(match.group(2))


def test_python_requires_matches_readme_range(readme_text):
    lo, hi = _readme_python_range(readme_text)
    assert _setup_kwargs()["python_requires"] == f">=3.{lo},<3.{hi + 1}"


def test_ci_matrix_covers_every_readme_python_version(readme_text):
    lo, hi = _readme_python_range(readme_text)
    ci_text = (REPO_ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8")

    matrix = re.search(r"python-version:\s*\[([^\]]*)\]", ci_text)
    assert matrix, "CI workflow has no python-version matrix"
    tested = {v.strip().strip("'\"") for v in matrix.group(1).split(",")}

    assert tested == {f"3.{minor}" for minor in range(lo, hi + 1)}


def test_running_python_is_within_readme_range(readme_text):
    lo, hi = _readme_python_range(readme_text)
    assert sys.version_info[0] == 3
    assert lo <= sys.version_info[1] <= hi


def test_pip_install_registers_vatic_det_command():
    """"Installing Vatic adds the command `vatic-det`"."""
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        scripts = eps.select(group="console_scripts")
    else:  # Python 3.8/3.9
        scripts = eps.get("console_scripts", [])

    vatic_det = [ep for ep in scripts if ep.name == "vatic-det"]
    assert len(vatic_det) == 1, "`vatic-det` console script is not installed"
    assert vatic_det[0].value == "vatic.command_line:run_deterministic"
    assert callable(vatic_det[0].load())


@pytest.mark.parametrize("module", ["pyomo.environ", "egret", "dill",
                                    "numpy", "pandas", "scipy", "matplotlib"])
def test_pip_install_brings_python_dependencies(module):
    """"this will install Vatic's Python package dependencies"."""
    __import__(module)


def test_rts_gmlc_submodule_points_at_gridmod(readme_text):
    """RTS-GMLC is fetched via `git submodule` from GridMod/RTS-GMLC."""
    assert "github.com/GridMod/RTS-GMLC" in readme_text

    gitmodules = configparser.ConfigParser()
    gitmodules.read(REPO_ROOT / ".gitmodules")
    section = 'submodule "vatic/data/grids/RTS-GMLC"'

    assert gitmodules.has_section(section)
    assert gitmodules[section]["path"] == "vatic/data/grids/RTS-GMLC"
    url = re.sub(r"\.git$", "", gitmodules[section]["url"].rstrip("/"))
    assert url == "https://github.com/GridMod/RTS-GMLC"
