"""README claims about the bundled grid datasets and the dates they support."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from conftest import LFS_POINTER_PREFIX, REPO_ROOT
from vatic.compat import hourly_resample
from vatic.data import load_input
from vatic.data.loaders import (_ROOT as INSTALLED_DATA_ROOT, RtsLoader,
                                T7k2030Loader, T7kLoader)

SUPPORTED_GRIDS = ("RTS-GMLC", "Texas-7k", "Texas-7k_2030")

# "For RTS-GMLC, only dates in 2020 are supported, while the Texas grids only
# support dates in 2018."
DOCUMENTED_YEAR = {"RTS-GMLC": 2020, "Texas-7k": 2018, "Texas-7k_2030": 2018}

GRID_DIRS = {"RTS-GMLC": "RTS_Data", "Texas-7k": "TX_Data",
             "Texas-7k_2030": "TX2030_Data"}


def test_readme_lists_the_supported_grids(readme_text):
    assert ('"RTS-GMLC", "Texas-7k", or "Texas-7k_2030."' in readme_text)


@pytest.mark.parametrize("grid", SUPPORTED_GRIDS)
def test_installed_package_ships_real_grid_data(grid):
    """Grid data must be installed with the package and must not be Git LFS
    pointer stubs (which is what you get when cloning without git-lfs)."""
    data_dir = Path(INSTALLED_DATA_ROOT, "grids", grid, GRID_DIRS[grid])
    csv_files = sorted(data_dir.rglob("*.csv"))
    assert csv_files, f"no grid data installed under {data_dir}"

    pointers = [str(f.relative_to(data_dir)) for f in csv_files
                if f.read_bytes()[:len(LFS_POINTER_PREFIX)]
                == LFS_POINTER_PREFIX]
    assert not pointers, (f"{len(pointers)} {grid} files are unfetched Git "
                          f"LFS pointers, e.g. {pointers[:3]}")


INIT_CONDS_COLUMNS = ["GEN", "UnitOnT0State", "PowerGeneratedT0"]


def _readme_cited_init_conditions_file(readme_text: str) -> Path:
    assert ("vatic/data/grids/Texas-7k/TX_Data/FormattedData/.../noTX/"
            "on_time_7.10.csv") in readme_text

    matches = list((REPO_ROOT / "vatic/data/grids/Texas-7k/TX_Data/"
                    "FormattedData").rglob("noTX/on_time_7.10.csv"))
    assert len(matches) == 1, matches
    return matches[0]


def test_readme_init_conditions_example_path_exists(readme_text):
    assert _readme_cited_init_conditions_file(readme_text).is_file()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "README cites a wide-format PLEXOS file as the Texas-7k default initial "
    "conditions; Vatic actually uses initial-state/Texas-7k/on_time_7.10.csv"))
def test_readme_init_conditions_example_is_the_default_used(readme_text):
    cited = _readme_cited_init_conditions_file(readme_text)
    assert Path(T7kLoader(None).init_state_file).resolve() == cited.resolve()


@pytest.mark.parametrize("loader_cls", [RtsLoader, T7kLoader, T7k2030Loader])
def test_default_init_conditions_use_init_conditions_file_format(loader_cls):
    """The bundled default initial conditions have the same format that
    --init-conditions-file and --last-conditions-file use."""
    init_states = pd.read_csv(loader_cls(None).init_state_file)
    assert list(init_states.columns) == INIT_CONDS_COLUMNS
    assert len(init_states) > 0
    assert init_states["GEN"].is_unique


def test_unsupported_grid_is_rejected():
    with pytest.raises(ValueError, match="Unsupported grid"):
        load_input("Texas-7k_2050", dt.date(2018, 7, 15), 1, None)


def _assert_simulatable(grid: str, day: dt.date) -> None:
    """A day is simulatable if Vatic can load NaN-free renewable and load data
    for that day plus the following look-ahead day used by the RUC."""
    template, gen_data, load_data = load_input(grid, day, 1, None)

    assert template["ThermalGenerators"], "grid template has no generators"
    expected_index = pd.date_range(pd.Timestamp(day, tz="utc"),
                                   periods=48, freq=pd.Timedelta(hours=1))

    for lbl, df in (("renewables", gen_data), ("load", load_data)):
        assert df.shape[1] > 0, f"no {lbl} columns"
        assert df.index.equals(expected_index), (
            f"{grid} {lbl} data for {day} covers {df.index.min()} to "
            f"{df.index.max()} ({len(df)} hours), expected 48 hours "
            f"starting at {expected_index[0]}"
        )
        n_missing = int(df.isna().sum().sum())
        assert n_missing == 0, f"{grid} {lbl} data for {day} has " \
                               f"{n_missing} missing values"


@pytest.mark.parametrize("grid", SUPPORTED_GRIDS)
@pytest.mark.parametrize("month", [2, 5, 8, 11])
def test_documented_year_is_supported(grid, month):
    _assert_simulatable(grid, dt.date(DOCUMENTED_YEAR[grid], month, 15))


def _known_gap(reason: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError,
                             reason=f"README/data mismatch: {reason}")


@pytest.mark.parametrize("grid, day", [
    pytest.param("RTS-GMLC", dt.date(2020, 1, 1)),
    pytest.param("Texas-7k", dt.date(2018, 1, 1), marks=_known_gap(
        "Texas-7k data starts 2018-01-02; loading 2018-01-01 silently "
        "returns 2018-01-02 data")),
    pytest.param("Texas-7k_2030", dt.date(2018, 1, 1), marks=_known_gap(
        "Texas-7k_2030 renewables data has gaps on 2018-01-01")),
])
def test_first_day_of_documented_year(grid, day):
    _assert_simulatable(grid, day)


@pytest.mark.parametrize("grid, day", [
    pytest.param("RTS-GMLC", dt.date(2020, 12, 31), marks=_known_gap(
        "the RUC look-ahead needs 2021-01-01, which RTS-GMLC lacks")),
    pytest.param("Texas-7k", dt.date(2018, 12, 31), marks=_known_gap(
        "Texas-7k data ends 2018-12-30")),
    pytest.param("Texas-7k_2030", dt.date(2018, 12, 31), marks=_known_gap(
        "Texas-7k_2030 data ends 2018-12-31 17:00 UTC")),
])
def test_last_day_of_documented_year(grid, day):
    _assert_simulatable(grid, day)


def _renewable_energy_share(loader_cls, year: int) -> float:
    loader = loader_cls(None)
    start = pd.Timestamp(year=year, month=1, day=1, tz="utc")
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz="utc")

    def energy(asset_type: str) -> float:
        hourly = hourly_resample(loader.get_actuals(asset_type)).mean()
        in_year = hourly[(hourly.index >= start) & (hourly.index < end)]
        assert len(in_year) > 24 * 300, f"too little {asset_type} data"
        return float(in_year.sum().sum())

    return (energy("WIND") + energy("PV")) / energy("Load")


def test_texas_renewables_penetration_matches_readme(readme_text):
    """"renewables penetration has been increased to roughly 50%, as opposed
    to 20% in the default 2020 version"."""
    assert "roughly 50%, as opposed to 20%" in readme_text

    share_2020 = _renewable_energy_share(T7kLoader, 2018)
    share_2030 = _renewable_energy_share(T7k2030Loader, 2018)

    assert share_2020 == pytest.approx(0.20, abs=0.05)
    assert share_2030 == pytest.approx(0.50, abs=0.10)
    assert share_2030 > 2 * share_2020


def test_rts_gmlc_loader_reads_submodule_data():
    assert RtsLoader(None).template["ThermalGenerators"]
