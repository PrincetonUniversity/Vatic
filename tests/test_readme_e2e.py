"""README claims about running real simulations and the output they produce.

These run `vatic-det` as a subprocess on the small RTS-GMLC grid with the free
CBC solver. Settings are tuned for speed (short SCED look-ahead) and simulations
are shared between tests via module-scoped fixtures.
"""

from __future__ import annotations

import bz2
import subprocess
from pathlib import Path

import dill as pickle
import pandas as pd
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("cbc_available")]

DATE = "2020-05-04"  # the README's example date
FAST_ARGS = ["--solver", "cbc", "--sced-horizon", "1", "--ruc-mipgap", "0.05"]

DETAIL0_KEYS = {"hourly_summary"}
DETAIL1_KEYS = DETAIL0_KEYS | {"runtimes", "total_runtime", "ruc_summary",
                               "thermal_detail", "renew_detail"}
DETAIL2_KEYS = DETAIL1_KEYS | {"daily_commits", "bus_detail", "line_detail"}


def vatic_det(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    proc = subprocess.run(["vatic-det", "RTS-GMLC", DATE, "1", *FAST_ARGS,
                           *args], cwd=cwd, capture_output=True, text=True,
                          timeout=1200)
    assert proc.returncode == 0, (
        f"vatic-det failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return proc


def load_output(out_dir: Path) -> dict:
    """Load results exactly as the README shows."""
    with bz2.BZ2File(Path(out_dir, "output.p.gz"), 'r') as f:
        output = pickle.load(f)

    assert isinstance(output, dict)
    return output


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="module")
def default_run(workdir):
    """Pickle output in the working directory (no --out-dir) at the lowest
    detail level, saving the initial RUC for re-use by later runs."""
    cwd = workdir / "default"
    cwd.mkdir()
    ruc_file = workdir / "init_ruc.p"

    assert not ruc_file.exists()
    vatic_det("--output-detail", "0", "--init-ruc-file", str(ruc_file),
              cwd=cwd)

    assert ruc_file.is_file(), "--init-ruc-file was not saved"
    return {"cwd": cwd, "ruc_file": ruc_file,
            "ruc_mtime": ruc_file.stat().st_mtime_ns}


@pytest.fixture(scope="module")
def detail1_run(workdir, default_run):
    """Default output detail via the -o alias, re-using the cached RUC and
    saving the final generator states."""
    out_dir = workdir / "detail1"
    last_conds = workdir / "last_conditions.csv"

    vatic_det("-o", str(out_dir), "--init-ruc-file",
              str(default_run["ruc_file"]),
              "--last-conditions-file", str(last_conds), cwd=workdir)
    return {"out_dir": out_dir, "last_conds": last_conds}


@pytest.fixture(scope="module")
def detailed_csv_run(workdir, default_run, detail1_run):
    """CSV output with maximum detail, plots, and LMPs into a nested output
    directory that does not exist yet, starting from the previous run's final
    generator states."""
    out_dir = workdir / "nested" / "does-not-exist-yet"
    assert not out_dir.parent.exists()

    vatic_det(
        "--csv", "--output-detail", "2", "--create-plots", "--lmps",
        "--out-dir", str(out_dir),
        "--init-ruc-file", str(default_run["ruc_file"]),
        "--init-conditions-file", str(detail1_run["last_conds"]),
        cwd=workdir,
    )
    return {"out_dir": out_dir}


def test_output_defaults_to_invocation_dir_as_compressed_pickle(default_run):
    cwd = default_run["cwd"]
    assert (cwd / "output.p.gz").is_file()
    assert not list(cwd.glob("*.csv")), "--csv was not given"

    output = load_output(cwd)
    summary = output["hourly_summary"]
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 24, "one simulated day should give 24 hours"
    assert summary["Demand"].gt(0).all()


def test_output_detail_0_is_hourly_system_summaries_only(default_run):
    assert set(load_output(default_run["cwd"])) == DETAIL0_KEYS


def test_output_detail_1_adds_generator_level_data(detail1_run):
    output = load_output(detail1_run["out_dir"])
    assert set(output) == DETAIL1_KEYS

    thermal = output["thermal_detail"]
    assert {"Dispatch", "Headroom"} <= set(thermal.columns)
    assert len(thermal) > 0


def test_output_detail_2_adds_bus_and_line_detail(detailed_csv_run):
    csv_names = {p.stem for p in detailed_csv_run["out_dir"].glob("*.csv")}
    # total_runtime is a scalar and is not written as a .csv
    assert csv_names == DETAIL2_KEYS - {"total_runtime"}

    buses = pd.read_csv(detailed_csv_run["out_dir"] / "bus_detail.csv")
    assert {"Mismatch", "LMP"} <= set(buses.columns)
    lines = pd.read_csv(detailed_csv_run["out_dir"] / "line_detail.csv")
    assert "Flow" in lines.columns and len(lines) > 0


def test_csv_option_replaces_pickle(detailed_csv_run):
    assert not (detailed_csv_run["out_dir"] / "output.p.gz").exists()


def test_out_dir_is_created_if_missing(detailed_csv_run):
    assert detailed_csv_run["out_dir"].is_dir()


def test_lmps_are_calculated_when_requested(detailed_csv_run):
    buses = pd.read_csv(detailed_csv_run["out_dir"] / "bus_detail.csv")
    assert buses["LMP"].notna().all()
    assert buses["LMP"].abs().sum() > 0


def test_create_plots_saves_plots(detailed_csv_run):
    plots = [p for p in (detailed_csv_run["out_dir"] / "plots").rglob("*")
             if p.is_file()]
    assert plots, "--create-plots produced no plot files"
    assert all(p.stat().st_size > 0 for p in plots)


def test_init_ruc_file_is_saved_then_reused(default_run, detail1_run,
                                            detailed_csv_run):
    ruc_file = default_run["ruc_file"]
    assert ruc_file.stat().st_size > 0

    # later runs loaded the cached RUC rather than overwriting it
    assert ruc_file.stat().st_mtime_ns == default_run["ruc_mtime"]

    with open(ruc_file, "rb") as f:
        assert pickle.load(f) is not None


def test_last_conditions_file_round_trips(detail1_run, detailed_csv_run):
    conds = pd.read_csv(detail1_run["last_conds"])
    assert list(conds.columns) == ["GEN", "UnitOnT0State", "PowerGeneratedT0"]
    assert len(conds) > 0
    assert (conds["UnitOnT0State"] != 0).all()

    # detailed_csv_run succeeded using it as --init-conditions-file
    assert (detailed_csv_run["out_dir"] / "hourly_summary.csv").is_file()
