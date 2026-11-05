"""README claims about the `vatic-det` command line interface.

The simulator and grid loading are replaced with mocks so these tests check
only how command line arguments are parsed and handed to the simulator.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from conftest import readme_option_defaults
from vatic import command_line

GRID_ARGS = ["RTS-GMLC", "2020-05-04", "1"]


def run_cli(argv: list[str], monkeypatch) -> mock.MagicMock:
    """Invoke `vatic-det` with mocked loading/simulation; return Simulator."""
    monkeypatch.setattr("sys.argv", ["vatic-det", *argv])

    with mock.patch.object(command_line, "load_input",
                           return_value=("tmpl", "gens", "loads")) as loader, \
            mock.patch.object(command_line, "Simulator") as simulator:
        command_line.run_deterministic()

    loader.assert_called_once()
    simulator.assert_called_once()
    simulator.return_value.simulate.assert_called_once_with()
    return simulator


def sim_kwargs(argv: list[str], monkeypatch) -> dict:
    return run_cli(argv, monkeypatch).call_args.kwargs


def test_positional_arguments_reach_loader_and_simulator(monkeypatch):
    monkeypatch.setattr("sys.argv", ["vatic-det", "Texas-7k", "2018-03-04",
                                     "3"])
    with mock.patch.object(command_line, "load_input",
                           return_value=("t", "g", "l")) as loader, \
            mock.patch.object(command_line, "Simulator") as simulator:
        command_line.run_deterministic()

    loader.assert_called_once_with("Texas-7k", dt.date(2018, 3, 4), 3, None)
    args, kwargs = simulator.call_args
    assert args == ("t", "g", "l")
    assert kwargs["start_date"] == dt.date(2018, 3, 4)
    assert kwargs["num_days"] == 3


@pytest.mark.parametrize("bad_date", ["05-04-2020", "2020/05/04", "20200504",
                                      "2020-13-01"])
def test_start_date_must_be_yyyy_mm_dd(bad_date, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(["RTS-GMLC", bad_date, "1"], monkeypatch)
    assert exc.value.code == 2
    assert "start_date" in capsys.readouterr().err


def test_num_days_must_be_an_integer(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_cli(["RTS-GMLC", "2020-05-04", "1.5"], monkeypatch)
    assert exc.value.code == 2


def test_documented_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    kwargs = sim_kwargs(GRID_ARGS, monkeypatch)

    assert kwargs["solver"] == "cbc"
    assert kwargs["solver_options"] == {"Threads": 1}
    assert kwargs["output_detail"] == 1
    assert kwargs["sced_horizon"] == 4
    assert kwargs["ruc_horizon"] == 48
    assert kwargs["mipgap"] == 0.01
    assert kwargs["reserve_factor"] == 0.05
    assert kwargs["verbosity"] == 0

    # "if not specified it will be saved to the location `vatic-det` was
    # invoked from"
    assert Path(kwargs["out_dir"]) == tmp_path.resolve()

    # flags documented as opt-in are off by default
    assert kwargs["run_lmps"] is False
    assert kwargs["create_plots"] is False
    assert kwargs["save_to_csv"] is False
    assert kwargs["init_ruc_file"] is None
    assert kwargs["last_conditions_file"] is None


def test_defaults_stated_in_readme_match_the_parser(readme_text, monkeypatch):
    """Every "the default value is X" in the README matches the code."""
    stated = readme_option_defaults(readme_text)
    assert set(stated) >= {"--solver", "--threads", "--sced-horizon",
                           "--ruc-horizon", "--ruc-mipgap",
                           "--reserve-factor"}, stated

    kwargs = sim_kwargs(GRID_ARGS, monkeypatch)
    actual = {
        "--solver": kwargs["solver"],
        "--threads": kwargs["solver_options"]["Threads"],
        "--sced-horizon": kwargs["sced_horizon"],
        "--ruc-horizon": kwargs["ruc_horizon"],
        "--ruc-mipgap": kwargs["mipgap"],
        "--reserve-factor": kwargs["reserve_factor"],
    }

    for opt, value in actual.items():
        assert str(value) == stated[opt], f"{opt}: README says " \
                                          f"{stated[opt]}, code uses {value}"

    assert "with `1` (the default)" in readme_text
    assert kwargs["output_detail"] == 1


def test_readme_example_invocation(monkeypatch, tmp_path):
    """vatic-det RTS-GMLC ${date} 1 --solver gurobi --csv --out-dir ${out_dir}
    """
    out_dir = tmp_path / "2020-05-04"
    kwargs = sim_kwargs(["RTS-GMLC", "2020-05-04", "1", "--solver", "gurobi",
                         "--csv", "--out-dir", str(out_dir)], monkeypatch)

    assert kwargs["solver"] == "gurobi"
    assert kwargs["save_to_csv"] is True
    assert kwargs["out_dir"] == out_dir


@pytest.mark.parametrize("flag", ["--out-dir", "-o"])
def test_out_dir_option(flag, monkeypatch, tmp_path):
    kwargs = sim_kwargs([*GRID_ARGS, flag, str(tmp_path / "x")], monkeypatch)
    assert kwargs["out_dir"] == tmp_path / "x"


def test_solver_args_are_parsed_as_python_literals(monkeypatch):
    kwargs = sim_kwargs([*GRID_ARGS, "--solver", "gurobi", "--solver-args",
                         "Cuts=1", "Presolve=1", "Heuristics=0.03"],
                        monkeypatch)
    assert kwargs["solver_options"] == {"Threads": 1, "Cuts": 1,
                                        "Presolve": 1, "Heuristics": 0.03}


@pytest.mark.parametrize("flag", ["--threads", "-t"])
def test_threads_option(flag, monkeypatch):
    kwargs = sim_kwargs([*GRID_ARGS, flag, "0"], monkeypatch)
    assert kwargs["solver_options"]["Threads"] == 0


@pytest.mark.parametrize("level", [0, 1, 2])
def test_output_detail_levels(level, monkeypatch):
    kwargs = sim_kwargs([*GRID_ARGS, "--output-detail", str(level)],
                        monkeypatch)
    assert kwargs["output_detail"] == level


@pytest.mark.parametrize("argv, key, expected", [
    (["--lmps"], "run_lmps", True),
    (["--create-plots"], "create_plots", True),
    (["-p"], "create_plots", True),
    (["--csv"], "save_to_csv", True),
    (["-v"], "verbosity", 1),
    (["--verbose", "--verbose"], "verbosity", 2),
    (["-vvv"], "verbosity", 3),
    (["--sced-horizon", "2"], "sced_horizon", 2),
    (["--ruc-horizon", "24"], "ruc_horizon", 24),
    (["--ruc-mipgap", "0.05"], "mipgap", 0.05),
    (["-g", "0.05"], "mipgap", 0.05),
    (["--reserve-factor", "0.1"], "reserve_factor", 0.1),
    (["-r", "0.1"], "reserve_factor", 0.1),
    (["--load-shed-penalty", "5000"], "load_shed_penalty", 5000.0),
    (["--reserve-shortfall-penalty", "250"], "reserve_shortfall_penalty",
     250.0),
])
def test_documented_options(argv, key, expected, monkeypatch):
    assert sim_kwargs([*GRID_ARGS, *argv], monkeypatch)[key] == expected


def test_file_options_are_passed_through(monkeypatch, tmp_path):
    ruc = tmp_path / "ruc.p"
    init = tmp_path / "init.csv"
    last = tmp_path / "last.csv"

    monkeypatch.setattr("sys.argv", [
        "vatic-det", *GRID_ARGS, "--init-ruc-file", str(ruc),
        "--init-conditions-file", str(init),
        "--last-conditions-file", str(last)])

    with mock.patch.object(command_line, "load_input",
                           return_value=("t", "g", "l")) as loader, \
            mock.patch.object(command_line, "Simulator") as simulator:
        command_line.run_deterministic()

    # initial conditions are applied when the grid is loaded
    assert loader.call_args.args[3] == init
    assert simulator.call_args.kwargs["init_ruc_file"] == ruc
    assert simulator.call_args.kwargs["last_conditions_file"] == last


def test_every_documented_option_exists_in_help(readme_text):
    """Each `--option (-x)` bullet in the README is accepted by vatic-det."""
    documented = re.findall(r"^ - `(--[a-z-]+)(?: \((-[a-z])\))?`",
                            readme_text, flags=re.MULTILINE)
    assert len(documented) >= 15, documented

    help_text = subprocess.run(["vatic-det", "--help"], check=True,
                               capture_output=True, text=True).stdout

    for long_opt, short_opt in documented:
        assert re.search(rf"(^|[\s,]){re.escape(long_opt)}\b", help_text), \
            f"{long_opt} is documented but not accepted"
        if short_opt:
            # argparse lists aliases together, e.g. "--out-dir OUT_DIR, -o
            # OUT_DIR" or "--create-plots, -p"
            alias_lines = [
                line for line in help_text.splitlines()
                if re.search(rf"(^|\s){re.escape(long_opt)}\b", line)
                and re.search(rf"(^|[\s,]){re.escape(short_opt)}\b", line)
            ]
            assert alias_lines, \
                f"{short_opt} is documented as an alias of {long_opt}"


def test_ruc_horizon_must_be_positive():
    """--ruc-horizon "Must be a positive integer" (enforced at sim setup)."""
    from vatic.time_manager import VaticTimeManager

    with pytest.raises(ValueError, match="RUC horizon"):
        VaticTimeManager(dt.date(2020, 5, 4), dt.date(2020, 5, 5),
                         ruc_execution_hour=16, ruc_every_hours=24,
                         ruc_horizon=0, sced_frequency_minutes=60)


_UNENFORCED = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="README states this constraint but vatic-det does not enforce it")


@pytest.mark.parametrize("argv", [
    pytest.param(["RTS-GMLC", "2020-05-04", "0"], id="num_days=0"),
    pytest.param(["RTS-GMLC", "2020-05-04", "-1"], id="num_days=-1"),
    pytest.param([*GRID_ARGS, "--threads", "-1"], id="threads=-1"),
    pytest.param([*GRID_ARGS, "--sced-horizon", "0"], id="sced_horizon=0"),
    pytest.param([*GRID_ARGS, "--output-detail", "-1"],
                 id="output_detail=-1"),
])
@_UNENFORCED
def test_invalid_numeric_options_are_rejected(argv, monkeypatch):
    try:
        run_cli(argv, monkeypatch)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError(f"vatic-det accepted invalid arguments {argv}")
