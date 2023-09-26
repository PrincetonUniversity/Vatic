"""Interfaces for running Vatic simulations from the command line."""

import argparse
from datetime import datetime
from pathlib import Path
from ast import literal_eval

from .data import load_input
from .engines import Simulator


def run_deterministic():
    """The interface for running the command `vatic-det`."""

    # we define here the options governing the behaviour of the simulation
    # that are accessible to the user through the command line interface...
    parser = argparse.ArgumentParser(
        "vatic-det", description="Simulate a deterministic scenario.")

    # ...starting with which power grid dataset should be used
    parser.add_argument("input_grid", type=str,
                        help="the name of or the directory containing the "
                             "input datasets of a power grid")

    # instead of using all the days in the input files we can choose a subset
    parser.add_argument("start_date",
                        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                        help="starting date for the scenario")
    parser.add_argument("num_days", type=int,
                        help="how many days to run the simulation for")

    parser.add_argument("--out-dir", "-o", type=Path, dest="out_dir",
                        help="directory where output will be stored")

    # which solver is used for optimizations within the simulation, and which
    # solver hyper-parameters will be used
    parser.add_argument("--solver", type=str, default="cbc",
                        help="How to solve RUCs and SCEDs.")
    parser.add_argument("--solver-args", nargs='*', dest="solver_args",
                        help="A list of arguments to pass to the solver for "
                             "both RUCs and SCEDs.")
    parser.add_argument("--threads", "-t", type=int, default=1,
                        help="How many compute cores to use for parallelizing "
                             "solver operations.")

    parser.add_argument(
        "--lmps", action='store_true',
        help="solve for locational marginal prices after each SCED?"
        )

    #TODO: separate mipgaps for RUCs and SCEDs
    parser.add_argument(
        "--ruc-mipgap", "-g", type=float, default=0.01, dest="ruc_mipgap",
        help="Specifies the mipgap for all deterministic RUC solves."
        )

    parser.add_argument("--load-shed-penalty",
                        type=float, default=1e4, dest="load_shed_penalty",
                        help="The #/MWh cost of failing to meet "
                             "load demand in each time step.")
    parser.add_argument("--reserve-shortfall-penalty",
                        type=float, default=1e3, dest="reserve_short_penalty",
                        help="The #/MWh cost of failing to meet "
                             "load demand in each time step.")

    parser.add_argument(
        "--reserve-factor", "-r",
        type=float, default=0.05, dest="reserve_factor",
        help="Spinning reserve factor as a constant fraction of demand."
        )

    parser.add_argument("--init-ruc-file", type=Path, dest="init_ruc_file",
                        help="where to save/load the initial RUC from")
    parser.add_argument("--renew-costs", "-c", nargs='*', dest="renew_costs",
                        help="use costs for renewables from input directory")

    parser.add_argument(
        "--init-conditions-file", type=Path, dest="init_conds_file",
        help="where to save/load the initial thermal generator conditions from"
        )
    parser.add_argument(
        "--last-conditions-file", type=Path, dest="last_conds_file",
        help="where to save the final thermal generator states"
        )

    # determines what kinds of output files we create
    parser.add_argument("--output-detail",
                        type=int, default=1, dest="output_detail",
                        help="how much information to save in the output file")
    parser.add_argument("--output-max-decimals",
                        type=int, default=4, dest="output_max_decimals",
                        help="How much precision to use when writing summary "
                             "output files.")

    parser.add_argument("--create-plots", "-p",
                        action='store_true', dest="create_plots",
                        help="Create summary plots of simulation stats?")
    parser.add_argument("--csv", action='store_true',
                        help="save output data to .csv files instead of a "
                             "compressed Python pickle")

    # how are reliability unit commitments run: how often, at which time, for
    # how long, and with (or without) what information?
    parser.add_argument("--ruc-every-hours",
                        type=int, dest="ruc_every_hours", default=24,
                        help="Specifies when the the RUC process is executed. "
                             "Negative values indicate time before horizon, "
                             "positive after.")
    parser.add_argument("--ruc-execution-hour",
                        type=int, dest="ruc_execution_hour", default=16,
                        help="Specifies when the RUC process is executed. "
                             "Negative values indicate time before horizon, "
                             "positive after.")
    parser.add_argument("--ruc-horizon",
                        type=int, dest="ruc_horizon", default=48,
                        help="The number of hours for which the reliability "
                             "unit commitment is executed. Must be <= 48 "
                             "hours and >= --ruc-every-hours.")
    parser.add_argument("--ruc-prescience-hour",
                        type=int, dest="ruc_prescience_hour", default=0,
                        help="Hour before which linear blending of forecast "
                             "and actuals takes place when running RUCs."
                             "The default value of 0 indicates we always take "
                             "the forecast.")

    # how are security-constrained economic dispatches run: for how long and
    # with what information?
    parser.add_argument(
        "--sced-horizon", type=int, default=4, dest="sced_horizon",
        help="Specifies the number of time periods in the look-ahead horizon "
             "for each SCED. Must be at least 1."
        )
    parser.add_argument("--prescient-sced-forecasts",
                        action='store_true', dest="prescient_sced_forecasts",
                        help="make forecasts used by SCEDs equal to actuals")

    parser.add_argument("--lmp-shortfall-costs",
                        action='store_true', dest="lmp_shortfall_costs",
                        help="take reserve shortfall costs into "
                             "account when calculating LMPs")

    # how generator startup and shutdown constraints are calculated and used
    parser.add_argument("--enforce-sced-shutdown-ramprate",
                        action='store_true',
                        dest="enforce_sced_shutdown_ramprate",
                        help="Enforces shutdown ramp-rate constraints in the "
                             "SCED. Enabling this options requires a long "
                             "SCED look-ahead (at least an hour) to ensure "
                             "the shutdown ramp-rate constraints can "
                             "be satisfied.")
    parser.add_argument("--no-startup-shutdown-curves",
                        action='store_true', dest="no_startup_shutdown_curves",
                        help="For thermal generators, do not infer "
                             "startup/shutdown ramping curves when "
                             "starting-up and shutting-down.")

    parser.add_argument("--verbose", "-v", action='count', default=0,
                        help="how much info to print about the ongoing state "
                             "of the simulator and its solvers")

    args = parser.parse_args()
    template_data, gen_data, load_data = load_input(
        args.input_grid, args.start_date, args.num_days, args.init_conds_file)

    # output is placed in this code repo directory if output path is not given
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = Path().absolute()

    solver_args = {'Threads': args.threads}
    if args.solver_args:
        for arg_str in args.solver_args:
            arg_name, arg_val = arg_str.split("=")
            solver_args[arg_name] = literal_eval(arg_val)

    if (args.renew_costs is not None and len(args.renew_costs) == 1
            and isinstance(args.renew_costs[0], (str, Path))):
        renew_costs = args.renew_costs[0]
    else:
        renew_costs = args.renew_costs

    Simulator(
        template_data, gen_data, load_data, out_dir=out_dir,
        start_date=args.start_date, num_days=args.num_days, solver=args.solver,
        solver_options=solver_args, run_lmps=args.lmps, mipgap=args.ruc_mipgap,
        load_shed_penalty=args.load_shed_penalty,
        reserve_shortfall_penalty=args.reserve_short_penalty,
        reserve_factor=args.reserve_factor,
        prescient_sced_forecasts=args.prescient_sced_forecasts,
        ruc_prescience_hour=args.ruc_prescience_hour,
        ruc_execution_hour=args.ruc_execution_hour,
        ruc_every_hours=args.ruc_every_hours,
        ruc_horizon=args.ruc_horizon, sced_horizon=args.sced_horizon,
        lmp_shortfall_costs=args.lmp_shortfall_costs,
        enforce_sced_shutdown_ramprate=args.enforce_sced_shutdown_ramprate,
        no_startup_shutdown_curves=args.no_startup_shutdown_curves,
        output_detail=args.output_detail, init_ruc_file=args.init_ruc_file,
        verbosity=args.verbose, output_max_decimals=args.output_max_decimals,
        create_plots=args.create_plots, renew_costs=renew_costs,
        save_to_csv=args.csv, last_conditions_file=args.last_conds_file
        ).simulate()