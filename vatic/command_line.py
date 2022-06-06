"""Interfaces for running Vatic simulations from the command line."""

import argparse
from datetime import datetime
from pathlib import Path
from ast import literal_eval

from .input.loaders import load_input
from .engines import Simulator, AllocationSimulator, AutoAllocationSimulator


def run_deterministic():
    """The interface for running the command `vatic-det`."""

    # we define here the options governing the behaviour of the simulation
    # that are accessible to the user through the command line interface...
    parser = argparse.ArgumentParser(
        'vatic-det', description="Simulate a deterministic scenario.")

    # ...starting with where input and output files are stored
    parser.add_argument('in_dir', type=Path,
                        help="directory containing input datasets")
    parser.add_argument('--out_dir', '-o', type=Path,
                        help="directory where output will be stored")

    # instead of using all the days in the input files we can choose a subset
    parser.add_argument('--start-date', '-s',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                        dest='start_date',
                        help="starting date for the scenario")
    parser.add_argument('--num-days', '-d', type=int, dest='num_days',
                        help="how many days to run the simulation for")

    # which solver is used for optimizations within the simulation, and which
    # solver hyper-parameters will be used
    parser.add_argument('--solver', type=str, default='cbc',
                        help="How to solve RUCs and SCEDs.")
    parser.add_argument('--solver-args', nargs='*', dest='solver_args',
                        help="A list of arguments to pass to the solver for "
                             "both RUCs and SCEDs.")
    parser.add_argument('--threads', '-t', type=int, default=1,
                        help="How many compute cores to use for parallelizing "
                             "solver operations.")

    parser.add_argument(
        '--ruc-mipgap', '-g', type=float, default=0.01, dest='ruc_mipgap',
        help="Specifies the mipgap for all deterministic RUC solves."
        )
    parser.add_argument(
        '--reserve-factor', '-r',
        type=float, default=0.05, dest='reserve_factor',
        help="Spinning reserve factor as a constant fraction of demand."
        )

    # determines what kinds of output files we create
    parser.add_argument('--light-output', '-l',
                        action='store_true', dest='light_output',
                        help="don't create hourly asset digests")
    parser.add_argument('--create-plots', '-p',
                        action='store_true', dest='create_plots',
                        help="Create summary plots of simulation stats?")
    parser.add_argument('--output-max-decimals',
                        type=int, default=4, dest='output_max_decimals',
                        help="How much precision to use when writing summary "
                             "output files.")

    parser.add_argument('--init-ruc-file', type=Path, dest='init_ruc_file',
                        help='where to save/load the initial RUC from')

    parser.add_argument('--renew-costs', '-c', nargs='*',
                        default=False, dest='renew_costs',
                        help="use costs for renewables from input directory")

    # how are reliability unit commitments run: how often, at which time, for
    # how long, and with (or without) what information?
    parser.add_argument('--ruc-every-hours',
                        type=int, dest='ruc_every_hours', default=24,
                        help="Specifies when the the RUC process is executed. "
                             "Negative values indicate time before horizon, "
                             "positive after.")
    parser.add_argument('--ruc-execution-hour',
                        type=int, dest='ruc_execution_hour', default=16,
                        help="Specifies when the RUC process is executed. "
                             "Negative values indicate time before horizon, "
                             "positive after.")
    parser.add_argument('--ruc-horizon',
                        type=int, dest='ruc_horizon', default=48,
                        help="The number of hours for which the reliability "
                             "unit commitment is executed. Must be <= 48 "
                             "hours and >= --ruc-every-hours.")
    parser.add_argument('--ruc-prescience-hour',
                        type=int, dest='ruc_prescience_hour', default=0,
                        help="Hour before which linear blending of forecast "
                             "and actuals takes place when running RUCs."
                             "The default value of 0 indicates we always take "
                             "the forecast.")

    # how are security-constrained economic dispatches run: for how long and
    # with what information?
    parser.add_argument(
        '--sced-horizon', type=int, default=4, dest='sced_horizon',
        help="Specifies the number of time periods in the look-ahead horizon "
             "for each SCED. Must be at least 1."
        )
    parser.add_argument('--prescient-sced-forecasts',
                        action='store_true', dest='prescient_sced_forecasts',
                        help="make forecasts used by SCEDs equal to actuals")

    # how generator startup and shutdown constraints are calculated and used
    parser.add_argument('--enforce-sced-shutdown-ramprate',
                        action='store_true',
                        dest="enforce_sced_shutdown_ramprate",
                        help="Enforces shutdown ramp-rate constraints in the "
                             "SCED. Enabling this options requires a long "
                             "SCED look-ahead (at least an hour) to ensure "
                             "the shutdown ramp-rate constraints can "
                             "be satisfied.")
    parser.add_argument('--no-startup-shutdown-curves',
                        action='store_true', dest="no_startup_shutdown_curves",
                        help="For thermal generators, do not infer "
                             "startup/shutdown ramping curves when "
                             "starting-up and shutting-down.")

    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help="how much info to print about the ongoing state "
                             "of the simulator and its solvers")

    args = parser.parse_args()
    template_data, gen_data, load_data = load_input(
        args.in_dir, args.start_date, args.num_days)

    # output is placed in input directory if output directory is not given
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = args.in_dir

    solver_args = {'Threads': args.threads}
    if args.solver_args:
        for arg_str in args.solver_args:
            arg_name, arg_val = arg_str.split('=')
            solver_args[arg_name] = literal_eval(arg_val)

    if args.renew_costs is False:
        Simulator(template_data, gen_data, load_data, out_dir=out_dir,
                  start_date=args.start_date, num_days=args.num_days,
                  solver=args.solver, solver_options=solver_args,
                  mipgap=args.ruc_mipgap, reserve_factor=args.reserve_factor,
                  prescient_sced_forecasts=args.prescient_sced_forecasts,
                  ruc_prescience_hour=args.ruc_prescience_hour,
                  ruc_execution_hour=args.ruc_execution_hour,
                  ruc_every_hours=args.ruc_every_hours,
                  ruc_horizon=args.ruc_horizon,
                  sced_horizon=args.sced_horizon,
                  enforce_sced_shutdown_ramprate=args.enforce_sced_shutdown_ramprate,
                  no_startup_shutdown_curves=args.no_startup_shutdown_curves,
                  light_output=args.light_output,
                  init_ruc_file=args.init_ruc_file, verbosity=args.verbose,
                  output_max_decimals=args.output_max_decimals,
                  create_plots=args.create_plots).simulate()

    elif not args.renew_costs:
        AllocationSimulator(in_dir=args.in_dir, out_dir=out_dir,
                            start_date=args.start_date, num_days=args.num_days,
                            light_output=args.light_output,
                            init_ruc_file=args.init_ruc_file,
                            save_init_ruc=args.save_init_ruc,
                            verbosity=args.verbose).simulate()

    else:
        AutoAllocationSimulator(cost_vals=args.renew_costs, in_dir=args.in_dir,
                                out_dir=out_dir,
                                start_date=args.start_date,
                                num_days=args.num_days,
                                light_output=args.light_output,
                                init_ruc_file=args.init_ruc_file,
                                save_init_ruc=args.save_init_ruc,
                                verbosity=args.verbose).simulate()
