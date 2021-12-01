
import argparse
from datetime import datetime
from pathlib import Path
from prescient.simulator import master_options
from .engines import Simulator


def run_deterministic():
    parser = argparse.ArgumentParser(
        'vatic-det',
        description="Simulate a deterministic scenario."
        )

    parser.add_argument('in_dir', type=Path,
                        help="directory containing input datasets")
    parser.add_argument('out_dir', type=Path,
                        help="directory where output will be stored")

    parser.add_argument('run_dates', nargs=2, type=str,
                        #type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                        help="start and end dates for the scenario")

    parser.add_argument('--solver', type=str, default='cbc',
                        help="How to solve RUCs and SCEDs.")
    parser.add_argument('--threads', '-t', type=int, default=1,
                        help="How many compute cores to use for parallelizing "
                             "solver operations.")

    parser.add_argument(
        '--sced-horizon', type=int, default=4, dest='sced_horizon',
        help="Specifies the number of time periods in the look-ahead horizon "
             "for each SCED. Must be at least 1."
        )

    parser.add_argument(
        '--ruc-mipgap', type=float, default=0.01, dest='ruc_mipgap',
        help="Specifies the mipgap for all deterministic RUC solves."
        )

    parser.add_argument(
        '--create-plots', '-p', action='store_true', dest='create_plots',
        help="Create daily stackgraphs?"
        )

    parser.add_argument('--solver-args', nargs='*', dest='solver_args',
                        help="A list of arguments to pass to the solver for "
                             "both RUCs and SCEDs.")

    args = parser.parse_args()
    start_date, end_date = args.run_dates

    if not args.in_dir.exists():
        raise ValueError(
            "Input directory {} does not exist!".format(args.in_dir))

    if args.solver_args is None:
        args.solver_args = list()

    ndays = (datetime.strptime(end_date, "%Y-%m-%d")
             - datetime.strptime(start_date, "%Y-%m-%d")).days

    solver_args = ' '.join(["Threads={}".format(args.threads)]
                           + args.solver_args)

    simulator_args =  [
        '--data-directory', str(args.in_dir), '--simulate-out-of-sample',
        '--run-sced-with-persistent-forecast-errors',
        '--output-directory', str(args.out_dir), '--start-date', start_date,
        '--num-days', str(ndays), '--sced-horizon', str(args.sced_horizon),
        '--traceback', '--output-sced-initial-conditions',
        '--output-sced-demands', '--output-ruc-initial-conditions',
        '--output-ruc-solutions', '--output-solver-logs',
        '--ruc-mipgap', str(args.ruc_mipgap), '--symbolic-solver-labels',
        '--reserve-factor', '0.0', '--deterministic-ruc-solver', args.solver,
        '--sced-solver', args.solver,
        '--deterministic-ruc-solver-options={}'.format(solver_args),
        '--sced-solver-options={}'.format(solver_args),
        ]

    if not args.create_plots:
        simulator_args += ['--disable-stackgraphs']

    Simulator(master_options.construct_options_parser().parse_args(
        simulator_args)).simulate()
