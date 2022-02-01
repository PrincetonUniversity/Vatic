
import argparse
from datetime import datetime
from pathlib import Path
from prescient.simulator import master_options
from .engines import Simulator, AllocationSimulator


def run_deterministic():
    parser = argparse.ArgumentParser(
        'vatic-det',
        description="Simulate a deterministic scenario."
        )

    parser.add_argument('in_dir', type=Path,
                        help="directory containing input datasets")
    parser.add_argument('run_dates', nargs=2, type=str,
                        #type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                        help="start and end dates for the scenario")

    parser.add_argument('--out_dir', '-o', type=Path,
                        help="directory where output will be stored")
    parser.add_argument('--light-output', '-l',
                        action='store_true', dest='light_output',
                        help="don't create hourly asset digests")
    parser.add_argument('--renew-costs', '-c',
                        action='store_true', dest='renew_costs',
                        help="use costs for renewables from input directory")

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
        '--reserve-factor', '-r',
        type=float, default=0.05, dest='reserve_factor',
        help="Spinning reserve factor as a constant fraction of demand."
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

    parser.add_argument('--init-ruc-file', dest='init_ruc_file',
                        help='where to save/load the initial RUC from')
    parser.add_argument('--save-init-ruc', dest='save_init_ruc',
                        nargs='?', const=True,
                        help="whether to save the initial solved RUC to file")

    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help="how much info to print about the ongoing state "
                             "of the simulator and its solvers")

    args = parser.parse_args()
    start_date, end_date = args.run_dates

    if not args.in_dir.exists():
        raise ValueError(
            "Input directory {} does not exist!".format(args.in_dir))

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = args.in_dir

    if args.solver_args is None:
        args.solver_args = list()

    ndays = (datetime.strptime(end_date, "%Y-%m-%d")
             - datetime.strptime(start_date, "%Y-%m-%d")).days

    solver_args = ' '.join(["Threads={}".format(args.threads)]
                           + args.solver_args)

    # Prescient options we want to have no matter what
    simulator_args = [
        '--simulate-out-of-sample',
        '--run-sced-with-persistent-forecast-errors',
        '--symbolic-solver-labels',
        '--traceback',
        ]

    if args.verbose > 1:
        simulator_args += ['--output-solver-logs']

    simulator_args += [
        '--start-date', start_date, '--num-days', str(ndays),
        '--sced-horizon', str(args.sced_horizon),
        '--ruc-mipgap', str(args.ruc_mipgap),
        '--reserve-factor', str(args.reserve_factor),
        '--deterministic-ruc-solver', args.solver,
        '--sced-solver', args.solver,
        '--deterministic-ruc-solver-options={}'.format(solver_args),
        '--sced-solver-options={}'.format(solver_args),
        ]

    if not args.create_plots:
        simulator_args += ['--disable-stackgraphs']

    parsed_args = master_options.construct_options_parser().parse_args(
        simulator_args)

    if args.renew_costs:
        SimCls = AllocationSimulator
    else:
        SimCls = Simulator

    SimCls(in_dir=args.in_dir, out_dir=out_dir, light_output=args.light_output,
           init_ruc_file=args.init_ruc_file, save_init_ruc=args.save_init_ruc,
           verbosity=args.verbose, prescient_options=parsed_args).simulate()
