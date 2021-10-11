
import argparse
import os
from datetime import datetime
from pathlib import Path
import shutil

from prescient.scripts import populator
from prescient.simulator import master_options
from .engines import Simulator


def run_deterministic():
    parser = argparse.ArgumentParser(
        'vatic-det',
        description="Create and simulate a deterministic scenario."
        )

    parser.add_argument('in_dir', type=Path,
                        help="directory containing input datasets")
    parser.add_argument('out_dir', type=Path,
                        help="directory where output will be stored")

    parser.add_argument('run_dates', nargs=2,
                        type=str,
                        #type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                        help="start and end dates for the scenario")

    parser.add_argument('--threads', '-t', type=int, default=1,
                        help="How many compute cores to use for parallelizing "
                             "solver operations.")

    parser.add_argument(
        '--sced_horizon', help="Specifies the number of time periods "
                               "in the look-ahead horizon for each SCED. "
                               "Must be at least 1.",
        type=int, default=4
        )

    parser.add_argument(
        '--ruc_mipgap',
        help="Specifies the mipgap for all deterministic RUC solves.",
        type=float, default=0.01
        )

    parser.add_argument('--solver', help="How to solve RUCs and SCEDs.",
                        type=str, default='cbc')

    args = parser.parse_args()
    start_date, end_date = args.run_dates

    if not args.in_dir.exists():
        raise ValueError(
            "Input directory {} does not exist!".format(args.in_dir))

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    shutil.copytree(args.in_dir, args.out_dir)
    os.chdir(args.out_dir)

    populator.main(populator_args=[
        '--start-date', start_date, '--end-date', end_date,
        '--sources-file', 'sources_with_network.txt',
        '--output-directory', 'scenarios',
        '--scenario-creator-options-file',
        'deterministic_scenario_creator_with_network.txt',
        '--traceback'
        ])

    ndays = (datetime.strptime(end_date, "%Y-%m-%d")
             - datetime.strptime(start_date, "%Y-%m-%d")).days

    Simulator(master_options.construct_options_parser().parse_args(
        ['--data-directory', 'scenarios', '--simulate-out-of-sample',
         '--run-sced-with-persistent-forecast-errors',
         '--output-directory', 'output', '--start-date', start_date,
         '--num-days', str(ndays), '--sced-horizon', str(args.sced_horizon),
         '--traceback', '--output-sced-initial-conditions',
         '--output-sced-demands', '--output-ruc-initial-conditions',
         '--output-ruc-solutions', '--output-solver-logs',
         '--ruc-mipgap', str(args.ruc_mipgap), '--symbolic-solver-labels',
         '--reserve-factor', '0.0', '--deterministic-ruc-solver', args.solver,
         '--sced-solver', args.solver,
         '--deterministic-ruc-solver-options=Threads={}'.format(args.threads),
         '--sced-solver-options=Threads={}'.format(args.threads), ]
        )).simulate()
