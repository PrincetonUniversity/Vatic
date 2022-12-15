from datetime import datetime
import multiprocessing
# import sys
import os
from pathlib import Path
#set parent directory to the root directory to use absolute reference to import files from not packages
# sys.path.insert(0, str(Path(__file__).resolve()))

#Profiling Packages
# import cProfile, pstats, infeasibleOrUnbounded
from pstats import SortKey

from vatic.engines import Simulator
from vatic.input import load_input

#relative import does not work when we run the file inside pycharm
#from ..engines import Simulator

#could either be key to access os environ or input dir
input_grid = '/Users/jf3375/PycharmProjects/Vatic_Local/Vatic/vatic_test/input/Vatic_Grids/RTS-GMLC'
start_date = datetime.strptime('2020-02-15', '%Y-%m-%d').date()
num_days = 1
out_dir = '/Users/jf3375/PycharmProjects/Vatic_Local/Vatic/vatic_test/output'
solver = 'gurobi'
solver_args = {'Threads': multiprocessing.cpu_count()-2}
lmps = False
ruc_mipgap = 0.01
reserve_factor = 0.05
prescient_sced_forecasts = False
ruc_prescience_hour = 0
ruc_execution_hour = 16
ruc_every_hours = 24
ruc_horizon = 48
sced_horizon = 4
enforce_sced_shutdown_ramprate = False
no_startup_shutdown_curves = False
output_detail = 1
verbose = 0
output_max_decimals = 4
create_plots = False
renew_costs = None #the renewable cost curve cosntraints
csv = False
init_ruc_file = None


#init_ruc_file: optional argument; the path storing the init_ruc_file




template_data, gen_data, load_data = load_input(input_grid, start_date, num_days)

# pr = cProfile.Profile()
# pr.enable()
Simulator(
    template_data, gen_data, load_data, out_dir=out_dir,
    start_date=start_date, num_days=num_days, solver=solver,
    solver_options=solver_args, run_lmps=lmps, mipgap=ruc_mipgap,
    reserve_factor=reserve_factor,
    prescient_sced_forecasts=prescient_sced_forecasts,
    ruc_prescience_hour=ruc_prescience_hour,
    ruc_execution_hour=ruc_execution_hour,
    ruc_every_hours=ruc_every_hours,
    ruc_horizon=ruc_horizon, sced_horizon=sced_horizon,
    enforce_sced_shutdown_ramprate=enforce_sced_shutdown_ramprate,
    no_startup_shutdown_curves=no_startup_shutdown_curves,
    output_detail=output_detail, init_ruc_file=init_ruc_file,
    verbosity=verbose, output_max_decimals=output_max_decimals,
    create_plots=create_plots, renew_costs=renew_costs,
    save_to_csv = csv
).simulate()
# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumtime', 'calls')
# ps.print_stats(30)
# print(s.getvalue())
#
# with open('vatic_run_test_cprofile.txt', 'w+') as f:
#     f.write(s.getvalue())

