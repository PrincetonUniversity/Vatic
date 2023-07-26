'''
Get optimization models objectives out of vatic-gurobipy
with Gurobi Solver
'''

from datetime import datetime
import multiprocessing
import gurobipy as gp

from vatic.engines import Simulator
from vatic.data import load_input

from vatic.models_gurobi.src.main_model_functions_gurobi.generate_model_gurobi import generate_model


model_old_path = '/Users/jf3375/Desktop/model_outputs/ruc_old_20200102.lp'

input_grid = 'RTS-GMLC'
start_date = datetime.strptime('2020-01-02', '%Y-%m-%d').date()

# input_grid = 'Texas-7k'
# start_date = datetime.strptime('2018-01-10', '%Y-%m-%d').date()

num_days = 1
out_dir = '/Users/jf3375/Desktop/Vatic_Run/outputs'
solver = 'gurobi'
solver_args = {'Threads': multiprocessing.cpu_count()-1}
lmps = True
ruc_mipgap = 0
reserve_factor = 0.15
load_shed_penalty = 1e4
reserve_shortfall_penalty = 1e3
lmp_shortfall_costs = False
prescient_sced_forecasts = False
ruc_prescience_hour = 0
ruc_execution_hour = 16
ruc_every_hours = 24
ruc_horizon = 48
sced_horizon = 2
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
template_data, gen_data, load_data = load_input(input_grid, start_date, num_days, None)

# pr = cProfile.Profile()
# pr.enable()
simulator = Simulator(
    template_data, gen_data, load_data, out_dir=out_dir,
    last_conditions_file=None,
    start_date=start_date, num_days=num_days, solver=solver,
    solver_options=solver_args, run_lmps=lmps, mipgap=ruc_mipgap,
    reserve_factor=reserve_factor,
    load_shed_penalty=load_shed_penalty,
    reserve_shortfall_penalty=reserve_shortfall_penalty,
    lmp_shortfall_costs=lmp_shortfall_costs,
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
)

ptdf_options = {'rel_ptdf_tol': 1e-06, 'abs_ptdf_tol': 1e-10, 'abs_flow_tol': 0.001, 'rel_flow_tol': 1e-05, 'lazy_rel_flow_tol': -0.01, 'iteration_limit': 100000, 'lp_iteration_limit': 100, 'max_violations_per_iteration': 5, 'lazy': True, 'branch_kv_threshold': None, 'kv_threshold_type': 'one', 'pre_lp_iteration_limit': 100, 'active_flow_tol': 50.0, 'lp_cleanup_phase': True}
ptdf_matrix_dict = None
relax_binaries = False

sim_state_for_ruc = None
first_step = simulator._time_manager.get_first_timestep()
model_data = simulator._data_provider.create_deterministic_ruc(
    first_step, sim_state_for_ruc)


ruc_new = generate_model('UnitCommitment',
            model_data, relax_binaries,
            ptdf_options,
            ptdf_matrix_dict)

ruc_new.Params.OutputFlag = 0
ruc_new.Params.Threads = multiprocessing.cpu_count()-1
ruc_new.Params.MIPGap = ruc_mipgap
ruc_new.optimize()

# solve old with gurobi solver
ruc_old = gp.read(model_old_path)
ruc_old.Params.OutputFlag = 0
ruc_old.Params.Threads = multiprocessing.cpu_count()-1
ruc_old.Params.MIPGap = ruc_mipgap

ruc_old.optimize()

print(ruc_new.ObjVal, ruc_old.ObjVal)
print('The difference in ObjVal between two versions:', ruc_new.ObjVal - ruc_old.ObjVal)



