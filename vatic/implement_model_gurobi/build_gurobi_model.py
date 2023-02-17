import os
import logging
from datetime import datetime

import multiprocessing
import gurobipy as gp

from vatic.engines import Simulator
from vatic.data import load_input
from vatic.models_gurobi import default_params, garver_3bin_vars, \
                                garver_power_vars, garver_power_avail_vars, \
                                file_non_dispatchable_vars, \
                                pan_guan_gentile_KOW_generation_limits, \
                                damcikurt_ramping,\
                                KOW_production_costs_tightened, \
                                rajan_takriti_UT_DT,\
                                KOW_startup_costs, \
                                storage_services, ancillary_services

component_name = 'data_loader'

logger = logging.getLogger('pyomo.core')

#relative import does not work when we run the file inside pycharm
#from ..engines import Simulator

#could either be key to access os environ or input dir
input_grid = 'RTS-GMLC'
start_date = datetime.strptime('2020-02-15', '%Y-%m-%d').date()
num_days = 1
out_dir = '/Users/jf3375/Desktop/Vatic_Run/outputs'
last_condition_file = '/Users/jf3375/Desktop/Vatic_Run/outputs/last_condition.csv'
solver = 'gurobi'
solver_args = {'Threads': multiprocessing.cpu_count()-1}
lmps = False
ruc_mipgap = 0.01
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
template_data, gen_data, load_data = load_input(input_grid, start_date, num_days)

# pr = cProfile.Profile()
# pr.enable()
simulator = Simulator(
    template_data, gen_data, load_data, out_dir=out_dir,
    last_conditions_file=last_condition_file,
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

sim_state_for_ruc = None
first_step = simulator._time_manager.get_first_timestep()
model_data = simulator._data_provider.create_deterministic_ruc(
    first_step, sim_state_for_ruc)
use_model = model_data.clone_in_service()

model = gp.Model('UnitCommitment')
model._model_data = use_model.to_egret()  #_model_data in model is egret object, while model_data is vatic object
# Set up attributes under the specific tighten unit commitment model
model._enforce_t1_ramp_rates = True

# Run the function to build variables and constraints
model = default_params(model, model_data)
model = garver_3bin_vars(model)
model = garver_power_vars(model)
model = garver_power_avail_vars(model)
model = file_non_dispatchable_vars(model)
model = pan_guan_gentile_KOW_generation_limits(model)
model = damcikurt_ramping(model)
model = KOW_production_costs_tightened(model)
model = rajan_takriti_UT_DT(model)
model = KOW_startup_costs(model)
model = storage_services(model)
model = ancillary_services(model)

# save gurobi model in a file
os.chdir('/Users/jf3375/Desktop/Gurobi/output/')
model.write('UnitCommitment.mps')
# more human readable than mps file, but might lose some info
model.write('UnitCommitment.lp')