import os
import sys
import logging
from datetime import datetime
import time
import multiprocessing
import pandas as pd
import gurobipy as gp

from vatic.engines import Simulator
from vatic.data import load_input
from vatic.models_gurobi import default_params, garver_3bin_vars, \
                                garver_power_vars, MLR_reserve_vars, \
                                file_non_dispatchable_vars, \
                                MLR_generation_limits, \
                                damcikurt_ramping,\
                                CA_production_costs, \
                                rajan_takriti_UT_DT,\
                                MLR_startup_costs, \
                                storage_services, ancillary_services, \
                                ptdf_power_flow, \
                                MLR_reserve_constraints, \
                                basic_objective

import egret.common.lazy_ptdf_utils as lpu

from vatic.main_model_functions_gurobi.generate_model_gurobi import generate_model
from vatic.main_model_functions_gurobi.solve_model_gurobi import solve_model

sys.path.extend(['/Users/jf3375/PycharmProjects/Vatic'])

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

# Run RUC to get simulation_states: actual, forecast, commits, power_generated...
simulator.initialize_oracle()

ptdf_options = {'lp_iteration_limit': 0,  'pre_lp_iteration_limit': 0}
ptdf_matrix_dict = None
relax_binaries = False
hours_in_objective = 1


sim_state_for_ruc = None
first_step = simulator._time_manager.get_first_timestep()

sced_model_data = simulator._data_provider.create_sced_instance(
    simulator._simulation_state, sced_horizon=sced_horizon)
simulator._ptdf_manager.mark_active(sced_model_data)
ptdf_options = simulator._ptdf_manager.sced_ptdf_options


# How to get shadow price
model_fixed = generate_model(model_name='EconomicDispatch',
                       model_data=sced_model_data, relax_binaries=True,
                       ptdf_options=ptdf_options,
                       ptdf_matrix_dict=simulator._ptdf_manager.PTDF_matrix_dict,
                       objective_hours=hours_in_objective,
                       save_model_file=True,
                       file_path_name='/Users/jf3375/Desktop/Gurobi/output/')

model_fixed.optimize()
model_fixed.getConstrs()[0].Pi

b = model_fixed._TransmissionBlock[1]

model_fixed.getConstrByName('eq_p_balance_at_period1').Pi