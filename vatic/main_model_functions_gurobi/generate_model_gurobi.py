import time

import gurobipy as gp

import egret.common.lazy_ptdf_utils as lpu

from vatic.models_gurobi import default_params, garver_3bin_vars, \
                                garver_power_vars, garver_power_avail_vars, \
                                file_non_dispatchable_vars, \
                                pan_guan_gentile_KOW_generation_limits, \
                                damcikurt_ramping,\
                                KOW_production_costs_tightened, \
                                rajan_takriti_UT_DT,\
                                KOW_startup_costs, \
                                storage_services, ancillary_services, \
                                ptdf_power_flow, \
                                CA_reserve_constraints, \
                                basic_objective


def generate_model(
            model_data, relax_binaries,
            ptdf_options,
            ptdf_matrix_dict,
            save_model_file = False, file_path_name = '/Users/jf3375/Desktop/Gurobi/output/UnitCommitment'):

    use_model = model_data.clone_in_service()

    model = gp.Model('UnitCommitment')
    model._model_data = use_model.to_egret()  # _model_data in model is egret object, while model_data is vatic object
    model._model_data_vatic = model_data
    model._fuel_supply = None
    model._fuel_consumption = None
    model._security_constraints = None

    # Set up attributes under the specific tighten unit commitment model
    ## munge PTDF options if necessary
    model._power_balance = 'ptdf_power_flow'
    if model._power_balance == 'ptdf_power_flow':

        _ptdf_options = lpu.populate_default_ptdf_options(ptdf_options)

        baseMVA = model_data.get_system_attr('baseMVA')
        lpu.check_and_scale_ptdf_options(_ptdf_options, baseMVA)

        model._ptdf_options = _ptdf_options

        if ptdf_matrix_dict is not None:
            model._PTDFs = ptdf_matrix_dict
        else:
            model._PTDFs = {}

    # enforce time 1 ramp rates, relax binaries
    model._enforce_t1_ramp_rates = True
    model._relax_binaries = relax_binaries

    generatemodel_start_time = time.time()
    # Set up parameters
    model = default_params(model, model_data)

    # Set up variables
    model = garver_3bin_vars(model)
    model = garver_power_vars(model)
    model = garver_power_avail_vars(model)
    model = file_non_dispatchable_vars(model)

    # Set up constraints
    model = pan_guan_gentile_KOW_generation_limits(model)
    model = damcikurt_ramping(model)
    model = KOW_production_costs_tightened(model)
    model = rajan_takriti_UT_DT(model)
    model = KOW_startup_costs(model)
    model = storage_services(model)
    model = ancillary_services(model)
    model = ptdf_power_flow(model)
    model = CA_reserve_constraints(model)

    # set up objective
    model = basic_objective(model)

    # save gurobi model in a file
    if save_model_file:
        model.write('{}.mps'.format(file_path_name))
        # more human readable than mps file, but might lose some info
        model.write('{}.lp'.format(file_path_name))
    return model