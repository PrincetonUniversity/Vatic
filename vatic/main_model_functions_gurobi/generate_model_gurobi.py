import time

import gurobipy as gp

import egret.common.lazy_ptdf_utils as lpu

from vatic.models_gurobi import default_params, garver_3bin_vars, \
                                garver_power_vars, \
                                garver_power_avail_vars,  MLR_reserve_vars,\
                                file_non_dispatchable_vars, \
                                pan_guan_gentile_KOW_generation_limits, MLR_generation_limits,\
                                damcikurt_ramping,\
                                KOW_production_costs_tightened, CA_production_costs,\
                                rajan_takriti_UT_DT,\
                                KOW_startup_costs, MLR_startup_costs,\
                                storage_services, ancillary_services, \
                                ptdf_power_flow, \
                                CA_reserve_constraints, MLR_reserve_constraints,\
                                basic_objective


def generate_model(model_name,
            model_data, relax_binaries,
            ptdf_options,
            ptdf_matrix_dict,  objective_hours = None,
            save_model_file = False, file_path_name = '/Users/jf3375/Desktop/Gurobi/output/'):

    #model name = 'UnitComitment'
    use_model = model_data.clone_in_service()

    model = gp.Model(model_name)
    model._model_data = use_model.to_egret()  # _model_data in model is egret object, while model_data is vatic object
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
    if model_name == 'UnitCommitment':
        model = garver_power_avail_vars(model)
    elif model_name == 'EconomicDispatch':
        model = MLR_reserve_vars(model)
    model = file_non_dispatchable_vars(model)

    # Set up constraints
    if model_name == 'UnitCommitment':
        model = pan_guan_gentile_KOW_generation_limits(model)
    elif model_name == 'EconomicDispatch':
        model = MLR_generation_limits(model)
    model = damcikurt_ramping(model)
    if model_name == 'UnitCommitment':
        model = KOW_production_costs_tightened(model)
    elif model_name == 'EconomicDispatch':
        model = CA_production_costs(model)
    model = rajan_takriti_UT_DT(model)
    if model_name == 'UnitCommitment':
        model = KOW_startup_costs(model)
    elif model_name == 'EconomicDispatch':
        model = MLR_startup_costs(model)
    model = storage_services(model)
    model = ancillary_services(model)
    model = ptdf_power_flow(model)
    if model_name == 'UnitCommitment':
        model = CA_reserve_constraints(model)
    elif model_name == 'EconomicDispatch':
        model = MLR_reserve_constraints(model)

    # set up objective
    model = basic_objective(model)

    if objective_hours:
        # Need to Deep Copy of Model Attributes so the Original Attribute does not get removed
        zero_cost_hours = model._TimePeriods.copy()

        for i, t in enumerate(model._TimePeriods):
            if i < objective_hours:
                zero_cost_hours.remove(t)
            else:
                break

        cost_gens = {g for g, _ in model._ProductionCost}
        for t in zero_cost_hours:
            for g in cost_gens:
                model.remove(model._ProductionCostConstr[g, t])
                model._ProductionCost[g, t].lb = 0.
                model._ProductionCost[g, t].ub = 0.

            for g in model._DualFuelGenerators:
                constr = model._DualFuelProductionCost[g, t]
                constr.rhs = 0
                constr.sense = '='

            if model._regulation_service:
                for g in model._AGC_Generators:
                    constr = model._RegulationCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = '='

            if model._spinning_reserve:
                for g in model._ThermalGenerators:
                    constr = model._SpinningReserveCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = '='

            if model._non_spinning_reserve:
                for g in model._ThermalGenerators:
                    constr = model._NonSpinningReserveCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = '='

            if model._supplemental_reserve:
                for g in model.ThermalGenerators:
                    constr = model._SupplementalReserveCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = '='

    # Update the Gurobi Model to incorporate the changes above
    model.update()
    # save gurobi model in a file
    if save_model_file:
        model.write('{}.mps'.format(file_path_name+model_name))
        # more human readable than mps file, but might lose some info
        model.write('{}.lp'.format(file_path_name+model_name))
    return model