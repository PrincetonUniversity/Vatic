"""
Construct unit commitment or economic dispatch models
"""

from vatic.model_data import VaticModelData
import vatic.models_gurobi.src.construct_models_gurobi.lazy_ptdf_utils_part_gurobi as lpu
from vatic.models_gurobi import (
    default_params,
    garver_3bin_vars,
    garver_power_vars,
    garver_power_avail_vars,
    MLR_reserve_vars,
    file_non_dispatchable_vars,
    pan_guan_gentile_KOW_generation_limits,
    MLR_generation_limits,
    damcikurt_ramping,
    KOW_production_costs_tightened,
    CA_production_costs,
    rajan_takriti_UT_DT,
    KOW_startup_costs,
    MLR_startup_costs,
    storage_services,
    ancillary_services,
    ptdf_power_flow,
    CA_reserve_constraints,
    MLR_reserve_constraints,
    basic_objective,
)

import gurobipy as gp
from typing import Optional


def generate_model(
        model_name: str,
        model_data: VaticModelData,
        relax_binaries: bool,
        ptdf_options: dict,
        ptdf_matrix_dict: dict,
        objective_hours: Optional[int] = None,
        save_model_file_path: str = "",
) -> gp.Model:
    """
    Generate the gurobi model of unit commitment or economic dispatch

    Parameters
    ----------
    model_name: str
        UnitCommitment or EconomicDispatch
    model_data: VaticModelData
        Data storing all the grid informaiton
    relax_binaries: bool
        True to relax all binary variables in the model
    ptdf_options: dict
        Dictonary of options for ptdf transmission model
    ptdf_matrix_dict: dict
        Dictionary of ptdf_utils.PTDFMatrix objects for use in
        model construction
    objective_hours: int
        Number of hours looking forward when planning
    save_model_file_path: str
        The file path where the output model lp file would be saved

    Returns
    -------
    gurobipy.Model
    """

    # Initiate gurobi model
    model = gp.Model(model_name)

    # Add necessary attributes to the model
    model._model_data = model_data
    model._fuel_supply = None
    model._fuel_consumption = None
    model._security_constraints = None

    # Enforce ptdf options
    model._power_balance = "ptdf_power_flow"
    if model._power_balance == "ptdf_power_flow":
        _ptdf_options = lpu.populate_default_ptdf_options(ptdf_options)

        baseMVA = model_data.get_system_attr("baseMVA")
        lpu.check_and_scale_ptdf_options(_ptdf_options, baseMVA)

        model._ptdf_options = _ptdf_options

        if ptdf_matrix_dict is not None:
            model._PTDFs = ptdf_matrix_dict
        else:
            model._PTDFs = {}

    # Enforce time 1 ramp rates, relax binaries
    model._enforce_t1_ramp_rates = True
    model._relax_binaries = relax_binaries

    # Set up parameters
    model = default_params(model, model_data)

    # Set up variables
    model = garver_3bin_vars(model)
    model = garver_power_vars(model)
    if model_name == "UnitCommitment":
        model = garver_power_avail_vars(model)
    elif model_name == "EconomicDispatch":
        model = MLR_reserve_vars(model)
    model = file_non_dispatchable_vars(model)

    # Set up constraints
    if model_name == "UnitCommitment":
        model = pan_guan_gentile_KOW_generation_limits(model)
    elif model_name == "EconomicDispatch":
        model = MLR_generation_limits(model)
    model = damcikurt_ramping(model)
    if model_name == "UnitCommitment":
        model = KOW_production_costs_tightened(model)
    elif model_name == "EconomicDispatch":
        model = CA_production_costs(model)
    model = rajan_takriti_UT_DT(model)
    if model_name == "UnitCommitment":
        model = KOW_startup_costs(model)
    elif model_name == "EconomicDispatch":
        model = MLR_startup_costs(model)
    model = storage_services(model)
    model = ancillary_services(model)
    model = ptdf_power_flow(model)
    if model_name == "UnitCommitment":
        model = CA_reserve_constraints(model)
    elif model_name == "EconomicDispatch":
        model = MLR_reserve_constraints(model)

    # Set up objective
    model = basic_objective(model)

    # Set up zero cost constraints
    if objective_hours:
        # Need to Deep Copy of Model Attributes so the Original Attribute
        # does not get removed
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
                model._ProductionCost[g, t].lb = 0.0
                model._ProductionCost[g, t].ub = 0.0

            for g in model._DualFuelGenerators:
                constr = model._DualFuelProductionCost[g, t]
                constr.rhs = 0
                constr.sense = "="

            if model._regulation_service:
                for g in model._AGC_Generators:
                    constr = model._RegulationCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = "="

            for g in model._ThermalGenerators:
                if model._spinning_reserve:
                    constr = model._SpinningReserveCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = "="

                if model._non_spinning_reserve:
                    constr = model._NonSpinningReserveCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = "="

                if model._supplemental_reserve:
                    constr = model._SupplementalReserveCostGeneration[g, t]
                    constr.rhs = 0
                    constr.sense = "="

    # Update the model to incorporate the changes above
    model.update()

    # Save the model in lp file
    if save_model_file_path != "":
        model.write("{}.lp".format(save_model_file_path))
    return model
