


def generate_model(self,
                   model_data,
                   relax_binaries: bool,
                   ptdf_options, ptdf_matrix_dict, objective_hours=None):
    # TODO: do we need to add scaling back in if baseMVA is always 1?
    # copy the model data
    use_model = model_data.clone_in_service()
    model = gp.Model('UnitCommitment')
    model._model_data = use_model.to_egret()  # _model_data in model is egret object, while model_data is vatic object
    model._model_data_vatic = model_data

    # enforce time 1 ramp rates, relax binaries
    model._enforce_t1_ramp_rates = True
    model._relax_binaries = relax_binaries

    # munge PTDF options if necessary
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
    model = ptdf_power_flow(model)
    model = CA_reserve_constraints(model)

    if 'fuel_supply' in model_data._data['elements'] and bool(
            model_data._data['elements']['fuel_supply']):
        fuel_consumption.fuel_consumption_model(model)
        fuel_supply.fuel_supply_model(model)

    else:
        model._fuel_supply = None
        model._fuel_consumption = None

    if 'security_constraint' in model_data._data['elements'] and bool(
            model_data._data['elements']['security_constraint']):
        security_constraints.security_constraint_model(model)
    else:
        model._security_constraints = None

    self._get_formulation('objective')(model)

    if objective_hours:
        zero_cost_hours = set(model._TimePeriods)

        for i, t in enumerate(model._TimePeriods):
            if i < objective_hours:
                zero_cost_hours.remove(t)
            else:
                break

        cost_gens = {g for g, _ in model._ProductionCost}
        for t in zero_cost_hours:
            for g in cost_gens:
                model._ProductionCostConstr[g, t].deactivate()
                model._ProductionCost[g, t].value = 0.
                model._ProductionCost[g, t].fix()

            for g in model._DualFuelGenerators:
                model._DualFuelProductionCost[g, t].expr = 0.

            if model._regulation_service:
                for g in model._AGC_Generators:
                    model._RegulationCostGeneration[g, t].expr = 0.

            if model._spinning_reserve:
                for g in model._ThermalGenerators:
                    model._SpinningReserveCostGeneration[g, t].expr = 0.

            if model._non_spinning_reserve:
                for g in model._ThermalGenerators:
                    model._NonSpinningReserveCostGeneration[g, t].expr = 0.

            if model._supplemental_reserve:
                for g in model._ThermalGenerators:
                    model._SupplementalReserveCostGeneration[g, t].expr = 0.

    self.gurobi_instance = model