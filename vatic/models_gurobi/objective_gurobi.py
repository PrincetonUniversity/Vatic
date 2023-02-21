from gurobipy import tupledict, LinExpr, quicksum, GRB

component_name = 'objective'


def _1bin_shutdown_costs(model, add_shutdown_cost_var=True):
    if add_shutdown_cost_var:
        model._ShutdownCost = model.addVars(
                    model.ThermalGenerators, model.TimePeriods,
                    lb = 0, ub = GRB.INFINITY)

    def compute_shutdown_costs_rule(m, g, t):
        if t == m._InitialTime:
            return m._ShutdownCost[g, t] >= m._ShutdownFixedCost[g] * (
                        m._UnitOnT0[g] - m._UnitOn[g, t])
        else:
            return m._ShutdownCost[g, t] >= m._ShutdownFixedCost[g] * (
                        m._UnitOn[g, t - 1] - m._UnitOn[g, t])

    model._ComputeShutdownCosts = model.addConstrs(
            (compute_shutdown_costs_rule(model, g, t)
                for g in model._ThermalGenerators for t in model._TimePeriods),
            name = 'ComputeShutdownCosts'
    )


def _3bin_shutdown_costs(model, add_shutdown_cost_var=True):
    #############################################################
    # compute the per-generator, per-time period shutdown costs #
    #############################################################
    ## BK -- replaced with UnitStop

    if add_shutdown_cost_var:
        model._ShutdownCost = model.addVars(
                    model._ThermalGenerators, model._TimePeriods,
                    lb = -GRB.INFINITY, ub = GRB.INFINITY)

    def compute_shutdown_costs_rule(m, g, t):
        linear_vars = [m._ShutdownCost[g, t], m._UnitStop[g, t]]
        linear_coefs = [-1., m._ShutdownFixedCost[g]]
        return (LinExpr(linear_coefs, linear_vars) <= 0)

    model._ComputeShutdownCosts = model.addConstrs(
        (compute_shutdown_costs_rule(model, g, t)
            for g in model._ThermalGenerators for t in model._TimePeriods),
        name = 'ComputeShutdownCosts'
    )

def _add_shutdown_costs(model, add_shutdown_cost_var=True):
    #NOTE: we handle shutdown costs in this manner because it's not a
    #      common point of contention in the literature, and they're
    #      often zero as is.
    if model._status_vars in ['garver_3bin_vars','garver_3bin_relaxed_stop_vars','garver_2bin_vars', 'ALS_state_transition_vars']:
        _3bin_shutdown_costs(model, add_shutdown_cost_var)
    elif model._status_vars in ['CA_1bin_vars',]:
        _1bin_shutdown_costs(model, add_shutdown_cost_var)
    else:
        raise Exception("Problem adding shutdown costs, cannot identify status_vars for this model")


def basic_objective(model):
    '''
    adds the objective and shutdown cost formulation to the model
    '''

    #############################################
    # constraints for computing cost components #
    #############################################

    def compute_no_load_cost_rule(m ,g ,t):
        return m._MinimumProductionCost[g ,t ] *m._UnitOn[g ,t ] *m._TimePeriodLengthHours

    model._NoLoadCost = {(g, t): compute_no_load_cost_rule(model, g, t)
                                    for g in model._SingleFuelGenerators
                                    for t in model._TimePeriods}
    model._NoLoadCost = tupledict(model._NoLoadCost)

    _add_shutdown_costs(model)

    # compute the total production costs, across all generators and time periods.
    def compute_total_production_cost_rule(m, t):
        return sum(m._ProductionCost[g, t] for g in m._SingleFuelGenerators) + \
               sum(m._DualFuelProductionCost[g ,t] for g in m._DualFuelGenerators)

    model._TotalProductionCost = {t: compute_total_production_cost_rule(model, t)
                                    for t in model._TimePeriods}

    #
    # Cost computations
    #

    def commitment_stage_cost_expression_rule(m, st):
        cc = sum(sum(m._NoLoadCost[g ,t] + m._StartupCost[g ,t] for g in m._SingleFuelGenerators) + \
                 sum(m._DualFuelCommitmentCost[g ,t] for g in m._DualFuelGenerators) + \
                 sum(m._ShutdownCost[g ,t] for g in m._ThermalGenerators)
                 for t in m._CommitmentTimeInStage[st])
        if m._regulation_service:
            cc += sum \
                (m._RegulationCostCommitment[g ,t] for g in m._AGC_Generators for t in m._CommitmentTimeInStage[st])
        return cc

    model._CommitmentStageCost = {st: commitment_stage_cost_expression_rule(model, st)
                                    for st in model._StageSet}

    def compute_reserve_shortfall_cost_rule(m, t):
        return m._ReserveShortfallPenalty *m._TimePeriodLengthHours \
               *m._ReserveShortfall[t]

    model._ReserveShortfallCost = {t: compute_reserve_shortfall_cost_rule(model, t)
                                    for t in model._TimePeriods}

    def generation_stage_cost_expression_rule(m, st):
        cc = sum(sum(m._ProductionCost[g, t] for g in m._SingleFuelGenerators) + \
                 sum(m._DualFuelProductionCost[g ,t] for g in m._DualFuelGenerators)
                 for t in m._GenerationTimeInStage[st]) + \
             sum(m._LoadMismatchCost[t] for t in m._GenerationTimeInStage[st]) + \
             sum(m._ReserveShortfallCost[t] for t in m._GenerationTimeInStage[st]) + \
             sum(m._BranchViolationCost[t] for t in m._GenerationTimeInStage[st]) + \
             sum(m._InterfaceViolationCost[t] for t in m._GenerationTimeInStage[st]) + \
             sum(m._ContingencyViolationCost[t] for t in m._GenerationTimeInStage[st]) + \
             sum(m._StorageCost[s ,t] for s in m._Storage for t in m._GenerationTimeInStage[st])
        if m._reactive_power:
            cc += sum(m._LoadMismatchCostReactive[t] for t in m._GenerationTimeInStage[st])
        if m._security_constraints:
            cc += sum(m._SecurityConstraintViolationCost[t] for t in m._GenerationTimeInStage[st])
        if m._regulation_service:
            cc += sum \
                (m._RegulationCostGeneration[g ,t] for g in m._AGC_Generators for t in m._GenerationTimeInStage[st]) \
                  + sum(m._RegulationCostPenalty[t] for t in m._GenerationTimeInStage[st])
        if m._spinning_reserve:
            cc += sum(m._SpinningReserveCostGeneration[g ,t] for g in m._ThermalGenerators for t in m._GenerationTimeInStage[st]) \
                  + sum(m._SpinningReserveCostPenalty[t] for t in m._GenerationTimeInStage[st])
        if m._non_spinning_reserve:
            cc += sum(m._NonSpinningReserveCostGeneration[g ,t] for g in m._NonSpinGenerators for t in m._GenerationTimeInStage[st]) \
                  + sum(m._NonSpinningReserveCostPenalty[t] for t in m._GenerationTimeInStage[st])
        if m._supplemental_reserve:
            cc += sum(m._SupplementalReserveCostGeneration[g ,t] for g in m._ThermalGenerators for t in m._GenerationTimeInStage[st]) \
                  + sum(m._SupplementalReserveCostPenalty[t] for t in m._GenerationTimeInStage[st])
        if m._flexible_ramping:
            cc += sum(m._FlexibleRampingCostPenalty[t] for t in m._GenerationTimeInStage[st])
        return cc

    model._GenerationStageCost = {st: generation_stage_cost_expression_rule(model, st)
                                    for st in model._StageSet}

    def stage_cost_expression_rule(m, st):
        return m._GenerationStageCost[st] + m._CommitmentStageCost[st]
    model._StageCost = {st: stage_cost_expression_rule(model, st)
                                    for st in model._StageSet}

    #
    # Objectives
    #

    def total_cost_objective_rule(m):
        return quicksum(m._StageCost[st] for st in m._StageSet)

    model.setObjective(total_cost_objective_rule(model), GRB.MINIMIZE)
    model.update()
    return model