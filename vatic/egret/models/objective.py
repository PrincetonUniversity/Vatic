
from pyomo.environ import *

from egret.model_library.unit_commitment.uc_utils import add_model_attr
from egret.model_library.unit_commitment.objective import _add_shutdown_costs


#TODO: this doesn't check if regulation_service is added first,
#      but this will only happen when there are regulation_services!
@add_model_attr('objective',
                requires = {'data_loader': None,
                            'status_vars': ['garver_3bin_vars',
                                            'CA_1bin_vars',
                                            'garver_2bin_vars',
                                            'garver_3bin_relaxed_stop_vars',
                                            'ALS_state_transition_vars'],
                            'power_vars': None,
                            'startup_costs': None,
                            'production_costs': None,
                            'power_balance': None,
                            'reserve_requirement': None,
                            'storage_service': None,
                            'ancillary_service': None,})
def vatic_objective(model):
    '''
    adds the objective and shutdown cost formulation to the model
    '''

    #############################################
    # constraints for computing cost components #
    #############################################

    def compute_no_load_cost_rule(m, g, t):
        return (m.MinimumProductionCost[g, t] * m.UnitOn[g, t]
                * m.TimePeriodLengthHours)

    model.NoLoadCost = Expression(model.SingleFuelGenerators,
                                  model.TimePeriods,
                                  rule=compute_no_load_cost_rule)

    _add_shutdown_costs(model)

    # compute the total production costs, across
    # all generators and time periods
    def compute_total_production_cost_rule(m, t):
        total_cost = sum(m.ProductionCost[g, t]
                         for g in m.SingleFuelGenerators)
        total_cost += sum(m.ProductionCost[g, t]
                          for g in m.AllNondispatchableGenerators)
        total_cost += sum(m.DualFuelProductionCost[g, t]
                          for g in m.DualFuelGenerators)

        return total_cost

    model.TotalProductionCost = Expression(
        model.TimePeriods, rule=compute_total_production_cost_rule)

    #
    # Cost computations
    #

    def commitment_stage_cost_expression_rule(m, st):
        cc = sum(sum(m.NoLoadCost[g, t] + m.StartupCost[g, t]
                     for g in m.SingleFuelGenerators)
                 + sum(m.DualFuelCommitmentCost[g, t]
                       for g in m.DualFuelGenerators)
                 + sum(m.ShutdownCost[g, t] for g in m.ThermalGenerators)

                 for t in m.CommitmentTimeInStage[st])

        if m.regulation_service:
            cc += sum(m.RegulationCostCommitment[g, t]
                      for g in m.AGC_Generators
                      for t in m.CommitmentTimeInStage[st])

        return cc

    model.CommitmentStageCost = Expression(
        model.StageSet, rule=commitment_stage_cost_expression_rule)

    def compute_reserve_shortfall_cost_rule(m, t):
        return (m.ReserveShortfallPenalty * m.ReserveShortfall[t]
                * m.TimePeriodLengthHours)

    model.ReserveShortfallCost = Expression(
        model.TimePeriods, rule=compute_reserve_shortfall_cost_rule)

    def generation_stage_cost_expression_rule(m, st):
        cc = sum(sum(m.ProductionCost[g, t] for g in m.SingleFuelGenerators)
                 + sum(m.ProductionCost[g, t]
                       for g in m.AllNondispatchableGenerators)
                 + sum(m.DualFuelProductionCost[g, t]
                       for g in m.DualFuelGenerators)

                 for t in m.GenerationTimeInStage[st])

        cc += sum(m.LoadMismatchCost[t] for t in m.GenerationTimeInStage[st])
        cc += sum(m.ReserveShortfallCost[t]
                  for t in m.GenerationTimeInStage[st])

        cc += sum(m.BranchViolationCost[t]
                  for t in m.GenerationTimeInStage[st])
        cc += sum(m.InterfaceViolationCost[t]
                  for t in m.GenerationTimeInStage[st])
        cc += sum(m.ContingencyViolationCost[t]
                  for t in m.GenerationTimeInStage[st])
        cc += sum(m.StorageCost[s,t]
                  for s in m.Storage for t in m.GenerationTimeInStage[st])

        if m.reactive_power:
            cc += sum(m.LoadMismatchCostReactive[t]
                      for t in m.GenerationTimeInStage[st])

        if m.security_constraints:
            cc += sum(m.SecurityConstraintViolationCost[t]
                      for t in m.GenerationTimeInStage[st])

        if m.regulation_service:
            cc += sum(m.RegulationCostGeneration[g, t]
                      for g in m.AGC_Generators
                      for t in m.GenerationTimeInStage[st])
            cc += sum(m.RegulationCostPenalty[t]
                      for t in m.GenerationTimeInStage[st])

        if m.spinning_reserve:
            cc += sum(m.SpinningReserveCostGeneration[g, t]
                      for g in m.ThermalGenerators
                      for t in m.GenerationTimeInStage[st])
            cc += sum(m.SpinningReserveCostPenalty[t]
                      for t in m.GenerationTimeInStage[st])

        if m.non_spinning_reserve:
            cc += sum(m.NonSpinningReserveCostGeneration[g, t]
                      for g in m.NonSpinGenerators
                      for t in m.GenerationTimeInStage[st])
            cc += sum(m.NonSpinningReserveCostPenalty[t]
                      for t in m.GenerationTimeInStage[st])

        if m.supplemental_reserve:
            cc += sum(m.SupplementalReserveCostGeneration[g, t]
                      for g in m.ThermalGenerators
                      for t in m.GenerationTimeInStage[st])
            cc += sum(m.SupplementalReserveCostPenalty[t]
                      for t in m.GenerationTimeInStage[st])

        if m.flexible_ramping:
            cc += sum(m.FlexibleRampingCostPenalty[t]
                      for t in m.GenerationTimeInStage[st])

        return cc

    model.GenerationStageCost = Expression(
        model.StageSet, rule=generation_stage_cost_expression_rule)

    def stage_cost_expression_rule(m, st):
        return m.GenerationStageCost[st] + m.CommitmentStageCost[st]

    model.StageCost = Expression(
        model.StageSet, rule=stage_cost_expression_rule)

    #
    # Objectives
    #

    def total_cost_objective_rule(m):
       return sum(m.StageCost[st] for st in m.StageSet)

    model.TotalCostObjective = Objective(
        rule=total_cost_objective_rule, sense=minimize)
