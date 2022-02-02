
import pyomo.environ as pe

from egret.model_library.unit_commitment.uc_utils import (
    add_model_attr, get_linear_expr)
from egret.model_library.unit_commitment.production_costs import (
    _step_coeff, _compute_total_production_cost)


def _get_piecewise_production_generators(model):

    # more than two points -> not linear
    def piecewise_generators_time_set(m):
        for g in m.ThermalGenerators | m.AllNondispatchableGenerators:
            for t in m.TimePeriods:
                if len(m.PowerGenerationPiecewisePoints[g, t]) > 2:
                    yield g, t

    model.PiecewiseGeneratorTimeIndexSet = pe.Set(
        dimen=2, initialize=piecewise_generators_time_set)

    # two points -> linear
    def linear_generators_time_set(m):
        for g in m.ThermalGenerators | m.AllNondispatchableGenerators:
            for t in m.TimePeriods:
                if len(m.PowerGenerationPiecewisePoints[g, t]) == 2:
                    yield g, t

    model.LinearGeneratorTimeIndexSet = pe.Set(
        dimen=2, initialize=linear_generators_time_set)

    # if there's only 1 or zero points, this has no marginal cost

    # compute the per-generator, per-time period production costs
    def piecewise_production_costs_index_set_generator(m):
        return (
            (g, t, i) for g,t in m.PiecewiseGeneratorTimeIndexSet
            for i in range(len(m.PowerGenerationPiecewisePoints[g, t]) - 1)
            )

    model.PiecewiseProductionCostsIndexSet = pe.Set(
        initialize=piecewise_production_costs_index_set_generator, dimen=3)


def _basic_production_costs_vars(model):
    _get_piecewise_production_generators(model)

    def piecewise_production_bounds_rule(m, g, t, i):
        return (0, m.PowerGenerationPiecewisePoints[g, t][i + 1] -
                m.PowerGenerationPiecewisePoints[g, t][i])

    model.PiecewiseProduction = pe.Var(model.PiecewiseProductionCostsIndexSet,
                                       within=pe.NonNegativeReals,
                                       bounds=piecewise_production_bounds_rule)

    linear_expr = get_linear_expr(model.PowerGeneratedAboveMinimum)

    def piecewise_production_sum_rule(m, g, t):
        linear_vars = [
            m.PiecewiseProduction[g, t, i]
            for i in range(len(m.PowerGenerationPiecewisePoints[g, t]) - 1)
            ]

        linear_coefs = [1.] * len(linear_vars)

        if g in m.ThermalGenerators:
            linear_vars.append(m.PowerGeneratedAboveMinimum[g, t])
        else:
            linear_vars.append(m.NondispatchablePowerUsed[g, t])

        linear_coefs.append(-1.)

        return (linear_expr(linear_vars=linear_vars,
                            linear_coefs=linear_coefs), 0.)

    model.PiecewiseProductionSum = pe.Constraint(
        model.PiecewiseGeneratorTimeIndexSet,
        rule=piecewise_production_sum_rule
        )


def _basic_production_costs_constr(model):
    model.ProductionCost = pe.Var(
        model.SingleFuelGenerators | model.AllNondispatchableGenerators,
        model.TimePeriods,
        within=pe.Reals
        )

    linear_expr = get_linear_expr()

    def piecewise_production_costs_rule(m, g, t):
        if (g, t) in m.PiecewiseGeneratorTimeIndexSet:
            points = m.PowerGenerationPiecewisePoints[g, t]
            costs = m.PowerGenerationPiecewiseCostValues[g, t]
            time_scale = m.TimePeriodLengthHours

            linear_coefs = [(time_scale * costs[i + 1] - time_scale * costs[i])
                            / (points[i + 1] - points[i])
                            for i in range(len(points) - 1)]

            linear_vars = [m.PiecewiseProduction[g, t, i]
                           for i in range(len(points) - 1)]

            linear_coefs.append(-1.)
            linear_vars.append(m.ProductionCost[g, t])
            return (linear_expr(linear_vars=linear_vars,
                                linear_coefs=linear_coefs), 0.)

        elif (g, t) in m.LinearGeneratorTimeIndexSet:
            i = 0
            points = m.PowerGenerationPiecewisePoints[g, t]
            costs = m.PowerGenerationPiecewiseCostValues[g, t]
            time_scale = m.TimePeriodLengthHours

            slope = time_scale * costs[i + 1] - time_scale * costs[i]
            slope /= points[i + 1] - points[i]

            if g in m.ThermalGenerators:
                linear_vars, linear_coefs = \
                    m._get_power_generated_above_minimum_lists(m, g, t)
            else:
                linear_vars = [m.NondispatchablePowerUsed[g, t]]
                linear_coefs = [1.]

            linear_coefs = [slope * coef for coef in linear_coefs]
            linear_vars.append(m.ProductionCost[g, t])
            linear_coefs.append(-1.)

            return (linear_expr(linear_vars=linear_vars,
                                linear_coefs=linear_coefs), 0.)

        else:
            return (m.ProductionCost[g, t], 0.)

    model.ProductionCostConstr1 = pe.Constraint(
        model.SingleFuelGenerators | model.AllNondispatchableGenerators,
        model.TimePeriods,
        rule=piecewise_production_costs_rule
        )

    _compute_total_production_cost(model)


@add_model_attr('production_costs',
                requires = {'data_loader': None,
                            'status_vars': ['garver_3bin_vars',
                                            'garver_3bin_relaxed_stop_vars',
                                            'garver_2bin_vars',
                                            'ALS_state_transition_vars'],
                            'power_vars': None})
def KOW_Vatic_production_costs_tightened(model):
    '''
    Base for similarities between tightend and not KOW production costs
    '''
    _basic_production_costs_vars(model)

    linear_expr = get_linear_expr(model.UnitOn, model.UnitStart,
                                  model.UnitStop)

    def piecewise_production_limits_rule(m, g, t, i):
        # these can always be tightened based on SU/SD, regardless of the
        # ramping/aggregation since PowerGenerationPiecewisePoints are scaled
        # to MinimumPowerOutput, we need to scale Startup/Shutdown ramps to it
        # as well
        if g in m.AllNondispatchableGenerators:
            return pe.Constraint.Skip

        upper = pe.value(m.PowerGenerationPiecewisePoints[g, t][i + 1])
        lower = pe.value(m.PowerGenerationPiecewisePoints[g, t][i])

        SU = pe.value(m.ScaledStartupRampLimit[g, t])
        minP = pe.value(m.MinimumPowerOutput[g, t])
        su_step = _step_coeff(upper, lower, SU - minP)

        if t < pe.value(m.NumTimePeriods):
            SD = pe.value(m.ScaledShutdownRampLimit[g, t])
            UT = pe.value(m.ScaledMinimumUpTime[g])
            sd_step = _step_coeff(upper, lower, SD - minP)

            if UT > 1:
                linear_vars = [m.PiecewiseProduction[g, t, i], m.UnitOn[g, t],
                               m.UnitStart[g, t], m.UnitStop[g, t + 1]]
                linear_coefs = [-1., (upper - lower), -su_step, -sd_step]

                return (0, linear_expr(linear_vars=linear_vars,
                                       linear_coefs=linear_coefs), None)

            else:  ## MinimumUpTime[g] <= 1
                linear_vars = [m.PiecewiseProduction[g, t, i], m.UnitOn[g, t],
                               m.UnitStart[g, t], ]
                linear_coefs = [-1., (upper - lower), -su_step, ]

                # tightening
                coef = -max(sd_step - su_step, 0)
                if coef != 0:
                    linear_vars.append(m.UnitStop[g, t + 1])
                    linear_coefs.append(coef)

                return (0, linear_expr(linear_vars=linear_vars,
                                       linear_coefs=linear_coefs), None)

        else:  ## t >= value(m.NumTimePeriods)
            linear_vars = [m.PiecewiseProduction[g, t, i], m.UnitOn[g, t],
                           m.UnitStart[g, t], ]
            linear_coefs = [-1., (upper - lower), -su_step, ]

            return (0, linear_expr(linear_vars=linear_vars,
                                   linear_coefs=linear_coefs), None)

    model.PiecewiseProductionLimits = pe.Constraint(
        model.PiecewiseProductionCostsIndexSet,
        rule=piecewise_production_limits_rule
        )

    def piecewise_production_limits_rule2(m, g, t, i):
        # these can always be tightened based on SU/SD, regardless of the
        # ramping/aggregation since PowerGenerationPiecewisePoints are scaled
        # to MinimumPowerOutput, we need to scale Startup/Shutdown ramps to
        # it as well

        if g in m.AllNondispatchableGenerators:
            return pe.Constraint.Skip

        UT = pe.value(m.ScaledMinimumUpTime[g])

        if UT <= 1 and t < pe.value(m.NumTimePeriods):
            upper = pe.value(m.PowerGenerationPiecewisePoints[g, t][i + 1])
            lower = pe.value(m.PowerGenerationPiecewisePoints[g, t][i])
            SD = pe.value(m.ScaledShutdownRampLimit[g, t])
            minP = pe.value(m.MinimumPowerOutput[g, t])

            sd_step = _step_coeff(upper, lower, SD - minP)
            linear_vars = [m.PiecewiseProduction[g, t, i], m.UnitOn[g, t],
                           m.UnitStop[g, t + 1], ]
            linear_coefs = [-1., (upper - lower), -sd_step, ]

            # tightening
            SU = pe.value(m.ScaledStartupRampLimit[g, t])
            su_step = _step_coeff(upper, lower, SU - minP)
            coef = -max(su_step - sd_step, 0)

            if coef != 0:
                linear_vars.append(m.UnitStart[g, t])
                linear_coefs.append(coef)

            return (0, linear_expr(linear_vars=linear_vars,
                                   linear_coefs=linear_coefs), None)

        # MinimumUpTime[g] > 1 or we added it in the
        # t == value(m.NumTimePeriods) clause above
        else:
            return pe.Constraint.Skip

    model.PiecewiseProductionLimits2 = pe.Constraint(
        model.PiecewiseProductionCostsIndexSet,
        rule=piecewise_production_limits_rule2
        )

    _basic_production_costs_constr(model)
