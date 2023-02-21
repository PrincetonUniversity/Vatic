from gurobipy import tupledict, LinExpr, quicksum, GRB

component_name = 'production_costs'

def _production_cost_function(m, g, t, i):
    return m._TimePeriodLengthHours * m._PowerGenerationPiecewiseCostValues[g,t][i]

def _compute_total_production_cost(model):
    ## helper function for PH
    def compute_production_costs_rule(m, g, t, avg_power):
        ## piecewise points for power
        piecewise_points = m._PowerGenerationPiecewisePoints[g, t]
        ## buckets
        piecewise_eval = [0] * (len(piecewise_points) - 1)
        ## fill the buckets (skip the first since it's min power)
        for l in range(len(piecewise_eval)):
            ## fill this bucket all the way
            if avg_power >= piecewise_points[l + 1]:
                piecewise_eval[l] = piecewise_points[l + 1] - piecewise_points[
                    l]
            ## fill the bucket part way and stop
            elif avg_power < piecewise_points[l + 1]:
                piecewise_eval[l] = avg_power - piecewise_points[l]
                break

                # slope * production
        return sum((_production_cost_function(m, g, t,
                                              l + 1) - _production_cost_function(
            m, g, t, l)) / (piecewise_points[l + 1] - piecewise_points[l]) *
                   piecewise_eval[l] for l in range(len(piecewise_eval)))

    model._ComputeProductionCosts = compute_production_costs_rule

def _step_coeff(upper, lower, susd):
    if lower < susd < upper:
        return upper - susd
    elif susd <= lower:
        return upper - lower
    elif upper <= susd:
        return 0
    print("Something went wrong, step_coeff is returning None...")
    print("lower: ", lower)
    print("susd: ", susd)
    print("upper: ", upper)
    return None

def _get_piecewise_production_generators(model):
    # more than two points -> not linear
    def piecewise_generators_time_set(m):
        for g in m._ThermalGenerators:
            for t in m._TimePeriods:
                if len(m._PowerGenerationPiecewisePoints[g,t]) > 2:
                    yield g,t
    model._PiecewiseGeneratorTimeIndexSet = \
        list(piecewise_generators_time_set(model))

    # two points -> linear
    def linear_generators_time_set(m):
        for g in m._ThermalGenerators:
            for t in m._TimePeriods:
                if len(m._PowerGenerationPiecewisePoints[g,t]) == 2:
                    yield g,t
    model._LinearGeneratorTimeIndexSet = \
       list(linear_generators_time_set(model))

    # if there's only 1 or zero points, this has no marginal cost

    # compute the per-generator, per-time period production costs. We'll do this by hand
    def piecewise_production_costs_index_set_generator(m):
        return ((g,t,i) for g,t in m._PiecewiseGeneratorTimeIndexSet
                for i in range(len(m._PowerGenerationPiecewisePoints[g,t])-1))
    model._PiecewiseProductionCostsIndexSet = \
        list(piecewise_production_costs_index_set_generator(model))

def _basic_production_costs_vars(model):
    _get_piecewise_production_generators(model)

    model._PiecewiseProduction = model.addVars(model._PiecewiseProductionCostsIndexSet,
        lb = 0,
        ub = [model._PowerGenerationPiecewisePoints[g_t_i[0], g_t_i[1]][g_t_i[2] + 1]
                - model._PowerGenerationPiecewisePoints[g_t_i[0], g_t_i[1]][g_t_i[2]]
             for g_t_i in model._PiecewiseProductionCostsIndexSet],
        name = 'PiecewiseProduction')

    def piecewise_production_sum_rule(m, g, t):
        linear_vars = list(m._PiecewiseProduction[g, t, i] for i in range(
            len(m._PowerGenerationPiecewisePoints[g, t]) - 1))
        linear_coefs = [1.] * len(linear_vars)
        linear_vars.append(m._PowerGeneratedAboveMinimum[g, t])
        linear_coefs.append(-1.)
        return LinExpr(linear_coefs, linear_vars) <= 0

    model._PiecewiseProductionSum = model.addConstrs((
        piecewise_production_sum_rule(model, g_t[0], g_t[1])
            for g_t in model._PiecewiseGeneratorTimeIndexSet),
        name = 'PiecewiseProductionSum')

def _basic_production_costs_constr(model):
    model._ProductionCost = model.addVars(
        model._SingleFuelGenerators, model._TimePeriods,
        lb =-GRB.INFINITY , ub = GRB.INFINITY,
        name = 'ProductionCost')

    def piecewise_production_costs_rule(m, g, t):
        if (g, t) in m._PiecewiseGeneratorTimeIndexSet:
            points = m._PowerGenerationPiecewisePoints[g, t]
            costs = m._PowerGenerationPiecewiseCostValues[g, t]
            time_scale = m._TimePeriodLengthHours
            linear_coefs = [
                (time_scale * costs[i + 1] - time_scale * costs[i]) / (
                            points[i + 1] - points[i]) \
                for i in range(len(points) - 1)]
            linear_vars = [m._PiecewiseProduction[g, t, i] for i in
                           range(len(points) - 1)]
            linear_coefs.append(-1.)
            linear_vars.append(m._ProductionCost[g, t])
            return LinExpr(linear_coefs, linear_vars) <= 0
        elif (g, t) in m._LinearGeneratorTimeIndexSet:
            i = 0
            points = m._PowerGenerationPiecewisePoints[g, t]
            costs = m._PowerGenerationPiecewiseCostValues[g, t]
            time_scale = m._TimePeriodLengthHours
            slope = (time_scale * costs[i + 1] - time_scale * costs[i]) / (
                        points[i + 1] - points[i])
            linear_vars, linear_coefs = m._get_power_generated_above_minimum_lists(
                m, g, t)
            linear_coefs = [slope * coef for coef in linear_coefs]
            linear_vars.append(m._ProductionCost[g, t])
            linear_coefs.append(-1.)
            return LinExpr(linear_coefs, linear_vars) <= 0
        else:
            return m.ProductionCost[g, t] <= 0.

    model._ProductionCostConstr = \
        model.addConstrs((piecewise_production_costs_rule(model, g, t)
                            for g in model._SingleFuelGenerators
                            for t in model._TimePeriods),
                          name = 'piecewise_production_costs')

    _compute_total_production_cost(model)

def KOW_production_costs_tightened(model):

    '''
    this is the (more ideal) production cost model introducted by:

    Ben Knueven, Jim Ostrowski, and Jean-Paul Watson. Exploiting identical
    generators in unit commitment. IEEE Transactions on Power Systems,
    33(4), 2018.

    equations (19d)--(19h) with some tightening for when SU != SD, as mentioned in text
    '''
    _KOW_production_costs(model, True)
    model.update()
    return model


def _KOW_production_costs(model, tightened=False):
    '''
    Base for similarities between tightend and not KOW production costs
    '''
    _basic_production_costs_vars(model)

    def piecewise_production_limits_rule(m, g, t, i):
        ### these can always be tightened based on SU/SD, regardless of the ramping/aggregation
        ### since PowerGenerationPiecewisePoints are scaled to MinimumPowerOutput, we need to scale Startup/Shutdown ramps to it as well
        upper = m._PowerGenerationPiecewisePoints[g, t][i + 1]
        lower = m._PowerGenerationPiecewisePoints[g, t][i]
        SU = m._ScaledStartupRampLimit[g, t]
        minP = m._MinimumPowerOutput[g, t]

        su_step = _step_coeff(upper, lower, SU - minP)
        if t < m._NumTimePeriods:
            SD = m._ScaledShutdownRampLimit[g, t]
            UT = m._ScaledMinimumUpTime[g]
            sd_step = _step_coeff(upper, lower, SD - minP)
            if UT > 1:
                linear_vars = [m._PiecewiseProduction[g, t, i], m._UnitOn[g, t],
                               m._UnitStart[g, t], m._UnitStop[g, t + 1]]
                linear_coefs = [-1., (upper - lower), -su_step, -sd_step]
                return LinExpr(linear_coefs, linear_vars) >= 0
            else:  ## MinimumUpTime[g] <= 1
                linear_vars = [m._PiecewiseProduction[g, t, i], m._UnitOn[g, t],
                               m._UnitStart[g, t], ]
                linear_coefs = [-1., (upper - lower), -su_step, ]
                if tightened:
                    coef = -max(sd_step - su_step, 0)
                    if coef != 0:
                        linear_vars.append(m._UnitStop[g, t + 1])
                        linear_coefs.append(coef)
                return LinExpr(linear_coefs, linear_vars) >= 0
        else:  ## t >= value(m.NumTimePeriods)
            linear_vars = [m._PiecewiseProduction[g, t, i], m._UnitOn[g, t],
                           m._UnitStart[g, t], ]
            linear_coefs = [-1., (upper - lower), -su_step, ]
            return LinExpr(linear_coefs, linear_vars) >= 0

    model._PiecewiseProductionLimits = model.addConstrs((
        piecewise_production_limits_rule(model, g_t_i[0], g_t_i[1], g_t_i[2])
        for g_t_i in model._PiecewiseProductionCostsIndexSet),
        name='piecewise_production_limits')

    def piecewise_production_limits_rule2(m, g, t, i):
        ### these can always be tightened based on SU/SD, regardless of the ramping/aggregation
        ### since PowerGenerationPiecewisePoints are scaled to MinimumPowerOutput, we need to scale Startup/Shutdown ramps to it as well
        UT = m._ScaledMinimumUpTime[g]
        if UT <= 1 and t < m._NumTimePeriods:
            upper = m._PowerGenerationPiecewisePoints[g, t][i + 1]
            lower = m._PowerGenerationPiecewisePoints[g, t][i]
            SD = m._ScaledShutdownRampLimit[g, t]
            minP = m._MinimumPowerOutput[g, t]

            sd_step = _step_coeff(upper, lower, SD - minP)
            linear_vars = [m._PiecewiseProduction[g, t, i], m._UnitOn[g, t],
                           m._UnitStop[g, t + 1], ]
            linear_coefs = [-1., (upper - lower), -sd_step, ]
            if tightened:
                SU = m._ScaledStartupRampLimit[g, t]
                su_step = _step_coeff(upper, lower, SU - minP)
                coef = -max(su_step - sd_step, 0)
                if coef != 0:
                    linear_vars.append(m._UnitStart[g, t])
                    linear_coefs.append(coef)
            return LinExpr(linear_coefs, linear_vars) >= 0
        else:  ## MinimumUpTime[g] > 1 or we added it in the t == value(m.NumTimePeriods) clause above
            return None

    piecewise_production_limits_rule2_cons = {}
    for idx in model._PiecewiseProductionCostsIndexSet:
        g, t, i = idx[0], idx[1], idx[2]
        cons = piecewise_production_limits_rule2(model, g, t, i)
        if cons != None:
            piecewise_production_limits_rule2_cons[idx] = cons

    model._PiecewiseProductionLimits2 = \
        model.addConstrs((piecewise_production_limits_rule2_cons[g_t_i]
                            for g_t_i in piecewise_production_limits_rule2_cons.keys()),
                         name = 'piecewise_production_limits_rule2')

    _basic_production_costs_constr(model)

