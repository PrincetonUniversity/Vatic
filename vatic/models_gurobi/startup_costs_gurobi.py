from gurobipy import tupledict, LinExpr, quicksum, GRB

from ._status_vars_gurobi import _is_relaxed

component_name = 'startup_costs'


def KOW_startup_costs(model, add_startup_cost_var=True):
    '''
    Start-up cost formulation "Match" from

    Ben Knueven, Jim Ostrowski, and Jean-Paul Watson. A novel matching
    formulation for startup costs in unit commitment, 2017.
    URL http://www.optimization-online.org/DB_FILE/2017/03/5897.pdf.
    '''

    # begin ostrowski startup costs
    # time_period_list = model._TimePeriods
    initial_time = model._TimePeriods[0]
    after_last_time = model._TimePeriods[-1] + 1

    def ValidShutdownTimePeriods_generator(m, g):
        ## for speed, if we don't have different startups
        if len(m._ScaledStartupLags[g]) <= 1:
            return []
        ## adds the necessary index for starting-up after a shutdown before the time horizon began
        unit_on_t0_state = m._UnitOnT0State[g]
        if unit_on_t0_state >= 0:
            return model._TimePeriods
        else:
            return model._TimePeriods + [initial_time + int(
                round(unit_on_t0_state / m._TimePeriodLengthHours))]

    model._ValidShutdownTimePeriods = {g: ValidShutdownTimePeriods_generator(model,
                                        g) for g in model._ThermalGenerators}

    def ShutdownHotStartupPairs_generator(m, g):
        ## for speed, if we don't have different startups
        if len(m._ScaledStartupLags[g]) <= 1:
            return ()
        first_lag = m._ScaledStartupLags[g][0]
        last_lag = m._ScaledStartupLags[g][-1]
        for t_prime in m._ValidShutdownTimePeriods[g]:
            t_first = first_lag + t_prime
            t_last = last_lag + t_prime
            if t_first < initial_time:
                t_first = initial_time
            if t_last > after_last_time:
                t_last = after_last_time
            for t in range(t_first, t_last):
                yield (t_prime, t)

    model._ShutdownHotStartupPairs = {g: list(ShutdownHotStartupPairs_generator(model, g))
             for g in model._ThermalGenerators}

    # (g,t',t) will be an inidicator for g for shutting down at time t' and starting up at time t

    model._StartupIndicator_domain = [(g, t_prime, t) for g in model._ThermalGenerators for t_prime, t in
                model._ShutdownHotStartupPairs[g]]

    if _is_relaxed(model):
        model._StartupIndicator = model.addVars(model._StartupIndicator_domain,
                                                    lb = 0, ub = 1, name = 'StartupIndicator')
    else:
        model._StartupIndicator = model.addVars(model._StartupIndicator_domain,
                                                     vtype=GRB.BINARY, name = 'StartupIndicator')

    ############################################################
    # compute the per-generator, per-time period startup costs #
    ############################################################

    model._GeneratorShutdownPeriods = [(g, t)
                                      for g in model._ThermalGenerators
                                      for t in model._ValidShutdownTimePeriods[g]]

    model._ShutdownsByStartups = {(g, t): [] for g in model._ThermalGenerators
                                    for t in model._TimePeriods}
    model._StartupsByShutdowns = {g_t: [] for g_t in model._GeneratorShutdownPeriods}

    for g, t_p, t in model._StartupIndicator_domain:
        model._ShutdownsByStartups[g, t].append(t_p)
        model._StartupsByShutdowns[g, t_p].append(t)

    def startup_match_rule(m, g, t):
        linear_vars = list(m._StartupIndicator[g, t_prime, t] for t_prime in
                           m._ShutdownsByStartups[g, t])
        linear_coefs = [1.] * len(linear_vars)
        linear_vars.append(m._UnitStart[g, t])
        linear_coefs.append(-1.)
        return (LinExpr(linear_coefs, linear_vars) <= 0)

    model._StartupMatch = model.addConstrs((startup_match_rule(model, g, t)
        for g in model._ThermalGenerators for t in model._TimePeriods),
                                    name='StartupMatch')

    def shutdown_match_rule(m, g, t):
        begin_times = m._StartupsByShutdowns[g, t]
        if not begin_times:  ##if this is empty
            return None
        linear_vars = list(
            m._StartupIndicator[g, t, t_p] for t_p in begin_times)
        linear_coefs = [1.] * len(linear_vars)
        if t < initial_time:
            return LinExpr(linear_coefs, linear_vars) <= 1
        else:
            linear_vars.append(m._UnitStop[g, t])
            linear_coefs.append(-1.)
            return LinExpr(linear_coefs, linear_vars) <= 0

    shutdown_match_cons = {}
    for g_t in model._GeneratorShutdownPeriods:
        cons = shutdown_match_rule(model, g_t[0], g_t[1])
        if cons != None:
            shutdown_match_cons[g_t] = cons


    model._ShutdownMatch = model.addConstrs((shutdown_match_cons[g_t]
                                                for g_t in shutdown_match_cons.keys()),
                                            name = 'ShutdownMatch')

    if add_startup_cost_var:
        model._StartupCost = model.addVars(model._SingleFuelGenerators, model._TimePeriods,
                                lb = -GRB.INFINITY, ub = GRB.INFINITY,
                                name = 'StartupCost')


    def ComputeStartupCost2_rule(m, g, t):
        startup_lags = m._ScaledStartupLags[g]
        startup_costs = m._StartupCosts[g]
        last_startup_cost = startup_costs[-1]

        linear_vars = [m._StartupCost[g, t], m._UnitStart[g, t]]
        linear_coefs = [-1., last_startup_cost]

        for tp in m._ShutdownsByStartups[g, t]:
            for s in m._StartupCostIndices[g]:
                this_lag = startup_lags[s]
                next_lag = startup_lags[s + 1]
                if this_lag <= t - tp < next_lag:
                    linear_vars.append(m._StartupIndicator[g, tp, t])
                    linear_coefs.append(startup_costs[s] - last_startup_cost)
                    break

        return LinExpr(linear_coefs, linear_vars) == 0

    model._ComputeStartupCosts = model.addConstrs(
        (ComputeStartupCost2_rule(model, g, t)
            for g in model._SingleFuelGenerators for t in model._TimePeriods)
                                           ,name='ComputeStartupCost2_rule')
    model.update()
    return model