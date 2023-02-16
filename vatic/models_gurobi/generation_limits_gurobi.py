from gurobipy import quicksum
from gurobipy import LinExpr

##LinExpr works faster than quicksum
component_name = 'generation_limits'

## generate new time periods by looking forward
def _get_look_forward_periods(m,g,t,UT_end):
    expr = 0.
    p_max_gt = m._MaximumPowerOutput[g,t]
    if UT_end is not None:
        end = min(UT_end, m._NumTimePeriods-t-1)
    else:
        end = m._NumTimePeriods-t-1
    if end <= 0:
        return end
    ramping_tot = 0
    for i in range(1,end+1):
        shutdown_gi = m._ScaledShutdownRampLimit[g,t+i]
        ramping_tot += m._ScaledNominalRampDownLimit[g,t+i]
        if shutdown_gi + ramping_tot >= p_max_gt:
            ## the prior index what the correct one
            return i-1
    ## then we can go to the end
    return i

## generate new time periods by looking back
def _get_look_back_periods(m,g,t,UT_end):
    expr = 0.
    p_max_gt = m._MaximumPowerOutput[g,t]
    if UT_end is not None:
        end = min(UT_end, t-m._InitialTime)
    else:
        end = t-m._InitialTime
    ramping_tot = 0
    if end <= 0:
        return end
    for i in range(1,end+1):
        startup_gi = m._ScaledStartupRampLimit[g,t-i]
        ramping_tot += m._ScaledNominalRampUpLimit[g,t-i]
        if startup_gi + ramping_tot >= p_max_gt:
            ## the prior index what the correct one
            return i-1
    ## then we can go to the end
    return i

def _get_initial_power_generated_upperbound_lists(m,g,t):
    linear_vars, linear_coefs = m._get_power_generated_lists(m,g,t)
    linear_vars.append(m._UnitOn[g,t])
    linear_coefs.append(-m._MaximumPowerOutput[g,t])
    return linear_vars, linear_coefs

def _get_initial_maximum_power_available_upperbound_lists(m,g,t):
    linear_vars, linear_coefs = m._get_maximum_power_available_lists(m,g,t)
    linear_vars.append(m._UnitOn[g,t])
    linear_coefs.append(-m._MaximumPowerOutput[g,t])
    return linear_vars, linear_coefs


def _CA_lower_limit(model):
    def enforce_generator_output_limits_rule_part_a(m, g, t):
        return m._MinimumPowerOutput[g, t] * m._UnitOn[g, t] <= m._PowerGenerated[g, t]

    model.addConstrs((enforce_generator_output_limits_rule_part_a(model, g, t) for g in model._ThermalGenerators
                     for t in model._TimePeriods), name = 'EnforceGeneratorOutputLimitsPartA')
    model.update()

def _MLR_generation_limits_uptime_1(model, tightened=False):
    ## equations (9), (10) in ME:
    def power_limit_from_start_rule(m, g, t):
        if m._ScaledMinimumUpTime[g] > 1:
            return None
        linear_vars, linear_coefs = _get_initial_maximum_power_available_upperbound_lists(m, g, t)
        linear_vars.append(m._UnitStart[g, t])
        linear_coefs.append(m._MaximumPowerOutput[g, t] - m._ScaledStartupRampLimit[g, t])
        if t == m._NumTimePeriods or not tightened:
            return LinExpr(linear_coefs, linear_vars) <= 0
        coef = max(m._ScaledStartupRampLimit[g, t] - m._ScaledShutdownRampLimit[g, t], 0)
        if coef != 0.:
            linear_vars.append(m._UnitStop[g, t + 1])
            linear_coefs.append(coef)
        return LinExpr(linear_coefs, linear_vars) <= 0

    _power_limit_from_start_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = power_limit_from_start_rule(model, g, t)
            if cons != None:
                _power_limit_from_start_cons[g,t] = cons

    if len(_power_limit_from_start_cons)!= 0:
        model._power_limit_from_start = model.addConstrs((_power_limit_from_start_cons[g_t] for g_t in _power_limit_from_start_cons.keys()), name = '_power_limit_from_start')

    def power_limit_from_stop_rule(m, g, t):
        if m._ScaledMinimumUpTime[g] > 1:
            return None
        if t == m._NumTimePeriods:
            return None  ## This case is handled above
        linear_vars, linear_coefs = _get_initial_maximum_power_available_upperbound_lists(m, g, t)
        linear_vars.append(m._UnitStop[g, t + 1])
        linear_coefs.append(m._MaximumPowerOutput[g, t] - m._ScaledShutdownRampLimit[g, t])
        if not tightened:
            return LinExpr(linear_coefs, linear_vars) <= 0

        coef = max(m._ScaledShutdownRampLimit[g, t] - m._ScaledStartupRampLimit[g, t], 0)
        if coef != 0.:
            linear_vars.append(m._UnitStart[g, t])
            linear_coefs.append(coef)
        return LinExpr(linear_coefs, linear_vars) <= 0

    _power_limit_from_stop_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = power_limit_from_stop_rule(model, g, t)
            if cons != None:
                _power_limit_from_stop_cons[(g,t)] = cons



    if len(_power_limit_from_stop_cons)!= 0:
        model._power_limit_from_stop = model.addConstrs((_power_limit_from_stop_cons[g_t] for g_t in _power_limit_from_stop_cons.keys()), name = '_power_limit_from_stop')

    model.update()


def _pan_guan_generation_limits(model, include_UT_1=True):

    # add the stronger ramp-up based inequality, which is a variant of power_limit_from_start_stop
    def power_limit_from_start_stop_rule(m, g, t):
        if (not include_UT_1) and (m._ScaledMinimumUpTime[g] <= 1):
            return None
        ## time to ramp-up
        Start = m._UnitStart
        Pmax = m._MaximumPowerOutput[g, t]
        SU = m._ScaledStartupRampLimit
        RU = m._ScaledNominalRampUpLimit
        if t == m._NumTimePeriods:
            ##### ^^^ in this case we can squeeze one more into the sum
            linear_vars, linear_coefs = _get_initial_maximum_power_available_upperbound_lists(m, g, t)
            for i in range(0, _get_look_back_periods(m, g, t, m._ScaledMinimumUpTime[g] - 1) + 1):
                linear_vars.append(Start[g, t - i])
                linear_coefs.append(Pmax - SU[g, t - i] - sum(RU[g, t - j] for j in range(1, i + 1)))
            return LinExpr(linear_coefs, linear_vars) <= 0
        else:
            linear_vars, linear_coefs = _get_initial_maximum_power_available_upperbound_lists(m, g, t)
            for i in range(0, _get_look_back_periods(m, g, t, m._ScaledMinimumUpTime[g] - 2) + 1):
                linear_vars.append(Start[g, t - i])
                linear_coefs.append(Pmax - SU[g, t - i] - sum(RU[g, t - j] for j in range(1, i + 1)))
            linear_vars.append(m._UnitStop[g, t + 1])
            linear_coefs.append(Pmax - m._ScaledShutdownRampLimit[g, t])
            return LinExpr(linear_coefs, linear_vars) <= 0

    _power_limit_from_start_stop_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = power_limit_from_start_stop_rule(model, g, t)
            if cons != None:
                _power_limit_from_start_stop_cons[(g,t)] = cons

    if len(_power_limit_from_start_stop_cons)!= 0:
        model._power_limit_from_start_stop = model.addConstrs((_power_limit_from_start_stop_cons[g_t] for g_t in _power_limit_from_start_stop_cons.keys()), name = '_power_limit_from_start_stop')

    model.update()

def _KOW_generation_limits(model):
    ## We'll assume _MLR_generation_limits_uptime_1 and _pan_guan_generation_limits are included
    def max_power_limit_from_starts_rule(m,g,t):
        time_RU = _get_look_back_periods(m,g,t,None)
        if time_RU <= 0:
            return None
        UT = m._ScaledMinimumUpTime[g]
        ## this case is handled better above
        if time_RU <= UT - 2 or t == m.NumTimePeriods:
            return None
        Start = m._UnitStart
        Pmax = m._MaximumPowerOutput[g,t]
        SU = m._ScaledStartupRampLimit
        RU = m._ScaledNominalRampUpLimit

        linear_vars, linear_coefs = _get_initial_maximum_power_available_upperbound_lists(m,g,t)
        for i in range(0, min(time_RU, UT-1, t-m._InitialTime)+1):
            linear_vars.append(Start[g,t-i])
            linear_coefs.append(Pmax - SU[g,t-i] - sum(RU[g,t-j] for j in range(1,i+1)))
        return LinExpr(linear_coefs, linear_vars) <= 0
        '''
        return m._MaximumPowerAvailable[g,t] <= (m._MaximumPowerOutput[g,t]) *m._UnitOn[g,t] \
                                              - sum((m._MaximumPowerOutput[g,t] - m._ScaledStartupRampLimit[g,t-i] - sum(m._ScaledNominalRampUpLimit[g,t-j] for j in range(1,i+1)))*m._UnitStart[g,t-i] \
                                                                for i in range(0, min(time_RU, UT-1, t-value(m._InitialTime))+1) )
        '''

    _max_power_limit_from_starts_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = max_power_limit_from_starts_rule(model, g, t)
            if cons != None:
                _max_power_limit_from_starts_cons[(g,t)] = cons

    if len(_max_power_limit_from_starts_cons)!= 0:
        model._power_limit_from_start_stop = model.addConstrs((_max_power_limit_from_starts_cons[g_t] for g_t in _max_power_limit_from_starts_cons.keys()), name = '_max_power_limit_from_starts')

    ## NOTE: it seems this tightening should really be done on the p^l variables, when they exist
    def power_limit_from_start_stops_rule(m,g,t):
        UT = m._ScaledMinimumUpTime[g]
        SD_time_limit = _get_look_forward_periods(m,g,t,UT-1)
        if SD_time_limit <= 0: ## this is handled by the _MLR_generation_limits or _pan_guan_generation_limits
                               ## and this is needed so this number isn't negative in the computation of SU_time_limit below
            return None
        SU_time_limit = _get_look_back_periods(m,g,t,UT-2-SD_time_limit)

        Start = m._UnitStart
        Stop = m._UnitStop
        Pmax = m._MaximumPowerOutput[g,t]

        SU = m._ScaledStartupRampLimit
        SD = m._ScaledShutdownRampLimit
        RU = m._ScaledNominalRampUpLimit
        RD = m._ScaledNominalRampDownLimit

        linear_vars, linear_coefs = _get_initial_power_generated_upperbound_lists(m,g,t)
        for i in range(0, SD_time_limit+1):
            linear_vars.append(Stop[g,t+i+1])
            linear_coefs.append(Pmax - SD[g,t+i] - sum(RD[g,t+j] for j in range(1,i+1)))
        for i in range(0, SU_time_limit+1):
            linear_vars.append(Start[g,t-i])
            linear_coefs.append(Pmax - SU[g,t-i] - sum(RU[g,t-j] for j in range(1,i+1)))

        full_range = (UT >= max(SU_time_limit,0) + max(SD_time_limit,0) + 2)
        if not full_range:
            i = SU_time_limit+1
            if (t-i) >= m._InitialTime:
                coef =  max((Pmax - SD[g,t+SD_time_limit] - sum(RD[g,t+j] for j in range(1,SD_time_limit+1)) - \
                        (Pmax - SU[g,t-i] - sum(RU[g,t-j] for j in range(1,i+1)))), 0)
                if coef != 0:
                    linear_vars.append(Start[g,t-i])
                    linear_coefs.append(coef)
        return LinExpr(linear_coefs, linear_vars) <= 0

    _power_limit_from_start_stop_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = power_limit_from_start_stops_rule(model, g, t)
            if cons != None:
                _power_limit_from_start_stop_cons[(g,t)] = cons

    if len(_power_limit_from_start_stop_cons)!= 0:
        model._power_limit_from_start_stop = model.addConstrs((_power_limit_from_start_stop_cons[g_t] for g_t in _power_limit_from_start_stop_cons.keys()), name = '_power_limit_from_start_stop')
    model.update()

def pan_guan_gentile_KOW_generation_limits(model):
    model._generation_limits = 'pan_guan_gentile_KOW_generation_limits'
    if model._power_vars in ['garver_power_vars', ]:
        pass
    else:
        _CA_lower_limit(model)
    # add the strenghtened MLR generation limits fot UT==1 generators
    _MLR_generation_limits_uptime_1(model, True)
    _pan_guan_generation_limits(model, False)
    _KOW_generation_limits(model)
    return model