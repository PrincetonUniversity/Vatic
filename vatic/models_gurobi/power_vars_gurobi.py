from gurobipy import tupledict, LinExpr, quicksum

def _add_reactive_power_vars(model):
    model._ReactivePowerGenerated = model.addVars(model._ThermalGenerators, model._TimePeriods,
                                                  lb = [model._MinimumReactivePowerOutput[g, t] for g in model._ThermalGenerators
                                                            for t in model._TimePeriods],
                                                  ub = [model._MaximumReactivePowerOutput[g, t] for g in model._ThermalGenerators
                                                            for t in model._TimePeriods], name = 'ReactivePowerGenerated')
    return model

def _add_power_generated_startup_shutdown(model, g, t):
    assert model._InitialTime == 1
    linear_vars, linear_coefs = [model._PowerGeneratedAboveMinimum[g,t], model._UnitOn[g,t]], [1., model._MinimumPowerOutput[g,t]]
    # first, discover if we have startup/shutdown
    # curves in the model
    model_has_startup_shutdown_curves = False
    for s in model._StartupCurve.values():
        if len(s) > 0:
            model_has_startup_shutdown_curves = True
            break
    if not model_has_startup_shutdown_curves:
        for s in model._ShutdownCurve.values():
            if len(s) > 0:
                model_has_startup_shutdown_curves = True
                break

    if model_has_startup_shutdown_curves:
        # check the status vars to see if we're compatible
        # with startup/shutdown curves
        if model._status_vars not in ['garver_2bin_vars', 'garver_3bin_vars', 'garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars']:
            raise RuntimeError(f"Status variable formulation {model._status_vars} is not compatible with startup or shutdown curves")

        startup_curve = model._StartupCurve[g]
        shutdown_curve = model._ShutdownCurve[g]
        time_periods_before_startup = model._TimePeriodsBeforeStartup[g]
        time_periods_since_shutdown = model._TimePeriodsSinceShutdown[g]

        future_startup_past_shutdown_production = 0.
        future_startup_power_index = time_periods_before_startup + model._NumTimePeriods - t
        if future_startup_power_index <= len(startup_curve):
            future_startup_past_shutdown_production += startup_curve[future_startup_power_index]

        past_shutdown_power_index = time_periods_since_shutdown + t
        if past_shutdown_power_index <= len(shutdown_curve):
            future_startup_past_shutdown_production += shutdown_curve[past_shutdown_power_index]

        linear_vars, linear_coefs = model._get_power_generated_lists(model,g,t)
        for startup_idx in range(1, min( len(startup_curve)+1, model._NumTimePeriods+1-t )):
            linear_vars.append(model._UnitStart[g,t+startup_idx])
            linear_coefs.append(startup_curve[startup_idx])
        for shutdown_idx in range(1, min( len(shutdown_curve)+1, t+1 )):
            linear_vars.append(model._UnitStop[g,t-shutdown_idx+1])
            linear_coefs.append(shutdown_curve[shutdown_idx])
        return LinExpr(linear_coefs, linear_vars) + future_startup_past_shutdown_production

        ## if we're here, then we can use 1-bin models
        ## and no need to do the additional work
    return LinExpr(linear_coefs, linear_vars)

# @add_model_attr(component_name, requires={'data_loader': None, 'status_vars': None})
def garver_power_vars(model):
    '''
    The main variable representing generator output is PowerGeneratedAboveMinimum,
    which is exactly what it says. Originally proposed in

    L. L. Garver. Power generation scheduling by integer programming-development
    of theory. Power Apparatus and Systems, Part III. Transactions of the
    American Institute of Electrical Engineers, 81(3): 730–734, April 1962. ISSN
    0097-2460.
    '''

    # NOTE: this should work with any formulation of the status_vars and data_loader currently

    # amount of power produced by each generator above minimum, at each time period.

    model._power_vars='garver_power_vars'
    # ensure the upper bound is ordered here ('101_CT_1', 1), ..., ('101_CT_1', 48), ('101_CT_2', 1), ..., ('101_CT_2', 48),..
    model._PowerGeneratedAboveMinimum = model.addVars(model._ThermalGenerators, model._TimePeriods, lb=0,
                                                      ub=[model._MaximumPowerOutput[g, t]
                                                          - model._MinimumPowerOutput[g, t]
                                                          for g in model._ThermalGenerators for t in
                                                          model._TimePeriods],
                                                      name='PowerGeneratedAboveMinimum')

    model._get_power_generated_above_minimum_lists = lambda m, g, t: ([model._PowerGeneratedAboveMinimum[g, t]], [1.])
    model._get_negative_power_generated_above_minimum_lists = lambda m, g, t: (
    [model._PowerGeneratedAboveMinimum[g, t]], [-1.])

    model._get_power_generated_lists = lambda m, g, t: (
    [model._PowerGeneratedAboveMinimum[g, t], model.UnitOn[g, t]], [1., model._MinimumPowerOutput[g, t]])
    model._get_negative_power_generated_lists = lambda m, g, t: (
    [model._PowerGeneratedAboveMinimum[g, t], model.UnitOn[g, t]], [-1., -model._MinimumPowerOutput[g, t]])

    def power_generated_expr_rule(model, g, t):
        return model._PowerGeneratedAboveMinimum[g, t] + model._MinimumPowerOutput[g, t] * model._UnitOn[g, t]

    model._PowerGenerated = tupledict({(g, t): power_generated_expr_rule(model, g, t) for g in model._ThermalGenerators
                             for t in model._TimePeriods})

    model._PowerGeneratedStartupShutdown = tupledict({(g, t): _add_power_generated_startup_shutdown(model, g, t)
                                         for g in model._ThermalGenerators for t in model._TimePeriods})
    return model














