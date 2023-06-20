## system variables and constraints
import math
from gurobipy import tupledict, LinExpr
component_name = 'reserve_vars'

def check_reserve_requirement(model):
    system = model._model_data.data['system']
    return ('reserve_requirement' in system)

def _add_zero_reserve_hooks(model):

    model._MaximumPowerAvailableAboveMinimum = tupledict({(g, t): model._PowerGeneratedAboveMinimum[g,t]
                                                          for g in model._ThermalGenerators for t in model._TimePeriods})

    model._get_maximum_power_available_above_minimum_lists = model._get_power_generated_above_minimum_lists

    model._MaximumPowerAvailable = tupledict({(g, t): model._PowerGenerated[g,t]
                                                          for g in model._ThermalGenerators for t in model._TimePeriods})

    model._get_maximum_power_available_lists = model._get_power_generated_lists

    model._ReserveProvided = tupledict({(g, t): 0 for g in model._ThermalGenerators for t in model._TimePeriods})

def check_reserve_requirement(model):
    system = model._model_data.data['system']
    return ('reserve_requirement' in system)

def garver_power_avail_vars(model):
    '''
    These never appear in Garver's paper, but they are an adaption of the
    idea from the Carrion-Arroyo paper for maximum power available
    to consider maximum power available over minimum
    '''

    model._reserve_vars='garver_power_avail_vars'
    ## only add reserves if the user specified them
    if not check_reserve_requirement(model):
        _add_zero_reserve_hooks(model)
        return

    # maximum power output above minimum for each generator, at each time period.
    # the upper bound is amount of power produced by each generator above minimum, at each time period.
    model._MaximumPowerAvailableAboveMinimum = model.addVars(model._ThermalGenerators, model._TimePeriods, lb=0,
                                                      ub=[model._MaximumPowerOutput[g, t] - model._MinimumPowerOutput[
                                                          g, t] \
                                                          for g in model._ThermalGenerators for t in
                                                          model._TimePeriods],
                                                      name='MaximumPowerAvailableAboveMinimum')

    model._get_maximum_power_available_above_minimum_lists = lambda m,g,t : ([m._MaximumPowerAvailableAboveMinimum[g,t]], [1.])

    ## Note: thes only get used in system balance constraints
    model._get_maximum_power_available_lists = lambda m,g,t : ([m._MaximumPowerAvailableAboveMinimum[g,t], m._UnitOn[g,t]], [1., m._MinimumPowerOutput[g,t]])

    model._MaximumPowerAvailable = tupledict({(g, t): model._MaximumPowerAvailableAboveMinimum[g,t]+model._UnitOn[g,t]*model._MinimumPowerOutput[g,t]
               for g in model._ThermalGenerators for t in model._TimePeriods})

    # m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t] <= m.MaximumPowerOutput[g] * m.UnitOn[g, t]
    # BK -- first <= now handled by bounds
    model._EnforceGeneratorOutputLimitsPartB = model.addConstrs((model._PowerGeneratedAboveMinimum[g, t]-
                                                model._MaximumPowerAvailableAboveMinimum[g, t] <= 0 \
                                                for g in model._ThermalGenerators for t in model._TimePeriods), name = 'EnforceGeneratorOutputLimitsPartB')
    ## BK -- for reserve pricing
    _get_reserves_provided_lists = lambda m, g, t: (
    [m._MaximumPowerAvailableAboveMinimum[g, t], m._PowerGeneratedAboveMinimum[g, t]], [1., -1.])

    model._ReserveProvided = tupledict({(g, t): model._MaximumPowerAvailableAboveMinimum[g,t] - model._PowerGeneratedAboveMinimum[g,t]
                                        for g in model._ThermalGenerators for t in model._TimePeriods})

    model.update()
    return model

def MLR_reserve_vars(model):
    '''
    Reserves provided variables as in

    G. Morales-Espana, J. M. Latorre, and A. Ramos. Tight and compact MILP
    formulation for the thermal unit commitment problem. IEEE Transactions on
    Power Systems, 28(4):4897–4908, 2013.

    '''

    model._reserve_vars = 'MLR_reserve_vars'
    ## only add reserves if the user specified them
    if not check_reserve_requirement(model):
        _add_zero_reserve_hooks(model)
        return

    # amount of power produced by each generator above minimum, at each time period.
    # variable for reserves offered
    model._ReserveProvided = model.addVars(model._ThermalGenerators, model._TimePeriods, lb=0,
                  ub=[model._MaximumPowerOutput[g, t] -
                      model._MinimumPowerOutput[g, t] \
                      for g in model._ThermalGenerators for t in model._TimePeriods],
                  name='ReserveProvided')

    ## Note: thes only get used in system balance constraints
    def _get_maximum_power_available_lists(m,g,t):
        linear_vars, linear_coefs = m._get_power_generated_lists(m,g,t)
        linear_vars.append(m._ReserveProvided[g,t])
        linear_coefs.append(1.)
        return linear_vars, linear_coefs

    model._MaximumPowerAvailable = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            linear_vars, linear_coefs = _get_maximum_power_available_lists(model, g, t)
            model._MaximumPowerAvailable[(g, t)] = LinExpr(linear_coefs, linear_vars)

    model._get_maximum_power_available_lists = _get_maximum_power_available_lists

    ## Note: thes only get used in system balance constraints
    def _get_maximum_power_available_above_minimum_lists(m,g,t):
        linear_vars, linear_coefs = m._get_power_generated_above_minimum_lists(m,g,t)
        linear_vars.append(m._ReserveProvided[g,t])
        linear_coefs.append(1.)
        return linear_vars, linear_coefs

    model._MaximumPowerAvailableAboveMinimum = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            linear_vars, linear_coefs = _get_maximum_power_available_above_minimum_lists(model, g, t)
            model._MaximumPowerAvailableAboveMinimum[(g, t)] = LinExpr(linear_coefs, linear_vars)

    model._get_maximum_power_available_above_minimum_lists = _get_maximum_power_available_above_minimum_lists

    model.update()
    return model

