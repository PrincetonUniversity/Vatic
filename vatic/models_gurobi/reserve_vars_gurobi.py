## system variables and constraints
import math
from gurobipy import tupledict
component_name = 'reserve_vars'
def _add_zero_reserve_hooks(model):

    model._MaximumPowerAvailableAboveMinimum = tupledict({(g, t): model._PowerGeneratedAboveMinimum[g,t] for g in model._ThermalGenerators
     for t in model._TimePeriods})

    model._get_maximum_power_available_above_minimum_lists = model._get_power_generated_above_minimum_lists

    model._MaximumPowerAvailable = tupledict({(g, t): model._PowerGenerated[g,t] for g in model._ThermalGenerators
     for t in model._TimePeriods})

    model._get_maximum_power_available_lists = model._get_power_generated_lists

    model._ReserveProvided = tupledict({(g, t): 0 for g in model._ThermalGenerators
     for t in model._TimePeriods})


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


