from gurobipy import GRB

def _is_relaxed(model):
    if hasattr(model, 'relax_binaries') and model._relax_binaries:
        return True
    else:
        return False

# @add_model_attr(component_name, requires = {'data_loader': None} )
def garver_3bin_vars(model):
    '''
    This add the common 3-binary variables per generator per time period.
    One for start, one for stop, and one for on, as originally proposed in

    L. L. Garver. Power generation scheduling by integer programming-development
    of theory. Power Apparatus and Systems, Part III. Transactions of the
    American Institute of Electrical Engineers, 81(3): 730–734, April 1962. ISSN
    0097-2460.

    '''

    if _is_relaxed(model):
        _add_unit_on_vars(model, True)
        _add_unit_start_vars(model, True)
        _add_unit_stop_vars(model, True)
    else:
        _add_unit_on_vars(model)
        _add_unit_start_vars(model)
        _add_unit_stop_vars(model)

    return

def _add_unit_on_vars(model, relaxed=False):
    # indicator variables for each generator, at each time period.
    if relaxed:
        UnitOn = model.addVars(model._ThermalGenerators, model._TimePeriods, lb = 0, ub =1)
    else:
        UnitOn = model.addVars(model._ThermalGenerators, model._TimePeriods, vtype=GRB.BINARY)

def _add_unit_start_vars(model, relaxed=False):
    # unit start
    if relaxed:
        UnitStart = model.addVars(model._ThermalGenerators, model._TimePeriods, lb = 0, ub =1)
    else:
        UnitStart = model.addVars(model._ThermalGenerators, model._TimePeriods, vtype=GRB.BINARY)

def _add_unit_stop_vars(model, relaxed=False):
    if relaxed:
        UnitStop = model.addVars(model._ThermalGenerators, model._TimePeriods, lb = 0, ub =1)
    else:
        UnitStop = model.addVars(model._ThermalGenerators, model._TimePeriods, vtype=GRB.BINARY)

