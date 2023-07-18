"""
Functions for adding the basic status variabkes
"""

from gurobipy import GRB, Model


def garver_3bin_vars(model: Model) -> Model:
    """
    This add the common 3-binary variables per generator per time period.
    One for start, one for stop, and one for on, as originally proposed in

    L. L. Garver. Power generation scheduling by integer programming-development
    of theory. Power Apparatus and Systems, Part III. Transactions of the
    American Institute of Electrical Engineers, 81(3): 730–734, April 1962. ISSN
    0097-2460.
    """

    model._status_vars = "garver_3bin_vars"
    if _is_relaxed(model):
        _add_unit_on_vars(model, True)
        _add_unit_start_vars(model, True)
        _add_unit_stop_vars(model, True)
    else:
        _add_unit_on_vars(model)
        _add_unit_start_vars(model)
        _add_unit_stop_vars(model)
    model.update()
    return model


def _add_unit_on_vars(model, relaxed=False):
    # unit on variables for each generator, at each time period.
    if relaxed:
        model._UnitOn = model.addVars(
            model._ThermalGenerators,
            model._TimePeriods,
            lb=0,
            ub=1,
            name="UnitOn",
        )
    else:
        model._UnitOn = model.addVars(
            model._ThermalGenerators,
            model._TimePeriods,
            vtype=GRB.BINARY,
            name="UnitOn",
        )
    return model


def _add_unit_start_vars(model, relaxed=False):
    # unit start
    if relaxed:
        model._UnitStart = model.addVars(
            model._ThermalGenerators,
            model._TimePeriods,
            lb=0,
            ub=1,
            name="UnitStart",
        )
    else:
        model._UnitStart = model.addVars(
            model._ThermalGenerators,
            model._TimePeriods,
            vtype=GRB.BINARY,
            name="UnitStart",
        )
    return model


def _add_unit_stop_vars(model, relaxed=False):
    # unit stop
    if relaxed:
        model._UnitStop = model.addVars(
            model._ThermalGenerators,
            model._TimePeriods,
            lb=0,
            ub=1,
            name="UnitStop",
        )
    else:
        model._UnitStop = model.addVars(
            model._ThermalGenerators,
            model._TimePeriods,
            vtype=GRB.BINARY,
            name="UnitStop",
        )
    return model


def _is_relaxed(model):
    if hasattr(model, "_relax_binaries") and model._relax_binaries:
        return True
    else:
        return False
