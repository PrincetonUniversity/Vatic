"""
Add non dispatchabe power used variables
"""

from gurobipy import Model


def file_non_dispatchable_vars(model: Model) -> Model:
    """Add non dispatchable power variables"""
    lb_ = []
    ub_ = []
    for n in model._AllNondispatchableGenerators:
        for t in model._TimePeriods:
            lb_.append(model._MinNondispatchablePower[n, t])
            ub_.append(model._MaxNondispatchablePower[n, t])

    model._NondispatchablePowerUsed = model.addVars(
        model._AllNondispatchableGenerators,
        model._TimePeriods,
        lb=lb_,
        ub=ub_,
        name="NondispatchablePowerUsed",
    )
    model.update()
    return model
