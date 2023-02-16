import math
component_name = 'non_dispatchable_vars'

## file non-dispatchable power -- determined by .dat file
def file_non_dispatchable_vars(model):
    # assume wind can be curtailed, then wind power is a decision variable
    lb_ = []
    ub_ = []
    for n in model._AllNondispatchableGenerators:
            for t in model._TimePeriods:
                lb_.append(model._MinNondispatchablePower[n, t])
                ub_.append(model._MaxNondispatchablePower[n, t])

    model._NondispatchablePowerUsed = model.addVars(model._AllNondispatchableGenerators, model._TimePeriods,
                                                    lb = lb_, ub = ub_)
    return model

