"""Add reserve constraints and reserve shortfall variables"""
from .reserve_vars_gurobi import check_reserve_requirement

from gurobipy import LinExpr, quicksum, Model


def MLR_reserve_constraints(model: Model) -> Model:
    """
    Add reserve constraints with slacks given by equation (5) in
    G. Morales-Espana, J. M. Latorre, and A. Ramos. Tight and compact MILP
    formulation for the thermal unit commitment problem. IEEE Transactions on
    Power Systems, 28(4):4897–4908, 2013.
    """

    if not check_reserve_requirement(model):
        _add_reserve_shortfall(model, fixed=True)
        return

    _add_reserve_shortfall(model)
    # ensure there is sufficient maximal power output available to meet both the
    # demand and the spinning reserve requirements in each time period.
    # encodes Constraint 3 in Carrion and Arroyo.

    # IMPT: In contrast to power balance, reserves are (1) not per-bus and (2) expressed in terms of
    #       maximum power available, and not actual power generated.
    _MLR_reserve_constraint(model)

    model.update()
    return model


def _MLR_reserve_constraint(model):
    """
    This is the reserve requirement with slacks given by equation (5) in

    G. Morales-Espana, J. M. Latorre, and A. Ramos. Tight and compact MILP
    formulation for the thermal unit commitment problem. IEEE Transactions on
    Power Systems, 28(4):4897–4908, 2013.
    """

    def enforce_reserve_requirements_rule(m, t):
        linear_vars = list(
            m._ReserveProvided[g, t] for g in m._ThermalGenerators
        )
        linear_vars.append(m._ReserveShortfall[t])
        linear_coefs = [1.0] * len(linear_vars)
        return LinExpr(linear_coefs, linear_vars) >= m._ReserveRequirement[t]

    model._EnforceReserveRequirements = model.addConstrs(
        (
            enforce_reserve_requirements_rule(model, t)
            for t in model._TimePeriods
        ),
        name="EnforceReserveRequirements",
    )


def CA_reserve_constraints(model: Model) -> Model:
    """
    Add reserve constraints with slacks given by equation (3) in
    Carrion, M. and Arroyo, J. (2006) A Computationally Efficient Mixed-Integer
    Liner Formulation for the Thermal Unit Commitment Problem. IEEE Transactions
    on Power Systems, Vol. 21, No. 3, Aug 2006.
    """

    if not check_reserve_requirement(model):
        _add_reserve_shortfall(model, fixed=True)
        return
    _add_reserve_shortfall(model)

    # ensure there is sufficient maximal power output available to meet both the
    # demand and the spinning reserve requirements in each time period.
    # encodes Constraint 3 in Carrion and Arroyo.

    # IMPT: In contrast to power balance, reserves are (1) not per-bus and (2) expressed in terms of
    #       maximum power available, and not actual power generated.

    def enforce_reserve_requirements_rule(m, t):
        # linear_expr = (quicksum(m._MaximumPowerAvailable.select('*', t))
        #             + quicksum(m._NondispatchablePowerUsed.select('*', t))
        #             +  quicksum(m._LoadGenerateMismatch.select('*', t))
        #             + m._ReserveShortfall[t])

        linear_expr = (
                m._MaximumPowerAvailable_atT[t]
                + quicksum(m._NondispatchablePowerUsed.select("*", t))
                + quicksum(m._LoadGenerateMismatch.select("*", t))
                + m._ReserveShortfall[t]
        )

        if hasattr(model, "_PowerOutputStorage"):
            linear_expr += quicksum(m._PowerOutputStorage.select("*", t))

        if hasattr(model, "_PowerInputStorage"):
            linear_expr -= quicksum(m._PowerInputStorage.select("*", t))

        return linear_expr >= (m._TotalDemand[t] + m._ReserveRequirement[t])

    model._EnforceReserveRequirements = model.addConstrs(
        (
            enforce_reserve_requirements_rule(model, t)
            for t in model._TimePeriods
        ),
        name="EnforceReserveRequirements",
    )

    model.update()
    return model


def _add_reserve_shortfall(model, fixed=False):
    """Add reserve shortfall variables"""
    if fixed:
        model._ReserveShortfall = {t: 0 for t in model._TimePeriods}
    else:
        # the reserve shortfall can't be more than the reserve requirement in any given time period.
        model._ReserveShortfall = model.addVars(
            model._TimePeriods,
            lb=0,
            ub=[model._ReserveRequirement[t] for t in model._TimePeriods],
            name="ReserveShortfall",
        )
