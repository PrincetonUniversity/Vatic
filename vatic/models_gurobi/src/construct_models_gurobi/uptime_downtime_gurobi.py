from gurobipy import LinExpr, Model

"""Add uptime and downtime constraints"""


def rajan_takriti_UT_DT(model: Model) -> Model:
    """
    Uptime/downtime constraints (3) and (4) from

    D. Rajan and S. Takriti. Minimum up/down polytopes of the unit commitment
    problem with start-up costs. IBM Res. Rep, 2005.
    """

    _add_fixed_and_initial(model)

    # up-time constraints

    def uptime_rule(m, g, t):
        if t < m._ScaledMinimumUpTime[g]:
            return None
        linear_vars = [
            m._UnitStart[g, i]
            for i in range(t - m._ScaledMinimumUpTime[g] + 1, t + 1)
        ]
        linear_coefs = [1.0] * len(linear_vars)
        linear_vars.append(m._UnitOn[g, t])
        linear_coefs.append(-1.0)
        return LinExpr(linear_coefs, linear_vars) <= 0

    uptime_rule_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = uptime_rule(model, g, t)
            if cons != None:
                uptime_rule_cons[g, t] = cons

    model._UpTime = model.addConstrs(
        (uptime_rule_cons[g_t] for g_t in uptime_rule_cons.keys()),
        name="UpTime",
    )

    # down-time constraints #

    def downtime_rule(m, g, t):
        if t < m._ScaledMinimumDownTime[g]:
            return None
        linear_vars = [
            m._UnitStop[g, i]
            for i in range(t - m._ScaledMinimumDownTime[g] + 1, t + 1)
        ]
        linear_coefs = [1.0] * len(linear_vars)
        linear_vars.append(m._UnitOn[g, t])
        linear_coefs.append(1.0)
        return LinExpr(linear_coefs, linear_vars) <= 1

    downtime_rule_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = downtime_rule(model, g, t)
            if cons != None:
                downtime_rule_cons[g, t] = cons

    model._DownTime = model.addConstrs(
        (downtime_rule_cons[g_t] for g_t in downtime_rule_cons.keys()),
        name="DownTime",
    )

    _3bin_logic(model)
    model.update()
    return model


def _3bin_logic(model):
    initial_time = model._InitialTime

    def logical_rule(m, g, t):
        if t == initial_time:
            linear_vars = [
                m._UnitOn[g, t],
                m._UnitStart[g, t],
                m._UnitStop[g, t],
            ]
            linear_coefs = [1.0, -1.0, 1.0]
            rhs = m._UnitOnT0[g]

            return LinExpr(linear_coefs, linear_vars) == rhs

        linear_vars = [
            m._UnitOn[g, t],
            m._UnitOn[g, t - 1],
            m._UnitStart[g, t],
            m._UnitStop[g, t],
        ]
        linear_coefs = [1.0, -1, -1.0, 1.0]
        rhs = 0
        return LinExpr(linear_coefs, linear_vars) == rhs

    model._Logical = model.addConstrs(
        (
            logical_rule(model, g, t)
            for g in model._ThermalGenerators
            for t in model._TimePeriods
        ),
        name="logical",
    )


def _add_initial(model):
    # constraint due to initial conditions.
    def enforce_up_time_constraints_initial(m, g):
        if m._InitialTimePeriodsOnLine[g] == 0:
            return
        for t in range(
                m._TimePeriods[0],
                m._InitialTimePeriodsOnLine[g] + m._TimePeriods[0],
        ):
            if m._status_vars == "ALS_state_transition_vars":
                m.addConstr(
                    m._UnitStayOn[g, t] == 1,
                    name="InitialAssetStateConstr_{}_{}".format(g, t),
                )

            else:
                m.addConstr(
                    m._UnitOn[g, t] == 1,
                    name="InitialAssetStateConstr_{}_{}".format(g, t),
                )

    for g in model._ThermalGenerators:
        enforce_up_time_constraints_initial(model, g)

    # constraint due to initial conditions.
    def enforce_down_time_constraints_initial(m, g):
        if m._InitialTimePeriodsOffLine[g] == 0:
            return
        for t in range(
                m._TimePeriods[0],
                m._InitialTimePeriodsOffLine[g] + m._TimePeriods[0],
        ):
            if m._status_vars == "ALS_state_transition_vars":
                m.addConstr(
                    m._UnitStayOn[g, t] == 0,
                    name="InitialAssetStateConstr_{}_{}".format(g, t),
                )
            else:
                m.addConstr(
                    m._UnitOn[g, t] == 0,
                    name="InitialAssetStateConstr_{}_{}".format(g, t),
                )

    for g in model._ThermalGenerators:
        enforce_down_time_constraints_initial(model, g)


def _add_fixed_and_initial(model):
    # Fixed commitment constraints
    def enforce_fixed_commitments_rule(m, g, t):
        if m._FixedCommitment[g, t] is not None:
            if m._status_vars == "ALS_state_transition_vars":
                m._UnitStayOn[g, t].lb = m._FixedCommitment[g, t]
                m._UnitStayOn[g, t].ub = m._FixedCommitment[g, t]
            else:
                m.addConstr(
                    m._UnitOn[g, t] == m._FixedCommitment[g, t],
                    name="FixedCommitment_{}_{}".format(g, t),
                )

    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            enforce_fixed_commitments_rule(model, g, t)

    _add_initial(model)
