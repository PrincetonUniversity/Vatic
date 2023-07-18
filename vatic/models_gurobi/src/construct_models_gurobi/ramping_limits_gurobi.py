"""
Add ramping up and down limits
"""

from gurobipy import LinExpr, Model

generation_limits_w_startup_shutdown = [
    "MLR_generation_limits",
    "gentile_generation_limits",
    "pan_guan_gentile_generation_limits",
    "pan_guan_gentile_KOW_generation_limits",
]


def damcikurt_ramping(model: Model) -> Model:
    """
    Add ramping limits based on Equations (3) and (18) from
    Pelin Damci-Kurt, Simge Kucukyavuz, Deepak Rajan, and Alper Atamturk. A
    polyhedral study of production ramping. Mathematical Programming,
    158(1-2):175–205, 2016.
    """
    _damcikurt_basic_ramping(model)
    return model


def _damcikurt_basic_ramping(model):
    # NOTE: with the expression MaximumPowerAvailableAboveMinimum and PowerGeneratedAboveMinimum,
    #       these constraints are expressed as needed, there's no cancelation even though we end
    #       up using these expressions
    def enforce_max_available_ramp_up_rates_rule(m, g, t):
        if _ramp_up_not_needed(m, g, t):
            return None
        if t == m._InitialTime:
            # if the unit was on in t0, then it's m.PowerGeneratedT0[g] >= m.MinimumPowerOutput[g], and m.UnitOnT0 == 1
            # if not, then m.UnitOnT0[g] == 0 and so (m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]) * m.UnitOnT0[g] is 0
            # assume m.MinimumPowerOutput[g,T0] == 0
            (
                linear_vars_power_t,
                linear_coefs_power_t,
            ) = m._get_maximum_power_available_above_minimum_lists(m, g, t)
            rhs_linear_vars = [m._UnitOn[g, t], m._UnitStart[g, t]]
            rhs_neg_linear_coefs = [
                -m._ScaledNominalRampUpLimit[g, t]
                - 0
                + m._MinimumPowerOutput[g, t],
                -m._ScaledStartupRampLimit[g, t]
                + 0
                + m._ScaledNominalRampUpLimit[g, t],
            ]

            linear_vars = [*linear_vars_power_t, *rhs_linear_vars]
            linear_coefs = [*linear_coefs_power_t, *rhs_neg_linear_coefs]

            RHS = m._PowerGeneratedT0[g]
            return LinExpr(linear_coefs, linear_vars) <= RHS

        else:
            (
                linear_vars_power_t,
                linear_coefs_power_t,
            ) = m._get_maximum_power_available_above_minimum_lists(m, g, t)
            (
                linear_vars_power_t_1,
                linear_coefs_power_t_1,
            ) = m._get_negative_power_generated_above_minimum_lists(
                m, g, t - 1
            )

            rhs_linear_vars = [m._UnitOn[g, t], m._UnitStart[g, t]]
            rhs_neg_linear_coefs = [
                -m._ScaledNominalRampUpLimit[g, t]
                - m._MinimumPowerOutput[g, t - 1]
                + m._MinimumPowerOutput[g, t],
                -m._ScaledStartupRampLimit[g, t]
                + m._MinimumPowerOutput[g, t - 1]
                + m._ScaledNominalRampUpLimit[g, t],
            ]

            linear_vars = [
                *linear_vars_power_t,
                *linear_vars_power_t_1,
                *rhs_linear_vars,
            ]
            linear_coefs = [
                *linear_coefs_power_t,
                *linear_coefs_power_t_1,
                *rhs_neg_linear_coefs,
            ]

            return LinExpr(linear_coefs, linear_vars) <= 0

    enforce_max_available_ramp_up_rates_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = enforce_max_available_ramp_up_rates_rule(model, g, t)
            if cons != None:
                enforce_max_available_ramp_up_rates_cons[g, t] = cons
    if len(enforce_max_available_ramp_up_rates_cons) != 0:
        model._EnforceMaxAvailableRampUpRates = model.addConstrs(
            (
                enforce_max_available_ramp_up_rates_cons[g_t]
                for g_t in enforce_max_available_ramp_up_rates_cons.keys()
            ),
            name="EnforceMaxAvailableRampUpRates",
        )

    def enforce_ramp_down_limits_rule(m, g, t):
        if _ramp_down_not_needed(m, g, t):
            return None
        if t == m._InitialTime:
            # assume m.MinimumPowerOutput[g,T0] == 0
            (
                linear_vars_power_t,
                linear_coefs_power_t,
            ) = m._get_power_generated_above_minimum_lists(m, g, t)

            lhs_linear_vars = [m._UnitStop[g, t]]
            lhs_neg_linear_coefs = [
                (
                        m._ScaledShutdownRampLimitT0[g]
                        - m._MinimumPowerOutput[g, t]
                        - m._ScaledNominalRampDownLimit[g, t]
                )
            ]

            linear_vars = [*linear_vars_power_t, *lhs_linear_vars]
            linear_coefs = [*linear_coefs_power_t, *lhs_neg_linear_coefs]

            LHS = (
                    m._PowerGeneratedT0[g]
                    - (
                            m._ScaledNominalRampDownLimit[g, t]
                            + m._MinimumPowerOutput[g, t]
                            - 0
                    )
                    * m._UnitOnT0[g]
            )

            return LinExpr(linear_coefs, linear_vars) >= LHS
        else:
            (
                linear_vars_power_t,
                linear_coefs_power_t,
            ) = m._get_power_generated_above_minimum_lists(m, g, t)
            (
                linear_vars_power_t_1,
                linear_coefs_power_t_1,
            ) = m._get_negative_power_generated_above_minimum_lists(
                m, g, t - 1
            )

            lhs_linear_vars = [m._UnitOn[g, t - 1], m._UnitStop[g, t]]
            lhs_neg_linear_coefs = [
                m._ScaledNominalRampDownLimit[g, t]
                + m._MinimumPowerOutput[g, t]
                - m._MinimumPowerOutput[g, t - 1],
                m._ScaledShutdownRampLimit[g, t - 1]
                - m._MinimumPowerOutput[g, t]
                - m._ScaledNominalRampDownLimit[g, t],
            ]
            linear_vars = [
                *linear_vars_power_t,
                *linear_vars_power_t_1,
                *lhs_linear_vars,
            ]
            linear_coefs = [
                *linear_coefs_power_t,
                *linear_coefs_power_t_1,
                *lhs_neg_linear_coefs,
            ]

            return LinExpr(linear_coefs, linear_vars) >= 0

    enforce_ramp_down_limits_cons = {}
    for g in model._ThermalGenerators:
        for t in model._TimePeriods:
            cons = enforce_ramp_down_limits_rule(model, g, t)
            if cons != None:
                enforce_ramp_down_limits_cons[g, t] = cons
    if len(enforce_ramp_down_limits_cons) != 0:
        model._EnforceScaledNominalRampDownLimits = model.addConstrs(
            (
                enforce_ramp_down_limits_cons[g_t]
                for g_t in enforce_ramp_down_limits_cons.keys()
            ),
            name="EnforceScaledNominalRampDownLimits",
        )

    model.update()
    return model


def _ramp_up_not_needed(m, g, t):
    """
    Decide if ramping up constraints are even added to the model
    based on some state data
    """

    if m._generation_limits not in generation_limits_w_startup_shutdown:
        return False
    if t == m._InitialTime:
        # no need for ramping constraints if the generator is off, and
        # we're enforcing startup/shutdown elsewhere
        if not m._UnitOnT0[g]:
            return True
        if m._ScaledNominalRampUpLimit[g, t] >= (
                m._MaximumPowerOutput[g, t] - m._PowerGeneratedT0[g]
        ):
            # the generator can get all the way to max at the first time period
            return True
        return False
    if m._ScaledNominalRampUpLimit[g, t] >= (
            m._MaximumPowerOutput[g, t] - m._MinimumPowerOutput[g, t - 1]
    ):
        return True
    return False


def _ramp_down_not_needed(m, g, t):
    if t == m._InitialTime:
        if not m._enforce_t1_ramp_rates:
            return True
        # if the unit is off, we don't need ramp down constraints
        if not m._UnitOnT0[g]:
            return True
        if m._ScaledNominalRampDownLimit[g, t] < (
                m._PowerGeneratedT0[g] - m._MinimumPowerOutput[g, t]
        ):
            return False
        # if this and the opposite of the above condition are true,
        # we don't need an inital ramp-down inequality
        if m._ScaledShutdownRampLimit[g, t] >= m._PowerGeneratedT0[g]:
            return True
        return False
    if m._generation_limits not in generation_limits_w_startup_shutdown:
        return False
    if m._ScaledNominalRampDownLimit[g, t] >= (
            m._MaximumPowerOutput[g, t - 1] - m._MinimumPowerOutput[g, t]
    ):
        return True
    return False
