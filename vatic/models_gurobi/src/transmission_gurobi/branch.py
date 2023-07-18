"""
Declarations for the modelling components
typically used for transmission lines
"""

from gurobipy import LinExpr, tupledict

from egret.model_library.defn import FlowType, ApproximationType


def declare_expr_pf(model, index_set):
    model._pf = {i: LinExpr() for i in index_set}


def get_power_flow_interface_expr_ptdf(
        model, interface_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None
):
    """
    Create a gurobipy power flow expression from PTDF matrix for an interface
    """
    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.0
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.0

    const = PTDF.get_interface_const(interface_name)
    max_coef = PTDF.get_interface_ptdf_abs_max(interface_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol * max_coef)

    m_p_nw = model._p_nw_tm
    # if model.p_nw is Var, we can use LinearExpression
    # to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_interface_ptdf_iterator(interface_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    expr = LinExpr(coef_list, var_list) + const

    return expr


def get_power_flow_expr_ptdf_approx(
        model, branch_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None
):
    """
    Create a gurobipy power flow expression from PTDF matrix
    """

    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.0
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.0

    const = PTDF.get_branch_const(branch_name)

    max_coef = PTDF.get_branch_ptdf_abs_max(branch_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol * max_coef)
    m_p_nw = model._p_nw_tm
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_branch_ptdf_iterator(branch_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    expr = LinExpr(coef_list, var_list) + const
    return expr


def generate_thermal_bounds(
        pf, llimit, ulimit, neg_slack=None, pos_slack=None
):
    """
    Create a constraint for thermal limits on a line given the power flow
    expression or variable pf, a lower limit llimit, a uppder limit ulimit,
    and the negative slack variable, neg_slack, (None if not needed) and
    positive slack variable, pos_slack, (None if not needed) added to this
    constraint.
    """
    expr = pf
    if neg_slack is not None:
        expr += neg_slack
    if pos_slack is not None:
        expr -= pos_slack
    return (llimit, expr, ulimit)


def declare_ineq_p_branch_thermal_bounds(
        model,
        parent_model,
        period,
        index_set,
        branches,
        p_thermal_limits,
        approximation_type=ApproximationType.BTHETA,
        slacks=False,
        slack_cost_expr=None,
):
    """
    Create an inequality constraint for the branch thermal limits
    based on the power variables or expressions at the branch level
    """
    # con_set = decl.declare_set('_con_ineq_p_branch_thermal_bounds',
    #                            model=model, index_set=index_set)
    # flag for if slacks are on the model
    m = parent_model

    if slacks:
        if not hasattr(model, "_pf_slack_pos"):
            raise Exception(
                "No positive slack branch variables on model, but slacks=True"
            )
        if not hasattr(model, "_pf_slack_neg"):
            raise Exception(
                "No negative slack branch variables on model, but slacks=True"
            )
        if slack_cost_expr is None:
            raise Exception("No cost expression for slacks, but slacks=True")

    m._ineq_pf_branch_thermal_ub = tupledict()
    m._ineq_pf_branch_thermal_lb = tupledict()
    if (
            approximation_type == ApproximationType.BTHETA
            or approximation_type == ApproximationType.PTDF
    ):
        for branch_name in index_set:
            limit = p_thermal_limits[branch_name]
            if limit is None:
                continue

            if slacks and branch_name in model._pf_slack_neg.index_set():
                assert branch_name in model.pf_slack_pos.index_set()
                neg_slack = model._pf_slack_neg[branch_name]
                pos_slack = model._pf_slack_pos[branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (
                        uc_model._TimePeriodLengthHours
                        * uc_model._BranchLimitPenalty[branch_name]
                        * (neg_slack + pos_slack)
                )
                assert len(model._pf_slack_pos) == len(model._pf_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub = generate_thermal_bounds(
                model._pf[branch_name], -limit, limit, neg_slack, pos_slack
            )
            m._ineq_pf_branch_thermal_ub[(branch_name, period)] = m.addConstr(
                (expr <= ub),
                name="ineq_pf_branch_thermal_upper_bounds_branch[{}]_period[{}]".format(
                    branch_name, period
                ),
            )
            m._ineq_pf_branch_thermal_lb[(branch_name, period)] = m.addConstr(
                (expr >= lb),
                name="ineq_pf_branch_thermal_lower_bounds_branch[{}]_period[{}]".format(
                    branch_name, period
                ),
            )


def declare_ineq_p_contingency_branch_thermal_bounds(
        model,
        parent_model,
        period,
        index_set,
        pc_thermal_limits,
        approximation_type=ApproximationType.PTDF,
        slacks=False,
        slack_cost_expr=None,
):
    """
    Create an inequality constraint for the branch thermal limits
    based on the power variables or expressions att he contigency level
    """
    # # flag for if slacks are on the model
    # if slacks:
    #     if not hasattr(model, 'pfc_slack_pos'):
    #         raise Exception('No positive slack branch variables on model, but slacks=True')
    #     if not hasattr(model, 'pfc_slack_neg'):
    #         raise Exception('No negative slack branch variables on model, but slacks=True')
    #     if slack_cost_expr is None:
    #         raise Exception('No cost expression for slacks, but slacks=True')
    #
    # m._ineq_pf_contingency_branch_thermal_bounds = pe.Constraint(index_set)
    m = parent_model
    if (
            approximation_type == ApproximationType.BTHETA
            or approximation_type == ApproximationType.PTDF
    ):
        for contingency_name, branch_name in index_set:
            limit = pc_thermal_limits[branch_name]
            if limit is None:
                continue

            if (
                    slacks
                    and (contingency_name, branch_name)
                    in model._pfc_slack_neg.index_set()
            ):
                assert (
                           contingency_name,
                           branch_name,
                       ) in model._pfc_slack_pos.index_set()
                neg_slack = model._pfc_slack_neg[contingency_name, branch_name]
                pos_slack = model._pfc_slack_pos[contingency_name, branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (
                        uc_model._TimePeriodLengthHours
                        * uc_model._ContingencyLimitPenalty
                        * (neg_slack + pos_slack)
                )
                assert len(model._pfc_slack_pos) == len(model._pfc_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub = generate_thermal_bounds(
                model._pfc[contingency_name, branch_name],
                -limit,
                limit,
                neg_slack,
                pos_slack,
            )
            m.addConstr(
                (expr <= ub),
                name="ineq_pf_interface_upper_bounds_branch[{}]_peiord[{}]".format(
                    branch_name, period
                ),
            )
            m.addConstr(
                (-expr <= -lb),
                name="ineq_pf_interface_lower_bounds_branch[{}]_period[{}]".format(
                    branch_name, period
                ),
            )


def declare_eq_branch_power_ptdf_approx(
        model,
        parent_model,
        period,
        index_set,
        PTDF,
        rel_ptdf_tol=None,
        abs_ptdf_tol=None,
):
    """
    Create the equality constraints or expressions for power (from PTDF
    approximation) at the branch level
    """

    m = parent_model

    # con_set = decl.declare_set("_con_eq_branch_power_ptdf_approx_set", model, index_set)

    # if pf_is_var:
    #     m._eq_pf_branch = pe.Constraint(con_set)
    # else:
    #     if not isinstance(m._pf, pe.Expression):
    #         raise Exception("Unrecognized type for m._pf", m._pf.pprint())

    for branch_name in index_set:
        expr = get_power_flow_expr_ptdf_approx(
            m,
            branch_name,
            PTDF,
            rel_ptdf_tol=rel_ptdf_tol,
            abs_ptdf_tol=abs_ptdf_tol,
        )

        m.addConstr(
            (model._pf[branch_name] == expr),
            name="_eq_pf_branch[{}]_period[{}]".format(branch_name, period),
        )


def declare_eq_interface_power_ptdf_approx(
        model,
        parent_model,
        period,
        index_set,
        PTDF,
        rel_ptdf_tol=None,
        abs_ptdf_tol=None,
):
    """
    Create equality constraints or expressions for power (from PTDF
    approximation) at the interface level
    """

    m = parent_model
    # con_set = decl.declare_set("_con_eq_interface_power_ptdf_approx_set", model, index_set)
    #
    # pfi_is_var = isinstance(m._pfi, pe.Var)
    #
    # if pfi_is_var:
    #     m._eq_pf_interface = pe.Constraint(con_set)
    # else:
    #     if not isinstance(m._pfi, pe.Expression):
    #         raise Exception("Unrecognized type for m._pfi", m._pfi.pprint())

    for interface_name in index_set:
        expr = get_power_flow_interface_expr_ptdf(
            m,
            interface_name,
            PTDF,
            rel_ptdf_tol=rel_ptdf_tol,
            abs_ptdf_tol=abs_ptdf_tol,
        )

        # if pfi_is_var:
        #     m._eq_pf_interface[interface_name] = \
        #             m._pfi[interface_name] == expr
        # else:
        m.addConstr(
            (expr),
            name="pfi_interface_[{}]_period[{}]".format(
                interface_name, period
            ),
        )


def declare_ineq_s_branch_thermal_limit(
        model,
        parent_model,
        period,
        index_set,
        branches,
        s_thermal_limits,
        flow_type=FlowType.POWER,
):
    """
    Create the inequality constraints for the branch thermal limits
    based on the power variables.
    """
    m = parent_model
    # con_set = decl.declare_set('_con_ineq_s_branch_thermal_limit',
    #                            model=model, index_set=index_set)
    #
    # m._ineq_sf_branch_thermal_limit = pe.Constraint(con_set)
    # m._ineq_st_branch_thermal_limit = pe.Constraint(con_set)

    if flow_type == FlowType.CURRENT:
        for branch_name in index_set:
            if s_thermal_limits[branch_name] is None:
                continue

            from_bus = branches[branch_name]["from_bus"]
            to_bus = branches[branch_name]["to_bus"]

            m.addConstr(
                (
                        (model._vr[from_bus] ** 2 + model._vj[from_bus] ** 2)
                        * (
                                model._ifr[branch_name] ** 2
                                + model._ifj[branch_name] ** 2
                        )
                        <= s_thermal_limits[branch_name] ** 2
                ),
                name="_ineq_sf_branch_thermal_limit[{}]_period[{}]".format(
                    branch_name, period
                ),
            )
            m.addConstr(
                (
                        (model._vr[to_bus] ** 2 + model._vj[to_bus] ** 2)
                        * (
                                model._itr[branch_name] ** 2
                                + model._itj[branch_name] ** 2
                        )
                        <= s_thermal_limits[branch_name] ** 2
                ),
                name="_ineq_st_branch_thermal_limit[{}]_period[{}]".format(
                    branch_name, period
                ),
            )

    elif flow_type == FlowType.POWER:
        for branch_name in index_set:
            if s_thermal_limits[branch_name] is None:
                continue
            m.addConstr(
                (
                        model._pf[branch_name] ** 2 + model._qf[
                    branch_name] ** 2
                        <= s_thermal_limits[branch_name] ** 2
                ),
                name="_ineq_sf_branch_thermal_limit_branch[{}]_period[{}]".format(
                    branch_name, period
                ),
            )
            m.addConstr(
                (
                        model._pt[branch_name] ** 2 + model._qt[
                    branch_name] ** 2
                        <= s_thermal_limits[branch_name] ** 2
                ),
                name="_ineq_st_branch_thermal_limit_branch[{}]_period[{}]".format(
                    branch_name, period
                ),
            )


def declare_ineq_p_interface_bounds(
        model,
        parent_model,
        period,
        index_set,
        interfaces,
        approximation_type=ApproximationType.BTHETA,
        slacks=False,
        slack_cost_expr=None,
):
    """
    Create the inequality constraints for the interface limits
    based on the power variables or expressions at the interface level
    """
    # p_interface_limits should be (lower, upper) tuple

    # con_set = decl.declare_set('_con_ineq_p_interface_bounds',
    #                            model=model, index_set=index_set)
    #
    # m._ineq_pf_interface_bounds = pe.Constraint(con_set)
    #
    # # flag for if slacks are on the model
    # if slacks:
    #     if not hasattr(model, 'pfi_slack_pos'):
    #         raise Exception('No positive slack interface variables on model, but slacks=True')
    #     if not hasattr(model, 'pfi_slack_neg'):
    #         raise Exception('No negative slack interface variables on model, but slacks=True')
    #     if slack_cost_expr is None:
    #         raise Exception('No cost expression for slacks, but slacks=True')
    m = parent_model

    if (
            approximation_type == ApproximationType.BTHETA
            or approximation_type == ApproximationType.PTDF
    ):
        for interface_name in index_set:
            interface = interfaces[interface_name]
            if (
                    interface["minimum_limit"] is None
                    and interface["maximum_limit"] is None
            ):
                continue

            if slacks and interface_name in model._pfi_slack_neg.index_set():
                assert interface_name in model._pfi_slack_pos.index_set()
                neg_slack = model._pfi_slack_neg[interface_name]
                pos_slack = model._pfi_slack_pos[interface_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (
                        uc_model._TimePeriodLengthHours
                        * uc_model._InterfaceLimitPenalty[interface_name]
                        * (neg_slack + pos_slack)
                )
                assert len(model._pfi_slack_pos) == len(model._pfi_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub = generate_thermal_bounds(
                model._pfi[interface_name],
                interface["minimum_limit"],
                interface["maximum_limit"],
                neg_slack,
                pos_slack,
            )
            m.addConstr(
                (expr <= ub),
                name="_ineq_pf_interface_upper_bounds_interface[{}]_period[{}]".format(
                    interface_name, period
                ),
            )

            m.addConstr(
                (-expr <= -lb),
                name="_ineq_pf_interface_upper_bounds_interface[{}]_period[{}]".format(
                    interface_name, period
                ),
            )
