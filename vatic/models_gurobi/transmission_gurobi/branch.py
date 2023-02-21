from gurobipy import LinExpr, quicksum

from egret.model_library.defn import FlowType, CoordinateType, ApproximationType, RelaxationType

def get_power_flow_interface_expr_ptdf(model, interface_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create a pyomo power flow expression from PTDF matrix for an interface
    """
    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    const = PTDF.get_interface_const(interface_name)
    max_coef = PTDF.get_interface_ptdf_abs_max(interface_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef)

    m_p_nw = model._p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_interface_ptdf_iterator(interface_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    expr = LinExpr(coef_list, var_list)+const

    return expr

def get_power_flow_expr_ptdf_approx(model, branch_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create a pyomo power flow expression from PTDF matrix
    """

    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    const = PTDF.get_branch_const(branch_name)

    max_coef = PTDF.get_branch_ptdf_abs_max(branch_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_p_nw = model._p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_branch_ptdf_iterator(branch_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    expr = LinExpr(coef_list, var_list)+const
    return expr

def generate_thermal_bounds(pf, llimit, ulimit, neg_slack=None, pos_slack=None):
    """
    Create a constraint for thermal limits on a line given the power flow
    expression or variable pf, a lower limit llimit, a uppder limit ulimit,
    and the negative slack variable, neg_slack, (None if not needed) and
    positive slack variable, pos_slack, (None if not needed) added to this
    constraint.
    """
    if pf:
        ## if necessary, copy again, so that m.pf[bn] **is** the flow
        add_vars = list()
        add_coefs = list()
        if neg_slack is not None:
            add_vars.append(neg_slack)
            add_coefs.append(1)
        if pos_slack is not None:
            add_vars.append(pos_slack)
            add_coefs.append(-1)
        if add_vars:
            ## create a copy
            old_expr = pf.expr
            expr = old_expr.add_Terms(add_coefs, add_vars)+old_expr
        else:
            expr = pf
    else:
        expr = pf
        if neg_slack is not None:
            expr += neg_slack
        if pos_slack is not None:
            expr -= pos_slack
    return (llimit, expr, ulimit)

def declare_ineq_p_branch_thermal_bounds(model, index_set,
                                        branches, p_thermal_limits,
                                        approximation_type=ApproximationType.BTHETA,
                                        slacks=False, slack_cost_expr=None):
    """
    Create an inequality constraint for the branch thermal limits
    based on the power variables or expressions.
    """
    m = model
    # con_set = decl.declare_set('_con_ineq_p_branch_thermal_bounds',
    #                            model=model, index_set=index_set)
    # flag for if slacks are on the model
    # if slacks:
    #     if not hasattr(model, 'pf_slack_pos'):
    #         raise Exception('No positive slack branch variables on model, but slacks=True')
    #     if not hasattr(model, 'pf_slack_neg'):
    #         raise Exception('No negative slack branch variables on model, but slacks=True')
    #     if slack_cost_expr is None:
    #         raise Exception('No cost expression for slacks, but slacks=True')

    # m._ineq_pf_branch_thermal_bounds = pe.Constraint(con_set)

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for branch_name in index_set:
            limit = p_thermal_limits[branch_name]
            if limit is None:
                continue

            if slacks and branch_name in m._pf_slack_neg.index_set():
                assert branch_name in m.pf_slack_pos.index_set()
                neg_slack = m._pf_slack_neg[branch_name]
                pos_slack = m._pf_slack_pos[branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model._TimePeriodLengthHours*uc_model._BranchLimitPenalty[branch_name] *
                                    (neg_slack + pos_slack) )
                assert len(m._pf_slack_pos) == len(m._pf_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub = generate_thermal_bounds(m.pf[branch_name], -limit, limit,
                                    neg_slack, pos_slack)
            m._ineq_pf_branch_thermal_upper_bounds = m.addConstr(
                    (expr <= ub),
                    name = 'ineq_pf_branch_thermal_upper_bounds[{}]'.format(branch_name))
            m._ineq_pf_branch_thermal_lower_bounds = m.addConstr(
                    (-expr <= -lb),
                    name = 'ineq_pf_branch_thermal_lower_bounds[{}]'.format(branch_name))

def declare_ineq_p_interface_bounds(model, index_set, interfaces,
                                        approximation_type=ApproximationType.BTHETA,
                                        slacks=False, slack_cost_expr=None):
    """
    Create the inequality constraints for the interface limits
    based on the power variables or expressions.

    p_interface_limits should be (lower, upper) tuple
    """
    m = model
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

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for interface_name in index_set:
            interface = interfaces[interface_name]
            if interface['minimum_limit'] is None and \
                    interface['maximum_limit'] is None:
                continue

            if slacks and interface_name in m._pfi_slack_neg.index_set():
                assert interface_name in m._pfi_slack_pos.index_set()
                neg_slack = m._pfi_slack_neg[interface_name]
                pos_slack = m._pfi_slack_pos[interface_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model._TimePeriodLengthHours*uc_model._InterfaceLimitPenalty[interface_name] *
                                    (neg_slack + pos_slack) )
                assert len(m._pfi_slack_pos) == len(m._pfi_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub =  generate_thermal_bounds(m._pfi[interface_name], interface['minimum_limit'], interface['maximum_limit'],
                                        neg_slack, pos_slack)
            m._ineq_pf_interface_upper_bounds = m.addConstr(
                    (expr <= ub),
                    name = 'ineq_pf_interface_upper_bounds[{}]'.format(interface_name))
            m._ineq_pf_interface_lower_bounds = m.addConstr(
                    (-expr <= -lb),
                    name = 'ineq_pf_interface_lower_bounds[{}]'.format(interface_name))

def declare_ineq_p_contingency_branch_thermal_bounds(model, index_set,
                                                     pc_thermal_limits,
                                                     approximation_type=ApproximationType.PTDF,
                                                     slacks=False, slack_cost_expr=None):
    """
    Create an inequality constraint for the branch thermal limits
    based on the power variables or expressions.
    """
    m = model
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

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for (contingency_name, branch_name) in index_set:
            limit = pc_thermal_limits[branch_name]
            if limit is None:
                continue

            if slacks and (contingency_name, branch_name) in m._pfc_slack_neg.index_set():
                assert (contingency_name, branch_name) in m._pfc_slack_pos.index_set()
                neg_slack = m._pfc_slack_neg[contingency_name, branch_name]
                pos_slack = m._pfc_slack_pos[contingency_name, branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model._TimePeriodLengthHours
                                         * uc_model._ContingencyLimitPenalty
                                         * (neg_slack + pos_slack) )
                assert len(m._pfc_slack_pos) == len(m._pfc_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub =  generate_thermal_bounds(m._pfc[contingency_name, branch_name], -limit, limit, neg_slack, pos_slack)
            m._ineq_pf_contingency_branch_thermal_upper_bounds = m.addConstr(
                    (expr <= ub),
                    name = 'ineq_pf_interface_upper_bounds[{}]'.format(branch_name))
            m._ineq_pf_contingency_branch_thermal_lower_bounds  = m.addConstr(
                    (-expr <= -lb),
                    name = 'ineq_pf_interface_lower_bounds[{}]'.format(branch_name))


def declare_eq_branch_power_ptdf_approx(model, index_set, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create the equality constraints or expressions for power (from PTDF
    approximation) in the branch
    """

    m = model

    # con_set = decl.declare_set("_con_eq_branch_power_ptdf_approx_set", model, index_set)


    # if pf_is_var:
    #     m._eq_pf_branch = pe.Constraint(con_set)
    # else:
    #     if not isinstance(m._pf, pe.Expression):
    #         raise Exception("Unrecognized type for m._pf", m._pf.pprint())

    for branch_name in index_set:
        expr = \
            get_power_flow_expr_ptdf_approx(m, branch_name, PTDF, rel_ptdf_tol=rel_ptdf_tol, abs_ptdf_tol=abs_ptdf_tol)

        m.addConstr((m._pf[branch_name] == expr), name = '_eq_pf_branch[{}]'.format(branch_name))



def declare_ineq_p_branch_thermal_bounds(model, index_set,
                                        branches, p_thermal_limits,
                                        approximation_type=ApproximationType.BTHETA,
                                        slacks=False, slack_cost_expr=None):
    """
    Create an inequality constraint for the branch thermal limits
    based on the power variables or expressions.
    """
    m = model
    # con_set = decl.declare_set('_con_ineq_p_branch_thermal_bounds',
    #                            model=model, index_set=index_set)
    # # flag for if slacks are on the model
    # if slacks:
    #     if not hasattr(model, 'pf_slack_pos'):
    #         raise Exception('No positive slack branch variables on model, but slacks=True')
    #     if not hasattr(model, 'pf_slack_neg'):
    #         raise Exception('No negative slack branch variables on model, but slacks=True')
    #     if slack_cost_expr is None:
    #         raise Exception('No cost expression for slacks, but slacks=True')

    # m._ineq_pf_branch_thermal_bounds = pe.Constraint(con_set)

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for branch_name in index_set:
            limit = p_thermal_limits[branch_name]
            if limit is None:
                continue

            if slacks and branch_name in m._pf_slack_neg.index_set():
                assert branch_name in m._pf_slack_pos.index_set()
                neg_slack = m._pf_slack_neg[branch_name]
                pos_slack = m._pf_slack_pos[branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model._TimePeriodLengthHours*uc_model._BranchLimitPenalty[branch_name] *
                                    (neg_slack + pos_slack) )
                assert len(m._pf_slack_pos) == len(m._pf_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub = generate_thermal_bounds(m._pf[branch_name], -limit, limit, neg_slack, pos_slack)
            m._ineq_pf_branch_thermal_upper_bounds = model.addConstrs(
                (expr >= ub), name = 'ineq_pf_branch_thermal_upper-bounds[{}]'.format(branch_name)
            )
            m._ineq_pf_branch_thermal_lower_bounds = model.addConstrs(
                (-expr >= -lb), name = 'ineq_pf_branch_thermal_lower_bounds[{}]'.format(branch_name)
            )


def declare_eq_interface_power_ptdf_approx(model, index_set, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create equality constraints or expressions for power (from PTDF
    approximation) across the interface
    """

    m = model
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
        expr = \
            get_power_flow_interface_expr_ptdf(m, interface_name, PTDF,
                    rel_ptdf_tol=rel_ptdf_tol, abs_ptdf_tol=abs_ptdf_tol)

        # if pfi_is_var:
        #     m._eq_pf_interface[interface_name] = \
        #             m._pfi[interface_name] == expr
        # else:
        m._pfi = m.addConstr(expr, name = 'pfi[{}]'.format(interface_name))



def declare_ineq_s_branch_thermal_limit(model, index_set,
                                        branches, s_thermal_limits,
                                        flow_type=FlowType.POWER):
    """
    Create the inequality constraints for the branch thermal limits
    based on the power variables.
    """
    m = model
    # con_set = decl.declare_set('_con_ineq_s_branch_thermal_limit',
    #                            model=model, index_set=index_set)
    #
    # m._ineq_sf_branch_thermal_limit = pe.Constraint(con_set)
    # m._ineq_st_branch_thermal_limit = pe.Constraint(con_set)

    if flow_type == FlowType.CURRENT:
        for branch_name in index_set:
            if s_thermal_limits[branch_name] is None:
                continue

            from_bus = branches[branch_name]['from_bus']
            to_bus = branches[branch_name]['to_bus']

            m.addConstr(
                ((m._vr[from_bus] ** 2 + m._vj[from_bus] ** 2) * (m._ifr[branch_name] ** 2 + m._ifj[branch_name] ** 2) \
                <= s_thermal_limits[branch_name] ** 2),
                name = '_ineq_sf_branch_thermal_limit[{}]'.format(branch_name))
            m.addConstr(
                ((m._vr[to_bus] ** 2 + m._vj[to_bus] ** 2) * (m._itr[branch_name] ** 2 + m._itj[branch_name] ** 2) \
                <= s_thermal_limits[branch_name] ** 2),
                name='_ineq_st_branch_thermal_limit[{}]'.format(branch_name)
            )

    elif flow_type == FlowType.POWER:
        for branch_name in index_set:
            if s_thermal_limits[branch_name] is None:
                continue
            m.addConstr(
                (m._pf[branch_name] ** 2 + m._qf[branch_name] ** 2 \
                <= s_thermal_limits[branch_name] ** 2),
                name = '_ineq_sf_branch_thermal_limit[{}]'.format(branch_name))
            m.addConstr(
                (m._pt[branch_name] ** 2 + m._qt[branch_name] ** 2 \
                <= s_thermal_limits[branch_name] ** 2),
                name='_ineq_st_branch_thermal_limit[{}]'.format(branch_name)
            )

def declare_ineq_p_interface_bounds(model, index_set, interfaces,
                                        approximation_type=ApproximationType.BTHETA,
                                        slacks=False, slack_cost_expr=None):
    """
    Create the inequality constraints for the interface limits
    based on the power variables or expressions.

    p_interface_limits should be (lower, upper) tuple
    """
    m = model
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

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for interface_name in index_set:
            interface = interfaces[interface_name]
            if interface['minimum_limit'] is None and \
                    interface['maximum_limit'] is None:
                continue

            if slacks and interface_name in m._pfi_slack_neg.index_set():
                assert interface_name in m._pfi_slack_pos.index_set()
                neg_slack = m._pfi_slack_neg[interface_name]
                pos_slack = m._pfi_slack_pos[interface_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model._TimePeriodLengthHours*uc_model._InterfaceLimitPenalty[interface_name] *
                                    (neg_slack + pos_slack) )
                assert len(m._pfi_slack_pos) == len(m._pfi_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            lb, expr, ub = generate_thermal_bounds(m._pfi[interface_name], interface['minimum_limit'], interface['maximum_limit'],
                                        neg_slack, pos_slack)
            m.addConstr(
                (expr <= ub),
                name = '_ineq_pf_interface_upper_bounds[{}]'.format(interface_name))

            m.addConstr(
                (-expr <= -lb),
                name = '_ineq_pf_interface_upper_bounds[{}]'.format(interface_name))