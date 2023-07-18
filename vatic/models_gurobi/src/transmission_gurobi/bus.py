"""
Declarations for the modeling components
typically used for buses (including loads and shunts)
"""

from gurobipy import quicksum


def _get_dc_dicts(
        dc_inlet_branches_by_bus, dc_outlet_branches_by_bus, con_set
):
    if dc_inlet_branches_by_bus is None:
        assert dc_outlet_branches_by_bus is None
        dc_inlet_branches_by_bus = {bn: () for bn in con_set}
    if dc_outlet_branches_by_bus is None:
        dc_outlet_branches_by_bus = dc_inlet_branches_by_bus
    return dc_inlet_branches_by_bus, dc_outlet_branches_by_bus


def declare_eq_p_net_withdraw_at_bus(
        model,
        parent_model,
        index_set,
        bus_p_loads,
        gens_by_bus,
        bus_gs_fixed_shunts,
        dc_inlet_branches_by_bus=None,
        dc_outlet_branches_by_bus=None,
):
    """
    Create a named gurobipy expression for bus net withdraw
    """
    m = parent_model
    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = _get_dc_dicts(
        dc_inlet_branches_by_bus, dc_outlet_branches_by_bus, index_set
    )
    m._eq_p_net_withdraw_at_bus = {}
    for b in index_set:
        rhs = (
                bus_gs_fixed_shunts[b]
                + (model._pl[b] if bus_p_loads[b] != 0.0 else 0.0)
                - quicksum(model._pg[g] for g in gens_by_bus[b])
                + quicksum(model._dcpf[branch_name]
                           for branch_name in dc_outlet_branches_by_bus[b]
                           )
                - quicksum(model._dcpf[branch_name]
                           for branch_name in dc_inlet_branches_by_bus[b]
                           )
        )

        m._eq_p_net_withdraw_at_bus[b, model._tm] = m.addConstr(
            (model._p_nw_tm[b] == rhs),
            name="_eq_p_net_withdraw_at_bus[{}]_at_period[{}]".format(
                b, model._tm
            ),
        )


def declare_eq_p_balance_ed(
        model,
        parent_model,
        index_set,
        bus_p_loads,
        gens_by_bus,
        bus_gs_fixed_shunts
):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.
    """
    m = parent_model
    p_expr = quicksum(
        model._pg[bus_name]
        for bus_name in index_set
        if len(gens_by_bus[bus_name]) != 0
    )
    p_expr -= quicksum(
        model._pl[bus_name]
        for bus_name in index_set
        if bus_p_loads[bus_name] is not None
    )
    p_expr -= quicksum(
        bus_gs_fixed_shunts[bus_name]
        for bus_name in index_set
        if bus_gs_fixed_shunts[bus_name] != 0.0
    )

    relaxed_balance = False

    if relaxed_balance:
        m.addConstr(
            (p_expr >= 0.0), name="eq_p_balance_at_period{}".format(model._tm)
        )
    else:
        m.addConstr(
            (p_expr == 0.0), name="eq_p_balance_at_period{}".format(model._tm)
        )
