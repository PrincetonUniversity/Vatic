from egret.model_library.defn import FlowType, CoordinateType, ApproximationType
from math import tan,  radians
from gurobipy import quicksum

def _get_dc_dicts(dc_inlet_branches_by_bus, dc_outlet_branches_by_bus, con_set):
    if dc_inlet_branches_by_bus is None:
        assert dc_outlet_branches_by_bus is None
        dc_inlet_branches_by_bus = {bn:() for bn in con_set}
    if dc_outlet_branches_by_bus is None:
        dc_outlet_branches_by_bus = dc_inlet_branches_by_bus
    return dc_inlet_branches_by_bus, dc_outlet_branches_by_bus

def declare_eq_p_balance_ed(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    p_expr = sum(m._pg[gen_name] for bus_name in index_set for gen_name in gens_by_bus[bus_name])
    p_expr -= sum(m._pl[bus_name] for bus_name in index_set if bus_p_loads[bus_name] is not None)
    p_expr -= sum(bus_gs_fixed_shunts[bus_name] for bus_name in index_set if bus_gs_fixed_shunts[bus_name] != 0.0)

    relaxed_balance = False

    if relaxed_balance:
        m._eq_p_balance = model.addConstr(p_expr >= 0.0, name = 'eq_p_balance')
    else:
        m._eq_p_balance = model.addConstr(expr = p_expr == 0.0, name = 'eq_p_balance')

def declare_eq_p_net_withdraw_at_bus(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts,
                                     dc_inlet_branches_by_bus=None, dc_outlet_branches_by_bus=None):
    """
    Create a named pyomo expression for bus net withdraw
    """
    m = model

    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = _get_dc_dicts(dc_inlet_branches_by_bus,
                                                                        dc_outlet_branches_by_bus,
                                                                        index_set)

    for b in index_set:
        print(b)
        rhs = ( bus_gs_fixed_shunts[b]+ (m._pl[b] if bus_p_loads[b] != 0.0 else 0.0 )
                                            - quicksum(m._pg[g] for g in gens_by_bus[b] )
                                            + quicksum(m._dcpf[branch_name] for branch_name
                                                   in dc_outlet_branches_by_bus[b])
                                            - quicksum(m._dcpf[branch_name] for branch_name
                                                   in dc_inlet_branches_by_bus[b])
                                            )

        m.addConstr((m._p_nw[b] == rhs) , name =  '_eq_p_net_withdraw_at_bus[{}]'.format(b))

def declare_eq_p_balance_ed(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts, **rhs_kwargs):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    p_expr = quicksum(m._pg[gen_name] for bus_name in index_set for gen_name in gens_by_bus[bus_name])
    p_expr -= quicksum(m._pl[bus_name] for bus_name in index_set if bus_p_loads[bus_name] is not None)
    p_expr -= quicksum(bus_gs_fixed_shunts[bus_name] for bus_name in index_set if bus_gs_fixed_shunts[bus_name] != 0.0)

    relaxed_balance = False

    if relaxed_balance:
        m._eq_p_balance = model.addConstr((p_expr >= 0.0), name = 'eq_p_balance')
    else:
        m._eq_p_balance = model.addConstr((p_expr == 0.0), name = 'eq_p_balance')

def declare_var_p_nw(model, index_set, **kwargs):
    """
    Create variable for the reactive power load at a bus
    """
    model._p_nw = model.addVars(index_set, name = 'p_nw')





