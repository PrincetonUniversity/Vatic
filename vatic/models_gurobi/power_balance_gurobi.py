from gurobipy import tupledict, LinExpr, quicksum, GRB


import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.data.ptdf_utils as ptdf_utils
import egret.common.lazy_ptdf_utils as lpu
from egret.model_library.defn import BasePointType, CoordinateType, ApproximationType

from .power_vars_gurobi import _add_reactive_power_vars
from .generation_limits_gurobi import _add_reactive_limits

component_name = 'power_balance'


def _add_hvdc(model):
    def dc_line_power_bounds_rule(m, k, t):
        if value(m.HVDCLineOutOfService[k,t]):
            return (0., 0.)
        return (-m.HVDCThermalLimit[k], m.HVDCThermalLimit[k])
    model.HVDCLinePower = Var(model.HVDCLines, model.TimePeriods, bounds=dc_line_power_bounds_rule)


def _add_q_load_mismatch(model):
    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model.LoadGenerateMismatchReactive = Var(model.Buses, model.TimePeriods,
                                             within=Reals)
    model.posLoadGenerateMismatchReactive = Var(model.Buses, model.TimePeriods,
                                                within=NonNegativeReals)  # load shedding
    model.negLoadGenerateMismatchReactive = Var(model.Buses, model.TimePeriods,
                                                within=NonNegativeReals)  # over generation

    def define_pos_neg_load_generate_mismatch_rule_reactive(m, b, t):
        return m.posLoadGenerateMismatchReactive[b, t] - \
               m.negLoadGenerateMismatchReactive[b, t] \
               == m.LoadGenerateMismatchReactive[b, t]

    model.DefinePosNegLoadGenerateMismatchReactive = Constraint(model.Buses,
                                                                model.TimePeriods,
                                                                rule=define_pos_neg_load_generate_mismatch_rule_reactive)

    # the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
    # the appearance of load generation mismatch component values in the range of *negative* e-5.
    # what these small negative values do is to cause the optimal objective to be a very large negative,
    # due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
    # at this point, but it does seem to work broadly. we tried a single global constraint, across all
    # buses, but that failed to correct the problem, and caused the solve times to explode.

    def pos_load_generate_mismatch_tolerance_rule_reactive(m, b):
        return sum((m.posLoadGenerateMismatchReactive[b, t] for t in
                    m.TimePeriods)) >= 0.0

    model.PosLoadGenerateMismatchToleranceReactive = Constraint(model.Buses,
                                                                rule=pos_load_generate_mismatch_tolerance_rule_reactive)

    def neg_load_generate_mismatch_tolerance_rule_reactive(m, b):
        return sum((m.negLoadGenerateMismatchReactive[b, t] for t in
                    m.TimePeriods)) >= 0.0

    model.NegLoadGenerateMismatchToleranceReactive = Constraint(model.Buses,
                                                                rule=neg_load_generate_mismatch_tolerance_rule_reactive)

    def compute_q_load_mismatch_cost_rule(m, t):
        return m.LoadMismatchPenaltyReactive * m.TimePeriodLengthHours * sum(
            m.posLoadGenerateMismatchReactive[b, t] +
            m.negLoadGenerateMismatchReactive[b, t] for b in m.Buses)

    model.LoadMismatchCostReactive = Expression(model.TimePeriods,
                                                rule=compute_q_load_mismatch_cost_rule)


def _add_blank_load_mismatch(model):
    model.LoadGenerateMismatch = Param(model.Buses, model.TimePeriods,
                                       default=0.)
    model.posLoadGenerateMismatch = Param(model.Buses, model.TimePeriods,
                                          default=0.)
    model.negLoadGenerateMismatch = Param(model.Buses, model.TimePeriods,
                                          default=0.)
    model.LoadMismatchCost = Param(model.TimePeriods, default=0.)


def _add_blank_q_load_mismatch(model):
    model.LoadGenerateMismatchReactive = Param(model.Buses, model.TimePeriods,
                                               default=0.)
    model.posLoadGenerateMismatchReactive = Param(model.Buses,
                                                  model.TimePeriods,
                                                  default=0.)
    model.negLoadGenerateMismatchReactive = Param(model.Buses,
                                                  model.TimePeriods,
                                                  default=0.)
    model.LoadMismatchCostReactive = Param(model.TimePeriods, default=0.)


def _add_blank_q_load_mismatch(model):
    model.LoadGenerateMismatchReactive = Param(model.Buses, model.TimePeriods,
                                               default=0.)
    model.posLoadGenerateMismatchReactive = Param(model.Buses,
                                                  model.TimePeriods,
                                                  default=0.)
    model.negLoadGenerateMismatchReactive = Param(model.Buses,
                                                  model.TimePeriods,
                                                  default=0.)
    model.LoadMismatchCostReactive = Param(model.TimePeriods, default=0.)

def _add_system_load_mismatch(model):
    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model.posLoadGenerateMismatch = Var(model.TimePeriods,
                                        within=NonNegativeReals)  # load shedding
    model.negLoadGenerateMismatch = Var(model.TimePeriods,
                                        within=NonNegativeReals)  # over generation

    ## for interfacing with the rest of the model code
    def define_pos_neg_load_generate_mismatch_rule(m, b, t):
        if b == value(m.ReferenceBus):
            return LinearExpression(linear_vars=[m.posLoadGenerateMismatch[t],
                                                 m.negLoadGenerateMismatch[t]],
                                    linear_coefs=[1., -1.])
        else:
            return 0

    model.LoadGenerateMismatch = Expression(model.Buses, model.TimePeriods,
                                            rule=define_pos_neg_load_generate_mismatch_rule)

    # the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
    # the appearance of load generation mismatch component values in the range of *negative* e-5.
    # what these small negative values do is to cause the optimal objective to be a very large negative,
    # due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
    # at this point, but it does seem to work broadly. we tried a single global constraint, across all
    # buses, but that failed to correct the problem, and caused the solve times to explode.

    def pos_load_generate_mismatch_tolerance_rule(m):
        linear_vars = list(m.posLoadGenerateMismatch.values())
        linear_coefs = [1.] * len(linear_vars)
        return (0., LinearExpression(linear_vars=linear_vars,
                                     linear_coefs=linear_coefs), None)

    model.PosLoadGenerateMismatchTolerance = Constraint(
        rule=pos_load_generate_mismatch_tolerance_rule)

    def neg_load_generate_mismatch_tolerance_rule(m):
        linear_vars = list(m.negLoadGenerateMismatch.values())
        linear_coefs = [1.] * len(linear_vars)
        return (0., LinearExpression(linear_vars=linear_vars,
                                     linear_coefs=linear_coefs), None)

    model.NegLoadGenerateMismatchTolerance = Constraint(
        rule=neg_load_generate_mismatch_tolerance_rule)

    def compute_load_mismatch_cost_rule(m, t):
        linear_vars = [*m.posLoadGenerateMismatch.values(),
                       *m.negLoadGenerateMismatch.values()]
        linear_coefs = [m.LoadMismatchPenalty * m.TimePeriodLengthHours] * len(
            linear_vars)
        return LinearExpression(linear_vars=linear_vars,
                                linear_coefs=linear_coefs)

    model.LoadMismatchCost = Expression(model.TimePeriods,
                                        rule=compute_load_mismatch_cost_rule)


def _add_load_mismatch(model):
    over_gen_maxes = {}
    over_gen_times_per_bus = {b: list() for b in model.Buses}

    load_shed_maxes = {}
    load_shed_times_per_bus = {b: list() for b in model.Buses}

    for b in model.Buses:

        # storage, for now, does not
        # have time-vary parameters
        storage_max_injections = 0.
        storage_max_withdraws = 0.

        for s in model.StorageAtBus[b]:
            storage_max_injections += value(model.MaximumPowerOutputStorage[s])
            storage_max_withdraws += value(model.MaximumPowerInputStorage[s])

        for t in model.TimePeriods:
            max_injections = storage_max_injections
            max_withdrawls = storage_max_withdraws

            for g in model.ThermalGeneratorsAtBus[b]:
                p_max = value(model.MaximumPowerOutput[g, t])
                p_min = value(model.MinimumPowerOutput[g, t])

                if p_max > 0:
                    max_injections += p_max
                if p_min < 0:
                    max_withdrawls += -p_min

            for n in model.NondispatchableGeneratorsAtBus[b]:
                p_max = value(model.MaxNondispatchablePower[n, t])
                p_min = value(model.MinNondispatchablePower[n, t])

                if p_max > 0:
                    max_injections += p_max
                if p_min < 0:
                    max_withdrawls += -p_min

            load = value(model.Demand[b, t])
            if load > 0:
                max_withdrawls += load
            elif load < 0:
                max_injections += -load

            if max_injections > 0:
                over_gen_maxes[b, t] = max_injections
                over_gen_times_per_bus[b].append(t)
            if max_withdrawls > 0:
                load_shed_maxes[b, t] = max_withdrawls
                load_shed_times_per_bus[b].append(t)

    model.OverGenerationBusTimes = Set(dimen=2,
                                       initialize=over_gen_maxes.keys())
    model.LoadSheddingBusTimes = Set(dimen=2,
                                     initialize=load_shed_maxes.keys())

    def get_over_gen_bounds(m, b, t):
        return (0, over_gen_maxes[b, t])

    model.OverGeneration = Var(model.OverGenerationBusTimes,
                               within=NonNegativeReals,
                               bounds=get_over_gen_bounds)  # over generation

    def get_load_shed_bounds(m, b, t):
        return (0, load_shed_maxes[b, t])

    model.LoadShedding = Var(model.LoadSheddingBusTimes,
                             within=NonNegativeReals,
                             bounds=get_load_shed_bounds)  # load shedding

    # the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
    # the appearance of load generation mismatch component values in the range of *negative* e-5.
    # what these small negative values do is to cause the optimal objective to be a very large negative,
    # due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
    # at this point, but it does seem to work broadly. we tried a single global constraint, across all
    # buses, but that failed to correct the problem, and caused the solve times to explode.

    def pos_load_generate_mismatch_tolerance_rule(m, b):
        if load_shed_times_per_bus[b]:
            linear_vars = list(
                m.LoadShedding[b, t] for t in load_shed_times_per_bus[b])
            linear_coefs = [1.] * len(linear_vars)
            return (0., LinearExpression(linear_vars=linear_vars,
                                         linear_coefs=linear_coefs), None)
        else:
            return Constraint.Feasible

    model.PosLoadGenerateMismatchTolerance = Constraint(model.Buses,
                                                        rule=pos_load_generate_mismatch_tolerance_rule)

    def neg_load_generate_mismatch_tolerance_rule(m, b):
        if over_gen_times_per_bus[b]:
            linear_vars = list(
                m.OverGeneration[b, t] for t in over_gen_times_per_bus[b])
            linear_coefs = [1.] * len(linear_vars)
            return (0., LinearExpression(linear_vars=linear_vars,
                                         linear_coefs=linear_coefs), None)
        else:
            return Constraint.Feasible

    model.NegLoadGenerateMismatchTolerance = Constraint(model.Buses,
                                                        rule=neg_load_generate_mismatch_tolerance_rule)

    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model.LoadGenerateMismatch = Expression(model.Buses, model.TimePeriods)
    for b in model.Buses:
        for t in model.TimePeriods:
            model.LoadGenerateMismatch[b, t].expr = 0.
    for b, t in model.LoadSheddingBusTimes:
        model.LoadGenerateMismatch[b, t].expr += model.LoadShedding[b, t]
    for b, t in model.OverGenerationBusTimes:
        model.LoadGenerateMismatch[b, t].expr -= model.OverGeneration[b, t]

    model.LoadMismatchCost = Expression(model.TimePeriods)
    for t in model.TimePeriods:
        model.LoadMismatchCost[t].expr = 0.
    for b, t in model.LoadSheddingBusTimes:
        model.LoadMismatchCost[
            t].expr += model.LoadMismatchPenalty * model.TimePeriodLengthHours * \
                       model.LoadShedding[b, t]
    for b, t in model.OverGenerationBusTimes:
        model.LoadMismatchCost[
            t].expr += model.LoadMismatchPenalty * model.TimePeriodLengthHours * \
                       model.OverGeneration[b, t]

def _copperplate_network_model(block, tm, relax_balance=None):

    m, gens_by_bus, bus_p_loads, bus_gs_fixed_shunts = \
            _setup_egret_network_model(block, tm)

    ### declare the p balance
    libbus.declare_eq_p_balance_ed(model=block,
                                   index_set=m.Buses,
                                   bus_p_loads=bus_p_loads,
                                   gens_by_bus=gens_by_bus,
                                   bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                   relax_balance = relax_balance,
                                   )


def _copperplate_relax_network_model(block,tm):
    _copperplate_network_model(block, tm, relax_balance=True)

def _copperplate_approx_network_model(block,tm):
    _copperplate_network_model(block, tm, relax_balance=False)

def _setup_branch_slacks(m,block,tm):
    # declare the branch slack variables
    # they have a sparse index set
    libbranch.declare_var_pf_slack_pos(model=block,
                                       index_set=m.BranchesWithSlack,
                                       domain=NonNegativeReals,
                                       dense=False)
    libbranch.declare_var_pf_slack_neg(model=block,
                                       index_set=m.BranchesWithSlack,
                                       domain=NonNegativeReals,
                                       dense=False)

def _setup_interface_slacks(m,block,tm):
    # declare the interface slack variables
    # they have a sparse index set
    libbranch.declare_var_pfi_slack_pos(model=block,
                                        index_set=m.InterfacesWithSlack,
                                        domain=NonNegativeReals,
                                        dense=False)
    libbranch.declare_var_pfi_slack_neg(model=block,
                                        index_set=m.InterfacesWithSlack,
                                        domain=NonNegativeReals,
                                        dense=False)

def _setup_contingency_slacks(m,block,tm):
    # declare the interface slack variables
    # they have a sparse index set
    block.pfc_slack_pos = Var(block._contingency_set, domain=NonNegativeReals, dense=False)
    block.pfc_slack_neg = Var(block._contingency_set, domain=NonNegativeReals, dense=False)

def _setup_egret_network_topology(m,tm):
    buses = m._buses
    branches = m._branches
    interfaces = m._interfaces
    contingencies = m._contingencies

    branches_in_service = tuple(l for l in m.TransmissionLines if not value(m.LineOutOfService[l,tm]))

    ## this will serve as a key into our dict of PTDF matricies,
    ## so that we can avoid recalculating them each time step
    ## with the same network topology
    branches_out_service = tuple(l for l in m.TransmissionLines if value(m.LineOutOfService[l,tm]))

    return buses, branches, branches_in_service, branches_out_service, interfaces, contingencies

def _setup_egret_network_model(block, tm):
    m = block.parent_block()

    ## this is not the "real" gens by bus, but the
    ## index of net injections from the UC model
    gens_by_bus = block.gens_by_bus

    ### declare (and fix) the loads at the buses
    bus_p_loads = {b: value(m.Demand[b,tm]) for b in m.Buses}

    block.pl = Param(m.Buses, initialize=bus_p_loads)

    bus_gs_fixed_shunts = m._bus_gs_fixed_shunts

    return m, gens_by_bus, bus_p_loads, bus_gs_fixed_shunts

def _ptdf_dcopf_network_model(block, tm):
    m, gens_by_bus, bus_p_loads, bus_gs_fixed_shunts = \
        _setup_egret_network_model(block, tm)

    buses, branches, \
    branches_in_service, branches_out_service, \
    interfaces, contingencies = _setup_egret_network_topology(m, tm)

    ptdf_options = m._ptdf_options

    libbus.declare_var_p_nw(block, m.Buses)

    ### declare net withdraw expression for use in PTDF power flows
    libbus.declare_eq_p_net_withdraw_at_bus(model=block,
                                            index_set=m.Buses,
                                            bus_p_loads=bus_p_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                            )

    ### declare the p balance
    libbus.declare_eq_p_balance_ed(model=block,
                                   index_set=m.Buses,
                                   bus_p_loads=bus_p_loads,
                                   gens_by_bus=gens_by_bus,
                                   bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                   )

    ### add "blank" power flow expressions
    libbranch.declare_expr_pf(model=block,
                              index_set=branches_in_service,
                              )

    _setup_branch_slacks(m, block, tm)

    ### interface setup
    libbranch.declare_expr_pfi(model=block,
                               index_set=interfaces.keys()
                               )

    _setup_interface_slacks(m, block, tm)

    ### contingency setup
    ### NOTE: important that this not be dense, we'll add elements
    ###       as we find violations
    block._contingency_set = Set(within=m.Contingencies * m.TransmissionLines)
    block.pfc = Expression(block._contingency_set)
    _setup_contingency_slacks(m, block, tm)

    ### Get the PTDF matrix from cache, from file, or create a new one
    ### m._PTDFs set in uc_model_generator
    if branches_out_service not in m._PTDFs:
        buses_idx = tuple(buses.keys())

        reference_bus = value(m.ReferenceBus)

        ## NOTE: For now, just use a flat-start for unit commitment
        PTDF = ptdf_utils.VirtualPTDFMatrix(branches, buses, reference_bus,
                                            BasePointType.FLATSTART,
                                            ptdf_options,
                                            contingencies=contingencies,
                                            branches_keys=branches_in_service,
                                            buses_keys=buses_idx,
                                            interfaces=interfaces)

        m._PTDFs[branches_out_service] = PTDF

    else:
        PTDF = m._PTDFs[branches_out_service]

    ### attach the current PTDF object to this block
    block._PTDF = PTDF
    rel_ptdf_tol = m._ptdf_options['rel_ptdf_tol']
    abs_ptdf_tol = m._ptdf_options['abs_ptdf_tol']

    if ptdf_options['lazy']:
        ### add "blank" real power flow limits
        libbranch.declare_ineq_p_branch_thermal_bounds(model=block,
                                                       index_set=branches_in_service,
                                                       branches=branches,
                                                       p_thermal_limits=None,
                                                       approximation_type=None,
                                                       slacks=True,
                                                       slack_cost_expr=
                                                       m.BranchViolationCost[
                                                           tm]
                                                       )
        ### declare the "blank" interface flow limits
        libbranch.declare_ineq_p_interface_bounds(model=block,
                                                  index_set=interfaces.keys(),
                                                  interfaces=interfaces,
                                                  approximation_type=None,
                                                  slacks=True,
                                                  slack_cost_expr=
                                                  m.InterfaceViolationCost[tm]
                                                  )
        ### declare the "blank" interface flow limits
        libbranch.declare_ineq_p_contingency_branch_thermal_bounds(model=block,
                                                                   index_set=block._contingency_set,
                                                                   pc_thermal_limits=None,
                                                                   approximation_type=None,
                                                                   slacks=True,
                                                                   slack_cost_expr=
                                                                   m.ContingencyViolationCost[
                                                                       tm]
                                                                   )

        ### add helpers for tracking monitored branches
        lpu.add_monitored_flow_tracker(block)

        ### add initial branches to monitored set
        lpu.add_initial_monitored_branches(block, branches,
                                           branches_in_service, ptdf_options,
                                           PTDF)

        ### add initial interfaces to monitored set
        lpu.add_initial_monitored_interfaces(block, interfaces, ptdf_options,
                                             PTDF)

    else:  ### add all the dense constraints
        if contingencies:
            raise RuntimeError(
                "Contingency constraints only supported in lazy mode")
        p_max = {k: branches[k]['rating_long_term'] for k in
                 branches_in_service}

        ### declare the branch power flow approximation constraints
        libbranch.declare_eq_branch_power_ptdf_approx(model=block,
                                                      index_set=branches_in_service,
                                                      PTDF=PTDF,
                                                      abs_ptdf_tol=abs_ptdf_tol,
                                                      rel_ptdf_tol=rel_ptdf_tol
                                                      )
        ### declare the real power flow limits
        libbranch.declare_ineq_p_branch_thermal_bounds(model=block,
                                                       index_set=branches_in_service,
                                                       branches=branches,
                                                       p_thermal_limits=p_max,
                                                       approximation_type=ApproximationType.PTDF,
                                                       slacks=True,
                                                       slack_cost_expr=
                                                       m.BranchViolationCost[
                                                           tm]
                                                       )

        ### declare the branch power flow approximation constraints
        libbranch.declare_eq_interface_power_ptdf_approx(model=block,
                                                         index_set=interfaces.keys(),
                                                         PTDF=PTDF,
                                                         abs_ptdf_tol=abs_ptdf_tol,
                                                         rel_ptdf_tol=rel_ptdf_tol
                                                         )

        ### declare the interface flow limits
        libbranch.declare_ineq_p_interface_bounds(model=block,
                                                  index_set=interfaces.keys(),
                                                  interfaces=interfaces,
                                                  approximation_type=ApproximationType.PTDF,
                                                  slacks=True,
                                                  slack_cost_expr=
                                                  m.InterfaceViolationCost[tm]
                                                  )

def ptdf_power_flow(model, slacks=True):
    _add_egret_power_flow(model, _ptdf_dcopf_network_model, reactive_power=False, slacks=slacks)

## Defines generic interface for egret tramsmission models
def _add_egret_power_flow(model, network_model_builder, reactive_power=False, slacks=True):

    ## save flag for objective
    model.reactive_power = reactive_power

    system_load_mismatch = (network_model_builder in \
                            [_copperplate_approx_network_model, \
                             _copperplate_relax_network_model, \
                            ]
                           )

    if slacks:
        if system_load_mismatch:
            _add_system_load_mismatch(model)
        else:
            _add_load_mismatch(model)
    else:
        if system_load_mismatch:
            _add_blank_system_load_mismatch(model)
        else:
            _add_blank_load_mismatch(model)

    _add_hvdc(model)

    if reactive_power:
        if system_load_mismatch:
            raise Exception("Need to implement system mismatch for reactive power")
        model = _add_reactive_power_vars(model)
        _add_reactive_limits(model)
        if slacks:
            _add_q_load_mismatch(model)
        else:
            _add_blank_q_load_mistmatch(model)

    # for transmission network
    model.TransmissionBlock = Block(model.TimePeriods, concrete=True)

    # for interface violation costs at a time step
    model.BranchViolationCost = Expression(model.TimePeriods, rule=lambda m,t:0.)

    # for interface violation costs at a time step
    model.InterfaceViolationCost = Expression(model.TimePeriods, rule=lambda m,t:0.)

    # for contingency violation costs at a time step
    model.ContingencyViolationCost = Expression(model.TimePeriods, rule=lambda m,t:0.)

    for tm in model.TimePeriods:
        b = model.TransmissionBlock[tm]
        ## this creates a fake bus generator for all the
        ## appropriate injection/withdraws from the unit commitment
        ## model
        b.pg = Expression(model.Buses, rule=_get_pg_expr_rule(tm))
        if reactive_power:
            b.qg = Expression(model.Buses, rule=_get_qg_expr_rule(tm))
        b.gens_by_bus = {bus : [bus] for bus in model.Buses}
        network_model_builder(b,tm)