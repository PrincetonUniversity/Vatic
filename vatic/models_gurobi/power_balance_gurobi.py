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
        if value(m._HVDCLineOutOfService[k,t]):
            return (0., 0.)
        return (-m._HVDCThermalLimit[k], m._HVDCThermalLimit[k])
    model._HVDCLinePower = Var(model._HVDCLines, model._TimePeriods, bounds=dc_line_power_bounds_rule)


def _add_q_load_mismatch(model):
    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model._LoadGenerateMismatchReactive = Var(model._Buses, model._TimePeriods,
                                             within=Reals)
    model._posLoadGenerateMismatchReactive = Var(model._Buses, model._TimePeriods,
                                                within=NonNegativeReals)  # load shedding
    model._negLoadGenerateMismatchReactive = Var(model._Buses, model._TimePeriods,
                                                within=NonNegativeReals)  # over generation

    def define_pos_neg_load_generate_mismatch_rule_reactive(m, b, t):
        return m._posLoadGenerateMismatchReactive[b, t] - \
               m._negLoadGenerateMismatchReactive[b, t] \
               == m._LoadGenerateMismatchReactive[b, t]

    model._DefinePosNegLoadGenerateMismatchReactive = Constraint(model._Buses,
                                                                model._TimePeriods,
                                                                rule=define_pos_neg_load_generate_mismatch_rule_reactive)

    # the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
    # the appearance of load generation mismatch component values in the range of *negative* e-5.
    # what these small negative values do is to cause the optimal objective to be a very large negative,
    # due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
    # at this point, but it does seem to work broadly. we tried a single global constraint, across all
    # buses, but that failed to correct the problem, and caused the solve times to explode.

    def pos_load_generate_mismatch_tolerance_rule_reactive(m, b):
        return sum((m._posLoadGenerateMismatchReactive[b, t] for t in
                    m._TimePeriods)) >= 0.0

    model._PosLoadGenerateMismatchToleranceReactive = Constraint(model._Buses,
                                                                rule=pos_load_generate_mismatch_tolerance_rule_reactive)

    def neg_load_generate_mismatch_tolerance_rule_reactive(m, b):
        return sum((m._negLoadGenerateMismatchReactive[b, t] for t in
                    m._TimePeriods)) >= 0.0

    model._NegLoadGenerateMismatchToleranceReactive = Constraint(model._Buses,
                                                                rule=neg_load_generate_mismatch_tolerance_rule_reactive)

    def compute_q_load_mismatch_cost_rule(m, t):
        return m._LoadMismatchPenaltyReactive * m._TimePeriodLengthHours * sum(
            m._posLoadGenerateMismatchReactive[b, t] +
            m._negLoadGenerateMismatchReactive[b, t] for b in m._Buses)

    model._LoadMismatchCostReactive = Expression(model._TimePeriods,
                                                rule=compute_q_load_mismatch_cost_rule)


def _add_blank_load_mismatch(model):
    model._LoadGenerateMismatch = Param(model._Buses, model._TimePeriods,
                                       default=0.)
    model._posLoadGenerateMismatch = Param(model._Buses, model._TimePeriods,
                                          default=0.)
    model._negLoadGenerateMismatch = Param(model._Buses, model._TimePeriods,
                                          default=0.)
    model._LoadMismatchCost = Param(model._TimePeriods, default=0.)

def _add_blank_q_load_mismatch(model):
    model._LoadGenerateMismatchReactive = Param(model._Buses, model._TimePeriods,
                                               default=0.)
    model._posLoadGenerateMismatchReactive = Param(model._Buses,
                                                  model._TimePeriods,
                                                  default=0.)
    model._negLoadGenerateMismatchReactive = Param(model._Buses,
                                                  model._TimePeriods,
                                                  default=0.)
    model._LoadMismatchCostReactive = Param(model._TimePeriods, default=0.)


def _add_blank_q_load_mismatch(model):
    model._LoadGenerateMismatchReactive = Param(model._Buses, model._TimePeriods,
                                               default=0.)
    model._posLoadGenerateMismatchReactive = Param(model._Buses,
                                                  model._TimePeriods,
                                                  default=0.)
    model._negLoadGenerateMismatchReactive = Param(model._Buses,
                                                  model._TimePeriods,
                                                  default=0.)
    model._LoadMismatchCostReactive = Param(model._TimePeriods, default=0.)

def _add_system_load_mismatch(model):
    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model._posLoadGenerateMismatch = Var(model._TimePeriods,
                                        within=NonNegativeReals)  # load shedding
    model._negLoadGenerateMismatch = Var(model._TimePeriods,
                                        within=NonNegativeReals)  # over generation

    ## for interfacing with the rest of the model code
    def define_pos_neg_load_generate_mismatch_rule(m, b, t):
        if b == value(m._ReferenceBus):
            return LinearExpression(linear_vars=[m._posLoadGenerateMismatch[t],
                                                 m._negLoadGenerateMismatch[t]],
                                    linear_coefs=[1., -1.])
        else:
            return 0

    model._LoadGenerateMismatch = Expression(model._Buses, model._TimePeriods,
                                            rule=define_pos_neg_load_generate_mismatch_rule)

    # the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
    # the appearance of load generation mismatch component values in the range of *negative* e-5.
    # what these small negative values do is to cause the optimal objective to be a very large negative,
    # due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
    # at this point, but it does seem to work broadly. we tried a single global constraint, across all
    # buses, but that failed to correct the problem, and caused the solve times to explode.

    def pos_load_generate_mismatch_tolerance_rule(m):
        linear_vars = list(m._posLoadGenerateMismatch.values())
        linear_coefs = [1.] * len(linear_vars)
        return (0., LinearExpression(linear_vars=linear_vars,
                                     linear_coefs=linear_coefs), None)

    model._PosLoadGenerateMismatchTolerance = Constraint(
        rule=pos_load_generate_mismatch_tolerance_rule)

    def neg_load_generate_mismatch_tolerance_rule(m):
        linear_vars = list(m._negLoadGenerateMismatch.values())
        linear_coefs = [1.] * len(linear_vars)
        return (0., LinearExpression(linear_vars=linear_vars,
                                     linear_coefs=linear_coefs), None)

    model._NegLoadGenerateMismatchTolerance = Constraint(
        rule=neg_load_generate_mismatch_tolerance_rule)

    def compute_load_mismatch_cost_rule(m, t):
        linear_vars = [*m._posLoadGenerateMismatch.values(),
                       *m._negLoadGenerateMismatch.values()]
        linear_coefs = [m._LoadMismatchPenalty * m._TimePeriodLengthHours] * len(
            linear_vars)
        return LinearExpression(linear_vars=linear_vars,
                                linear_coefs=linear_coefs)

    model._LoadMismatchCost = Expression(model._TimePeriods,
                                        rule=compute_load_mismatch_cost_rule)


def _add_load_mismatch(model):
    over_gen_maxes = {}
    over_gen_times_per_bus = {b: list() for b in model._Buses}

    load_shed_maxes = {}
    load_shed_times_per_bus = {b: list() for b in model._Buses}

    for b in model._Buses:

        # storage, for now, does not
        # have time-vary parameters
        storage_max_injections = 0.
        storage_max_withdraws = 0.

        for s in model._StorageAtBus[b]:
            storage_max_injections += value(model._MaximumPowerOutputStorage[s])
            storage_max_withdraws += value(model._MaximumPowerInputStorage[s])

        for t in model._TimePeriods:
            max_injections = storage_max_injections
            max_withdrawls = storage_max_withdraws

            for g in model._ThermalGeneratorsAtBus[b]:
                p_max = value(model._MaximumPowerOutput[g, t])
                p_min = value(model._MinimumPowerOutput[g, t])

                if p_max > 0:
                    max_injections += p_max
                if p_min < 0:
                    max_withdrawls += -p_min

            for n in model._NondispatchableGeneratorsAtBus[b]:
                p_max = value(model._MaxNondispatchablePower[n, t])
                p_min = value(model._MinNondispatchablePower[n, t])

                if p_max > 0:
                    max_injections += p_max
                if p_min < 0:
                    max_withdrawls += -p_min

            load = value(model._Demand[b, t])
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

    model._OverGenerationBusTimes = Set(dimen=2,
                                       initialize=over_gen_maxes.keys())
    model._LoadSheddingBusTimes = Set(dimen=2,
                                     initialize=load_shed_maxes.keys())

    def get_over_gen_bounds(m, b, t):
        return (0, over_gen_maxes[b, t])

    model._OverGeneration = Var(model._OverGenerationBusTimes,
                               within=NonNegativeReals,
                               bounds=get_over_gen_bounds)  # over generation

    def get_load_shed_bounds(m, b, t):
        return (0, load_shed_maxes[b, t])

    model._LoadShedding = Var(model._LoadSheddingBusTimes,
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
                m._LoadShedding[b, t] for t in load_shed_times_per_bus[b])
            linear_coefs = [1.] * len(linear_vars)
            return (0., LinearExpression(linear_vars=linear_vars,
                                         linear_coefs=linear_coefs), None)
        else:
            return Constraint.Feasible

    model._PosLoadGenerateMismatchTolerance = Constraint(model._Buses,
                                                        rule=pos_load_generate_mismatch_tolerance_rule)

    def neg_load_generate_mismatch_tolerance_rule(m, b):
        if over_gen_times_per_bus[b]:
            linear_vars = list(
                m._OverGeneration[b, t] for t in over_gen_times_per_bus[b])
            linear_coefs = [1.] * len(linear_vars)
            return (0., LinearExpression(linear_vars=linear_vars,
                                         linear_coefs=linear_coefs), None)
        else:
            return Constraint.Feasible

    model._NegLoadGenerateMismatchTolerance = Constraint(model._Buses,
                                                        rule=neg_load_generate_mismatch_tolerance_rule)

    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model._LoadGenerateMismatch = Expression(model._Buses, model._TimePeriods)
    for b in model._Buses:
        for t in model._TimePeriods:
            model._LoadGenerateMismatch[b, t].expr = 0.
    for b, t in model._LoadSheddingBusTimes:
        model._LoadGenerateMismatch[b, t].expr += model._LoadShedding[b, t]
    for b, t in model._OverGenerationBusTimes:
        model._LoadGenerateMismatch[b, t].expr -= model._OverGeneration[b, t]

    model._LoadMismatchCost = Expression(model._TimePeriods)
    for t in model._TimePeriods:
        model._LoadMismatchCost[t].expr = 0.
    for b, t in model._LoadSheddingBusTimes:
        model._LoadMismatchCost[
            t].expr += model._LoadMismatchPenalty * model._TimePeriodLengthHours * \
                       model._LoadShedding[b, t]
    for b, t in model._OverGenerationBusTimes:
        model._LoadMismatchCost[
            t].expr += model._LoadMismatchPenalty * model._TimePeriodLengthHours * \
                       model._OverGeneration[b, t]

def _copperplate_network_model(block, tm, relax_balance=None):

    m, gens_by_bus, bus_p_loads, bus_gs_fixed_shunts = \
            _setup_egret_network_model(block, tm)

    ### declare the p balance
    libbus.declare_eq_p_balance_ed(model=block,
                                   index_set=m._Buses,
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
                                       index_set=m._BranchesWithSlack,
                                       domain=NonNegativeReals,
                                       dense=False)
    libbranch.declare_var_pf_slack_neg(model=block,
                                       index_set=m._BranchesWithSlack,
                                       domain=NonNegativeReals,
                                       dense=False)

def _setup_interface_slacks(m,block,tm):
    # declare the interface slack variables
    # they have a sparse index set
    libbranch.declare_var_pfi_slack_pos(model=block,
                                        index_set=m._InterfacesWithSlack,
                                        domain=NonNegativeReals,
                                        dense=False)
    libbranch.declare_var_pfi_slack_neg(model=block,
                                        index_set=m._InterfacesWithSlack,
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

    branches_in_service = tuple(l for l in m._TransmissionLines if not value(m._LineOutOfService[l,tm]))

    ## this will serve as a key into our dict of PTDF matricies,
    ## so that we can avoid recalculating them each time step
    ## with the same network topology
    branches_out_service = tuple(l for l in m._TransmissionLines if value(m._LineOutOfService[l,tm]))

    return buses, branches, branches_in_service, branches_out_service, interfaces, contingencies

def _setup_egret_network_model(block, tm):
    m = block.parent_block()

    ## this is not the "real" gens by bus, but the
    ## index of net injections from the UC model
    gens_by_bus = block.gens_by_bus

    ### declare (and fix) the loads at the buses
    bus_p_loads = {b: value(m._Demand[b,tm]) for b in m._Buses}

    block.pl = Param(m._Buses, initialize=bus_p_loads)

    bus_gs_fixed_shunts = m._bus_gs_fixed_shunts

    return m, gens_by_bus, bus_p_loads, bus_gs_fixed_shunts

def _ptdf_dcopf_network_model(block, tm):
    m, gens_by_bus, bus_p_loads, bus_gs_fixed_shunts = \
        _setup_egret_network_model(block, tm)

    buses, branches, \
    branches_in_service, branches_out_service, \
    interfaces, contingencies = _setup_egret_network_topology(m, tm)

    ptdf_options = m._ptdf_options

    libbus.declare_var_p_nw(block, m._Buses)

    ### declare net withdraw expression for use in PTDF power flows
    libbus.declare_eq_p_net_withdraw_at_bus(model=block,
                                            index_set=m._Buses,
                                            bus_p_loads=bus_p_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                            )

    ### declare the p balance
    libbus.declare_eq_p_balance_ed(model=block,
                                   index_set=m._Buses,
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
    block._contingency_set = Set(within=m._Contingencies * m._TransmissionLines)
    block.pfc = Expression(block._contingency_set)
    _setup_contingency_slacks(m, block, tm)

    ### Get the PTDF matrix from cache, from file, or create a new one
    ### m._PTDFs set in uc_model_generator
    if branches_out_service not in m._PTDFs:
        buses_idx = tuple(buses.keys())

        reference_bus = value(m._ReferenceBus)

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
                                                       m._BranchViolationCost[
                                                           tm]
                                                       )
        ### declare the "blank" interface flow limits
        libbranch.declare_ineq_p_interface_bounds(model=block,
                                                  index_set=interfaces.keys(),
                                                  interfaces=interfaces,
                                                  approximation_type=None,
                                                  slacks=True,
                                                  slack_cost_expr=
                                                  m._InterfaceViolationCost[tm]
                                                  )
        ### declare the "blank" interface flow limits
        libbranch.declare_ineq_p_contingency_branch_thermal_bounds(model=block,
                                                                   index_set=block._contingency_set,
                                                                   pc_thermal_limits=None,
                                                                   approximation_type=None,
                                                                   slacks=True,
                                                                   slack_cost_expr=
                                                                   m._ContingencyViolationCost[
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
                                                       m._BranchViolationCost[
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
                                                  m._InterfaceViolationCost[tm]
                                                  )

def ptdf_power_flow(model, slacks=True):
    _add_egret_power_flow(model, _ptdf_dcopf_network_model, reactive_power=False, slacks=slacks)

## Defines generic interface for egret tramsmission models
def _add_egret_power_flow(model, network_model_builder, reactive_power=False, slacks=True):

    ## save flag for objective
    model._reactive_power = reactive_power

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
            raise Exception('_add_blank_system_load_mismatch() is not defined in Egret')
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
    model._TransmissionBlock = Block(model._TimePeriods, concrete=True)

    # for interface violation costs at a time step
    model._BranchViolationCost = Expression(model._TimePeriods, rule=lambda m,t:0.)

    # for interface violation costs at a time step
    model._InterfaceViolationCost = Expression(model._TimePeriods, rule=lambda m,t:0.)

    # for contingency violation costs at a time step
    model._ContingencyViolationCost = Expression(model._TimePeriods, rule=lambda m,t:0.)

    for tm in model._TimePeriods:
        b = model._TransmissionBlock[tm]
        ## this creates a fake bus generator for all the
        ## appropriate injection/withdraws from the unit commitment
        ## model
        b.pg = Expression(model._Buses, rule=_get_pg_expr_rule(tm))
        if reactive_power:
            b.qg = Expression(model._Buses, rule=_get_qg_expr_rule(tm))
        b.gens_by_bus = {bus : [bus] for bus in model._Buses}
        network_model_builder(b,tm)