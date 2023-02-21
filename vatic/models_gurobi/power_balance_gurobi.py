from gurobipy import tupledict, LinExpr, GRB, quicksum
import gurobipy as gp

import egret.model_library.transmission.branch as libbranch
import egret.data.ptdf_utils as ptdf_utils
import egret.common.lazy_ptdf_utils as lpu
from egret.model_library.defn import BasePointType, CoordinateType, ApproximationType

from .power_vars_gurobi import _add_reactive_power_vars
from .generation_limits_gurobi import _add_reactive_limits
import vatic.models_gurobi.transmission_gurobi.bus as libbus
import vatic.models_gurobi.transmission_gurobi.branch as libbranch
from vatic.models_gurobi.params_gurobi import default_params
from vatic.models_gurobi.power_vars_gurobi import _add_power_generated_startup_shutdown
from vatic.models_gurobi.non_dispatchable_vars_gurobi import file_non_dispatchable_vars

component_name = 'power_balance'

def _get_pg_expr_rule(t, m, bus):
    return quicksum(m._PowerGeneratedStartupShutdown[g, t] for g in m._ThermalGeneratorsAtBus[bus]) \
                + quicksum(m._PowerOutputStorage[s, t] for s in m._StorageAtBus[bus])\
                - quicksum(m._PowerInputStorage[s, t] for s in m._StorageAtBus[bus])\
                + quicksum(m._NondispatchablePowerUsed[g, t] for g in m._NondispatchableGeneratorsAtBus[bus]) \
                + quicksum(m._HVDCLinePower[k,t] for k in m._HVDCLinesTo[bus]) \
                - quicksum(m._HVDCLinePower[k,t] for k in m._HVDCLinesFrom[bus]) \
                + m._LoadGenerateMismatch[bus,t]

## helper defining reacative power injection at a bus
def _get_qg_expr_rule(t, m, bus):
    return sum(m._ReactivePowerGenerated[g, t] for g in m._ThermalGeneratorsAtBus[bus]) \
            + m._LoadGenerateMismatchReactive[bus,t]

def _add_hvdc(model):
    def dc_line_power_bounds_rule(m, k, t):
        if m._HVDCLineOutOfService[k,t]:
            return (0., 0.)
        return (-m._HVDCThermalLimit[k], m._HVDCThermalLimit[k])

    lbs = []
    ubs = []
    for k in model._HVDCLines:
        for t in model._TimePeriods:
            lb, ub = dc_line_power_bounds_rule(model, k, t)
            lbs.append(lb)
            ubs.append(ub)

    model._HVDCLinePower = model.addVars(model._HVDCLines, model._TimePeriods,
                                         lb = lbs, ub = ubs, name='HVDCLinePower')


def _add_q_load_mismatch(model):
    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model._LoadGenerateMismatchReactive = model.addVars(model._Buses, model._TimePeriods,
                                             lb = -GRB.INFINITY, ub = GRB.INFINITY,
                                             name = 'LoadGenerateMismatchReactive')
    model._posLoadGenerateMismatchReactive = model.addVars(model._Buses, model._TimePeriods,
                                                 lb = 0, ub = GRB.INFINITY,
                                                 name = 'posLoadGenerateMismatchReactive')  # load shedding
    model._negLoadGenerateMismatchReactive = model.addVars(model._Buses, model._TimePeriods,
                                                lb=0,
                                                ub=GRB.INFINITY,
                                                name='negLoadGenerateMismatchReactive')  # over generation

    def define_pos_neg_load_generate_mismatch_rule_reactive(m, b, t):
        return m._posLoadGenerateMismatchReactive[b, t] - \
               m._negLoadGenerateMismatchReactive[b, t] \
               == m._LoadGenerateMismatchReactive[b, t]

    model._DefinePosNegLoadGenerateMismatchReactive = model.addConstrs((
                                                        define_pos_neg_load_generate_mismatch_rule_reactive(model, b, t)
                                                        for b in model._Buses
                                                        for t in model._TimePeriods),
                                                        name = 'DefinePosNegLoadGenerateMismatchReactive')


    # the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
    # the appearance of load generation mismatch component values in the range of *negative* e-5.
    # what these small negative values do is to cause the optimal objective to be a very large negative,
    # due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
    # at this point, but it does seem to work broadly. we tried a single global constraint, across all
    # buses, but that failed to correct the problem, and caused the solve times to explode.

    def pos_load_generate_mismatch_tolerance_rule_reactive(m, b):
        return sum((m._posLoadGenerateMismatchReactive[b, t] for t in
                    m._TimePeriods)) >= 0.0

    model._PosLoadGenerateMismatchToleranceReactive = model.addConstrs((pos_load_generate_mismatch_tolerance_rule_reactive(model, b)
                                                                        for b in model._Buses ),name = 'PosLoadGenerateMismatchToleranceReactive')

    def neg_load_generate_mismatch_tolerance_rule_reactive(m, b):
        return sum((m._negLoadGenerateMismatchReactive[b, t] for t in
                    m._TimePeriods)) >= 0.0

    model._NegLoadGenerateMismatchToleranceReactive = model.addConstrs((neg_load_generate_mismatch_tolerance_rule_reactive(model, b)
                                                                        for b in model._Buses ),name = 'NegLoadGenerateMismatchToleranceReactive')

    def compute_q_load_mismatch_cost_rule(m, t):
        return m._LoadMismatchPenaltyReactive * m._TimePeriodLengthHours * sum(
            m._posLoadGenerateMismatchReactive[b, t] +
            m._negLoadGenerateMismatchReactive[b, t] for b in m._Buses)

    model._LoadMismatchCostReactive = {t: compute_q_load_mismatch_cost_rule(model, t)
                                       for t in model._TimePeriods}

def _add_blank_load_mismatch(model):
    model._LoadGenerateMismatch = tupledict({(b, t): 0 for b in model._Buses for t in model._TimePeriods})

    model._posLoadGenerateMismatch = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._negLoadGenerateMismatch = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._LoadMismatchCost = {t: 0 for t in model._TimePeriods}

def _add_blank_q_load_mismatch(model):
    model._LoadGenerateMismatchReactive = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._posLoadGenerateMismatchReactive = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._negLoadGenerateMismatchReactive = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._LoadMismatchCostReactive = {t: 0 for t in model._TimePeriods}


def _add_blank_q_load_mismatch(model):
    model._LoadGenerateMismatchReactive = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._posLoadGenerateMismatchReactive = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._negLoadGenerateMismatchReactive = tupledict({(b, t): 0 for b in model._Buses \
                                                for t in model._TimePeriods})
    model._LoadMismatchCostReactive = {t: 0 for t in model._TimePeriods}
    return model

def _add_system_load_mismatch(model):
    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model._posLoadGenerateMismatch = model.addVars(model._TimePeriods,
                                                   lb = 0,
                                                   ub  = GRB.INFINITY,
                                                   name = 'posLoadGenerateMismatch')
  # load shedding
    model._negLoadGenerateMismatch = model.addVars(model._TimePeriods,
                                                   lb = 0,
                                                   ub  = GRB.INFINITY,
                                                   name = 'negLoadGenerateMismatch')  # over generation

    ## for interfacing with the rest of the model code
    def define_pos_neg_load_generate_mismatch_rule(m, b, t):
        if b == m._ReferenceBus:
            return m._posLoadGenerateMismatch[t]-m._negLoadGenerateMismatch[t]

        else:
            return 0

    model._LoadGenerateMismatch = tupledict({(b, t): define_pos_neg_load_generate_mismatch_rule(model, b, t)
                                                for b in model._Buses for t in model._TimePeriods})

    # the following constraints are necessarily, at least in the case of CPLEX 12.4, to prevent
    # the appearance of load generation mismatch component values in the range of *negative* e-5.
    # what these small negative values do is to cause the optimal objective to be a very large negative,
    # due to obviously large penalty values for under or over-generation. JPW would call this a heuristic
    # at this point, but it does seem to work broadly. we tried a single global constraint, across all
    # buses, but that failed to correct the problem, and caused the solve times to explode.

    def pos_load_generate_mismatch_tolerance_rule(m):
        linear_vars = list(m._posLoadGenerateMismatch.values())
        linear_coefs = [1.] * len(linear_vars)
        return LinExpr(linear_coefs, linear_vars) >= 0

    model._PosLoadGenerateMismatchTolerance = model.addConstr(
        pos_load_generate_mismatch_tolerance_rule(model),
        name = 'PosLoadGenerateMismatchTolerance'
    )

    def neg_load_generate_mismatch_tolerance_rule(m):
        linear_vars = list(m._negLoadGenerateMismatch.values())
        linear_coefs = [1.] * len(linear_vars)
        return (LinExpr(linear_coefs, linear_vars) >= 0)

    model._NegLoadGenerateMismatchTolerance = model.addConstr(
        neg_load_generate_mismatch_tolerance_rule,
        name = 'NegLoadGenerateMismatchTolerance'
    )

    def compute_load_mismatch_cost_rule(m, t):
        linear_vars = [*m._posLoadGenerateMismatch.values(),
                       *m._negLoadGenerateMismatch.values()]
        linear_coefs = [m._LoadMismatchPenalty * m._TimePeriodLengthHours] * len(
            linear_vars)
        return LinExpr(linear_coefs, linear_vars)

    model._LoadMismatchCost = {t: compute_load_mismatch_cost_rule(t)
                                for t in model._TimePeriods}

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
            storage_max_injections += model._MaximumPowerOutputStorage[s]
            storage_max_withdraws += model._MaximumPowerInputStorage[s]

        for t in model._TimePeriods:
            max_injections = storage_max_injections
            max_withdrawls = storage_max_withdraws

            for g in model._ThermalGeneratorsAtBus[b]:
                p_max = model._MaximumPowerOutput[g, t]
                p_min = model._MinimumPowerOutput[g, t]

                if p_max > 0:
                    max_injections += p_max
                if p_min < 0:
                    max_withdrawls += -p_min

            for n in model._NondispatchableGeneratorsAtBus[b]:
                p_max = model._MaxNondispatchablePower[n, t]
                p_min = model._MinNondispatchablePower[n, t]

                if p_max > 0:
                    max_injections += p_max
                if p_min < 0:
                    max_withdrawls += -p_min

            load = model._Demand[b, t]
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

    model._OverGenerationBusTimes = over_gen_maxes.keys()
    model._LoadSheddingBusTimes = load_shed_maxes.keys()

    def get_over_gen_bounds(m, b, t):
        return (0, over_gen_maxes[b, t])

    model._OverGeneration = model.addVars(model._OverGenerationBusTimes,
                                          lb = 0, ub = GRB.INFINITY,
                                          name = 'OverGeneration') # over generation

    def get_load_shed_bounds(m, b, t):
        return (0, load_shed_maxes[b, t])

    model._LoadShedding = model.addVars(model._LoadSheddingBusTimes,
                                        lb=0, ub=GRB.INFINITY,
                                        name='LoadShedding'
                                        )

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
            return (LinExpr(linear_coefs, linear_vars) >= 0)
        else:
            return None

    for bus in model._Buses:
        const = pos_load_generate_mismatch_tolerance_rule(model, bus)
        if const != None:
            model.addConstr(
                const, name = 'PosLoadGenerateMismatchTolerance[{}]'.format(bus))

    def neg_load_generate_mismatch_tolerance_rule(m, b):
        if over_gen_times_per_bus[b]:
            linear_vars = list(
                m._OverGeneration[b, t] for t in over_gen_times_per_bus[b])
            linear_coefs = [1.] * len(linear_vars)
            return (LinExpr(linear_coefs, linear_vars) >= 0)
        else:
            return None

    for bus in model._Buses:
        const = neg_load_generate_mismatch_tolerance_rule(model, bus)
        if const != None:
            model.addConstr(
                const, name = 'NegLoadGenerateMismatchTolerance[{}]'.format(bus))

    #####################################################
    # load "shedding" can be both positive and negative #
    #####################################################
    model._LoadGenerateMismatch = {}
    for b in model._Buses:
        for t in model._TimePeriods:
            model._LoadGenerateMismatch[b, t] = 0.
    for b, t in model._LoadSheddingBusTimes:
        model._LoadGenerateMismatch[b, t] += model._LoadShedding[b, t]
    for b, t in model._OverGenerationBusTimes:
        model._LoadGenerateMismatch[b, t]= model._OverGeneration[b, t]

    model._LoadMismatchCost = {}
    for t in model._TimePeriods:
        model._LoadMismatchCost[t] = 0.
    for b, t in model._LoadSheddingBusTimes:
        model._LoadMismatchCost[
            t] += model._LoadMismatchPenalty * model._TimePeriodLengthHours * \
                       model._LoadShedding[b, t]
    for b, t in model._OverGenerationBusTimes:
        model._LoadMismatchCost[
            t] += model._LoadMismatchPenalty * model._TimePeriodLengthHours * \
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
    block._pf_slack_pos = block.addVars(m._BranchesWithSlack, lb = 0, ub = GRB.INFINITY,
                    name = 'pf_slack_pos')

    block._pf_slack_pos = block.addVars(m._BranchesWithSlack, lb = 0, ub = GRB.INFINITY,
                    name = 'pf_slack_neg')

def _setup_interface_slacks(m,block,tm):
    # declare the interface slack variables
    # they have a sparse index set
    block._pfi_slack_pos = block.addVars(m._InterfacesWithSlack, lb = 0, ub = GRB.INFINITY,
                    name = 'pfi_slack_pos')

    block._pfi_slac_neg =  block.addVars(m._InterfacesWithSlack, lb = 0, ub = GRB.INFINITY,
                    name = 'pfi_slack_neg')

def _setup_contingency_slacks(m,block,tm):
    # declare the interface slack variables
    # they have a sparse index set

    block._pfc_slack_pos = block.addVars(m._Contingencies, m._TransmissionLines, lb=0, ub=GRB.INFINITY,
                  name='pfc_slack_pos')

    block._pfc_slack_neg = block.addVars(m._Contingencies, m._TransmissionLines, lb=0, ub=GRB.INFINITY,
                  name='pfc_slack_neg')

def _setup_egret_network_topology(m,tm):
    buses = m._buses
    branches = m._branches
    interfaces = m._interfaces
    contingencies = m._contingencies

    branches_in_service = tuple(l for l in m._TransmissionLines if not m._LineOutOfService[l,tm])

    ## this will serve as a key into our dict of PTDF matricies,
    ## so that we can avoid recalculating them each time step
    ## with the same network topology
    branches_out_service = tuple(l for l in m._TransmissionLines if m._LineOutOfService[l,tm])

    return buses, branches, branches_in_service, branches_out_service, interfaces, contingencies

def _setup_egret_network_model(block, tm):
    m = block

    ## this is not the "real" gens by bus, but the
    ## index of net injections from the UC model
    gens_by_bus = block._gens_by_bus

    ### declare (and fix) the loads at the buses
    bus_p_loads = {b: m._Demand[b,tm] for b in m._Buses}

    block._pl = bus_p_loads

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
    block._branches_inservice = branches_in_service

    _setup_branch_slacks(m, block, tm)

    ### interface setup
    block._interfae_keys = interfaces.keys()

    _setup_interface_slacks(m, block, tm)

    ### contingency setup
    ### NOTE: important that this not be dense, we'll add elements
    ###       as we find violations
    block._contingency_set = [(c, i) for c in m._Contingencies for i in m._TransmissionLines]
    # block._pfc = Expression(block._contingency_set)
    _setup_contingency_slacks(m, block, tm)

    ### Get the PTDF matrix from cache, from file, or create a new one
    ### m._PTDFs set in uc_model_generator
    if branches_out_service not in m._PTDFs:
        buses_idx = tuple(buses.keys())

        reference_bus = m._ReferenceBus

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

        # ### add helpers for tracking monitored branches
        # lpu.add_monitored_flow_tracker(block)
        #
        # ### add initial branches to monitored set
        # lpu.add_initial_monitored_branches(block, branches,
        #                                    branches_in_service, ptdf_options,
        #                                    PTDF)
        #
        # ### add initial interfaces to monitored set
        # lpu.add_initial_monitored_interfaces(block, interfaces, ptdf_options,
        #                                      PTDF)

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
    return model

# Defines generic interface for egret tramsmission models
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

    # for interface violation costs at a time step
    model._BranchViolationCost = {t: 0 for t in model._TimePeriods}

    # for interface violation costs at a time step
    model._InterfaceViolationCost = {t: 0 for t in model._TimePeriods}

    # for contingency violation costs at a time step
    model._ContingencyViolationCost = {t: 0 for t in model._TimePeriods}


    for tm in model._TimePeriods:
        b = default_params(model, model._model_data_vatic)
        b = model.copy()

        # The deep copy does not copy private attributes starting with _
        b._ThermalGenerators = model._ThermalGenerators
        b._TimePeriods = model._TimePeriods
        b._ThermalGeneratorsAtBus = model._ThermalGeneratorsAtBus
        # b._PowerOutputStorage = model._PowerOutputStorage
        b._StorageAtBus = model._StorageAtBus

        # b._PowerInputStorage = model._PowerInputStorage

        b._NondispatchableGeneratorsAtBus = model._NondispatchableGeneratorsAtBus
        b._HVDCLinePower = model._HVDCLinePower
        b._HVDCLinesTo = model._HVDCLinesTo
        b._HVDCLinesFrom = model._HVDCLinesTo

        # b._ReactivePowerGenerated = model._ReactivePowerGenerated
        # b._LoadGenerateMismatchReactive = model._LoadGenerateMismatchReactive
        b._ptdf_options = model._ptdf_options
        b._Buses = model._Buses
        b._Contingencies = model._Contingencies
        b._contingencies = model._contingencies
        b._TransmissionLines = model._TransmissionLines
        b._PTDFs = model._PTDFs
        b._ReferenceBus = model._ReferenceBus
        b._BranchViolationCost = model._BranchViolationCost
        b._InterfaceViolationCost = model._InterfaceViolationCost
        # b._gens_by_bus = model._gens_by_bus
        b._Demand = model._Demand
        b._bus_gs_fixed_shunts = model._bus_gs_fixed_shunts
        b._buses = model._buses
        b._branches = model._branches
        b._interfaces = model._interfaces
        b._LineOutOfService = model._LineOutOfService
        b._InterfacesWithSlack = model._InterfacesWithSlack
        b._BranchesWithSlack = model._BranchesWithSlack
        b._InitialTime = model._InitialTime



        # Add private attributes needed in the function below
        # The gurobi variables and thus the related expressions could not be copied; must recreated
        b._PowerGeneratedStartupShutdown = tupledict({(g, t): _add_power_generated_startup_shutdown(b, g, t)
                                         for g in b._ThermalGenerators for t in b._TimePeriods})
        b = file_non_dispatchable_vars(b)
        b = _add_blank_load_mismatch(model)

        b._PowerGeneratedAboveMinimum = {}
        for g in model._ThermalGenerators:
            for t in model._TimePeriods:
                b._PowerGeneratedAboveMinimum = \
                    b.getVarByName('PowerGeneratedAboveMinimum[{}, {}]').format(g, t)

        b._pg = {bus: _get_pg_expr_rule(tm, b, bus) for bus in b._Buses}
        if reactive_power:
            b._qg = {bus: _get_qg_expr_rule(tm, b, bus) for bus in b._Buses}
        b._gens_by_bus = {bus : [bus] for bus in b._Buses}
        network_model_builder(b,tm)

