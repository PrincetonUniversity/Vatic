import math

import gurobipy as gp
import pyomo.environ as pe
from gurobipy import tupledict, tuplelist
from ordered_set import OrderedSet

from egret.model_library.transmission import tx_utils
from egret.common.log import logger
from egret.model_library.unit_commitment.uc_utils import (
    add_model_attr, uc_time_helper)
from vatic.models._utils import ModelError
from vatic.model_data import VaticModelData
from typing import Optional

def load_base_params(
        model: gp.Model,
        model_data: Optional[VaticModelData] = None, renew_costs: bool = False
        ) -> gp.Model:
    warn_neg_load = False
    time_keys = model_data.get_system_attr('time_keys')

    ## NOTE: generator, bus, and load should be in here for
    # a well-defined problem
    loads = tupledict(model_data.elements(element_type='load'))
    gens = tupledict(model_data.elements(element_type='generator'))

    thermal_gens = tupledict(model_data.elements(element_type='generator',
                                            generator_type='thermal'))
    renewable_gens = tupledict(model_data.elements(element_type='generator',
                                              generator_type='renewable'))

    buses = tupledict(model_data.elements(element_type='bus'))
    shunts = tupledict()
    branches = tupledict(model_data.elements(element_type='branch'))
    interfaces = tupledict(model_data.elements(element_type='interface'))
    contingencies = tupledict()
    storage = tupledict(model_data.elements(element_type='storage'))
    dc_branches = tupledict()

    thermal_gen_attrs = model_data.attributes(element_type='generator',
                                              generator_type='thermal')
    renewable_gen_attrs = model_data.attributes(element_type='generator',
                                                generator_type='renewable')

    bus_attrs = model_data.attributes(element_type='bus')
    branch_attrs = model_data.attributes(element_type='branch')
    interface_attrs = model_data.attributes(element_type='interface')

    storage_attrs = model_data.attributes(element_type='storage')
    storage_by_bus = tx_utils.gens_by_bus(buses, storage)

    dc_branch_attrs = tupledict(names=list())

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(dc_branches, buses)

    thermal_gens_by_bus = tx_utils.gens_by_bus(buses, thermal_gens)
    renewable_gens_by_bus = tx_utils.gens_by_bus(buses, renewable_gens)

    ### get the fixed shunts at the buses
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = \
        tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ## attach some of these to the model object for ease/speed later
    # model._loads = loads
    model._buses = buses
    model._branches = branches
    model._shunts =shunts
    model._bus_gs_fixed_shunts = bus_gs_fixed_shunts
    model._interfaces = interfaces
    model._contingencies = contingencies
    model._dc_branches = dc_branches

    #
    # Parameters
    #

    ##############################################
    # string indentifiers for the OrderedSet of busses. #
    ##############################################

    model._Buses = OrderedSet(bus_attrs['names'])
    ref_bus = model_data.get_system_attr('reference_bus', '')
    if not ref_bus or ref_bus not in model._Buses:
        ref_bus = model._Buses[0]

    ##Initialize Gurobi Model Parameters, not Variables
    model._ReferenceBus = ref_bus
    ref_angle = model_data.get_system_attr('reference_bus_angle', 0.)
    model._ReferenceBusAngle = ref_angle

    ## in minutes, assert that this must be a positive integer
    model._TimePeriodLengthMinutes = model_data.get_system_attr('time_period_length_minutes')

    ## IN HOURS, assert athat this must be a positive number
    model._TimePeriodLengthHours = model._TimePeriodLengthMinutes/60

    model._NumTimePeriods = len(time_keys)

    model._InitialTime = 1
    #Pyomo start periods from 1
    model._TimePeriods = [i for i in range(model._InitialTime, model._NumTimePeriods+1)]
    TimeMapper = uc_time_helper(model._TimePeriods)

    ## For now, hard code these. Probably need to be able to specify in model_data
    model._StageSet = ['Stage_1', 'Stage_2']

    # the following sets must must come from the data files or from an initialization function that uses
    # a parameter that tells you when the stages end (and that thing needs to come from the data files)

    model._CommitmentTimeInStage = {'Stage_1': model._TimePeriods, 'Stage_2': list()}

    model._GenerationTimeInStage = {'Stage_1': list(), 'Stage_2': model._TimePeriods}


    ##############################################
    # Network definition (S)
    ##############################################

    model._TransmissionLines = branch_attrs['names']
    model._HVDCLines = dc_branch_attrs['names']

    model._BusFrom = tupledict(branch_attrs.get('from_bus', {}))
    model._BusTo = tupledict(branch_attrs.get('to_bus', {}))

    model._HVDCBusFrom = tupledict(dc_branch_attrs.get('from_bus', {}))

    model._HVDCBusTo = tupledict(dc_branch_attrs.get('to_bus', {}))


    model._LinesTo = tupledict(inlet_branches_by_bus)
    model._LinesFrom = tupledict(outlet_branches_by_bus)

    model._HVDCLinesTo = tupledict(dc_inlet_branches_by_bus)
    model._HVDCLinesFrom = tupledict(dc_outlet_branches_by_bus)

    model._Impedence = tupledict(branch_attrs.get('reactance', {}))

    model._ThermalLimit = tupledict(branch_attrs.get('rating_long_term', {}))  # max flow across the line

    model._HVDCThermalLimit = tupledict(dc_branch_attrs.get('rating_long_term', {})) # max flow across the line

    #default value is False
    #TiemMapper will automatically add t
    model._LineOutOfService = tupledict(TimeMapper(branch_attrs.get('planned_outage', {l: False for l in model._TransmissionLines})))

    model._HVDCLineOutOfService = tupledict(TimeMapper(dc_branch_attrs.get('planned_outage', {l: False for l in model._HVDCLines})))



    _branch_penalties = tupledict()
    _md_violation_penalties = branch_attrs.get('violation_penalty')

    if _md_violation_penalties is not None:
        for i, val in _md_violation_penalties.items():
            if val is not None:
                _branch_penalties[i] = val

                if val <= 0:
                    logger.warning(
                        "Branch {} has a non-positive penalty {}, this will "
                        "cause its limits to be ignored!".format(i, val)
                    )

    model._BranchesWithSlack = _branch_penalties.keys()

    model._BranchLimitPenalty = _branch_penalties

    ## Interfaces
    model._Interfaces = interface_attrs['names']
    model._InterfaceLines = dict(interface_attrs.get('lines', {}))

    model._InterfaceMinFlow = tupledict(interface_attrs.get('minimum_limit', {}))

    model._InterfaceMaxFlow = tupledict(interface_attrs.get('maximum_limit', {}))


    def check_min_less_max_interface_flow_limits(m):
        for intfc in m._Interfaces:
            if m._InterfaceMinFlow[intfc] > m._InterfaceMaxFlow[intfc]:
                raise ModelError(
                    "Interface {} has a minimum_limit which is greater than "
                    "the maximum_limit".format(intfc)
                )

    check_min_less_max_interface_flow_limits(model)

    _interface_line_orientation_dict = tupledict()
    for intfc, interface in interfaces.items():
        for line, sign in zip(interface['lines'],
                              interface['line_orientation']):
            _interface_line_orientation_dict[intfc, line] = sign

    model._InterfaceLineOrientation = _interface_line_orientation_dict

    _interface_penalties = tupledict()
    _md_violation_penalties = tupledict(interface_attrs.get('violation_penalty', {}))

    if _md_violation_penalties is not None:
        for intfc, val in _md_violation_penalties.items():
            if val is not None:
                _interface_penalties[intfc] = val

                if val <= 0:
                    logger.warning(
                        "Interface {} has a non-positive penalty {}, this "
                        "will cause its limits to be ignored!".format(
                            intfc, val)
                    )




    model._InterfacesWithSlack = _interface_penalties.keys()
    model._InterfaceLimitPenalty = _interface_penalties

    ##########################################################
    # string indentifiers for the set of thermal generators. #
    # and their locations. (S)                               #
    ##########################################################
    model._ThermalGenerators = OrderedSet(thermal_gen_attrs['names'])
    model._ThermalGeneratorsAtBus = thermal_gens_by_bus

    #default type is 'C'
    model._ThermalGeneratorType = tupledict(thermal_gen_attrs.get('fuel', {'g': 'C' for g in model._ThermalGenerators}))


    def verify_thermal_generator_buses_rule(m):
        assert set(m._ThermalGenerators) == {
            gen for bus in m._Buses for gen in m._ThermalGeneratorsAtBus[bus]}

    verify_thermal_generator_buses_rule(model)
    model._QuickStart = tupledict(thermal_gen_attrs.get('fast_start', {'g': False for g in model._ThermalGenerators}))

    def init_quick_start_generators(m):
        return [g for g in m._ThermalGenerators
                if m._QuickStart[g] == 1]

    model._QuickStartGenerators = OrderedSet(init_quick_start_generators(model))

    # optionally force a unit to be on/off
    model._FixedCommitmentTypes = [0, 1, None]

    #TimeMapper will automatically add t key
    model._FixedCommitment = tupledict(TimeMapper(thermal_gen_attrs.get('fixed_commitment', {g: None for g in model._ThermalGenerators}
    )))

    model._AllNondispatchableGenerators = OrderedSet(renewable_gen_attrs['names'])
    model._NondispatchableGeneratorsAtBus = renewable_gens_by_bus

    model._NondispatchableGeneratorType = tupledict(renewable_gen_attrs.get('fuel', {}))

    def verify_renew_generator_buses_rule(m):
        assert set(m._AllNondispatchableGenerators) == {
            gen for bus in m._Buses
            for gen in m._NondispatchableGeneratorsAtBus[bus]
        }

    verify_renew_generator_buses_rule(model)

    #################################################################
    # the global system demand, for each time period. units are MW. #
    # demand as at busses (S) so total demand is derived            #
    #################################################################

    # at the moment, we allow for negative demand. this is probably
    # not a good idea, as "stuff" representing negative demand - including
    # renewables, interchange schedules, etc. - should probably be modeled
    # explicitly.

    bus_loads = tupledict({(b, t): 0
                 for b in bus_attrs['names'] for t in model._TimePeriods})

    for lname, load in loads.items():
        load_time = TimeMapper(load['p_load'])
        bus = load['bus']

        if isinstance(bus, tupledict):
            assert bus['data_type'] == 'load_distribution_factor'

            for bn, multi in bus['values'].items():
                for t in model._TimePeriods:
                    bus_loads[bn, t] += multi * load_time[t]

        else:
            for t in model._TimePeriods:
                bus_loads[bus, t] += load_time[t]

    model._Demand = bus_loads

    #sum over demand of buses at each hr of 24 hrs
    def calculate_total_demand(m, t):
        return sum(m._Demand[b, t] for b in sorted(m._Buses))
    model._TotalDemand = tupledict({t: calculate_total_demand(model, t) for t in model._TimePeriods})



    # at this point, a user probably wants to see if they have negative demand.
    def warn_about_negative_demand_rule(m):
        for b in m._Buses:
            for t in m._TimePeriods:
                this_demand = m._Demand[(b, t)]

                if this_demand < 0.0:
                    logger.warning(
                        "***WARNING: The demand at bus `{}` for time period <{}> is "
                        "negative - value {:.4f}}; model={}".format(
                            b, t, this_demand, m._name)
                    )

    warn_about_negative_demand_rule(model)

    ##################################################################
    # the global system reserve, for each time period. units are MW. #
    ##################################################################

    reserve_req = tupledict(TimeMapper(
        model_data.get_system_attr('reserve_requirement', 0.)))

    model._ReserveRequirement = reserve_req

    ##########################################################################################################
    # the minimum number of time periods that a generator must be on-line (off-line) once brought up (down). #
    ##########################################################################################################

    model._MinimumUpTime = thermal_gen_attrs['min_up_time']

    model._MinimumDownTime = thermal_gen_attrs['min_down_time']


    ## Assert that MUT and MDT are at least 1 in the time units of the model.
    ## Otherwise, turn on/offs may not be enforced correctly.
    def scale_min_uptime(m, g):
        scaled_up_time = int(math.ceil(m._MinimumUpTime[g]
                                       / m._TimePeriodLengthHours))

        return min(max(scaled_up_time, 1), m._NumTimePeriods)

    model._ScaledMinimumUpTime = tupledict({g: scale_min_uptime(model, g) for g in model._ThermalGenerators})


    def scale_min_downtime(m, g):
        scaled_down_time = int(math.ceil(m._MinimumDownTime[g]
                                         / m._TimePeriodLengthHours))

        return min(max(scaled_down_time, 1), m._NumTimePeriods)

    model._ScaledMinimumDownTime = tupledict({g: scale_min_downtime(model, g) for g in model._ThermalGenerators})
    ####################################################################################
    # minimum and maximum generation levels, for each thermal generator. units are MW. #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################

    # you can enter generator limits either once for the generator or for each period (or just take 0)

    if renew_costs:
        cost_gens = (model._ThermalGenerators
                     | model._AllNondispatchableGenerators)
        cost_attrs = model_data.attributes(element_type='generator')

    else:
        cost_gens = model._ThermalGenerators
        cost_attrs = thermal_gen_attrs

    model._MinimumPowerOutput = TimeMapper(cost_attrs['p_min'])

    model._MaximumPowerOutput = TimeMapper(cost_attrs['p_max'])


    model._MinimumReactivePowerOutput = tupledict(TimeMapper(thermal_gen_attrs.get('q_min', {g: 0 for g in model._ThermalGenerators})))

    model._MaximumReactivePowerOutput = tupledict(TimeMapper(thermal_gen_attrs.get('q_max', {g: 0 for g in model._ThermalGenerators})))

    # wind is similar, but max and min will be equal for non-dispatchable wind

    model._MinNondispatchablePower = tupledict(TimeMapper(renewable_gen_attrs.get('p_min', {g: 0 for g in model._ThermalGenerators})))

    model._MaxNondispatchablePower = tupledict(TimeMapper(renewable_gen_attrs.get('p_max', {g: 0 for g in model._ThermalGenerators})))


    #################################################
    # generator ramp up/down rates. units are MW/h. #
    # IMPORTANT: Generator ramp limits can exceed   #
    # the maximum power output, because it is the   #
    # ramp limit over an hour. If the unit can      #
    # fully ramp in less than an hour, then this    #
    # will occur.                                   #
    #################################################

    ## be sure the generator can ramp
    ## between all the p_min/p_max values
    def ramp_up_validator(m):
        for g in m._ThermalGenerators:
            t1 = m._InitialTime

            for t in m._TimePeriods:
                if t == t1:
                    continue

                diff = m._MinimumPowerOutput[g, t]- m._MaximumPowerOutput[g, t - 1]

                if m._NominalRampUpLimit[g] * m._TimePeriodLengthHours < diff:
                    logger.error("Generator {} has an infeasible ramp up between "
                                 "time periods {} and {}".format(g, t - 1, t))

                    return False
        return True


    ## be sure the generator can ramp
    ## between all the p_min/p_max values
    #v is the parameter value
    def ramp_down_validator(m):
        for g in m._ThermalGenerators:
            t1 = m._InitialTime
            for t in m._TimePeriods:
                if t == t1:
                    continue

                diff = m._MinimumPowerOutput[g, t - 1] - m._MaximumPowerOutput[g, t]

                if m._NominalRampDownLimit[g] * m._TimePeriodLengthHours < diff:
                    logger.error(
                        "Generator {} has an infeasible ramp down between time "
                        "periods {} and {}".format(g, t - 1, t)
                    )

                    return False

        return True


    # limits for normal time periods
    model._NominalRampUpLimit = thermal_gen_attrs['ramp_up_60min']


    model._NominalRampDownLimit = thermal_gen_attrs['ramp_down_60min']

    ramp_up_validator(model)
    ramp_down_validator(model)


    #############################################
    # unit on state at t=0 (initial condition). #
    #############################################

    # if positive, the number of hours prior to (and including) t=0 that the unit has been on.
    # if negative, the number of hours prior to (and including) t=0 that the unit has been off.
    # the value cannot be 0, by definition.

    model._UnitOnT0State = thermal_gen_attrs['initial_status']

    if not all([i!=0 for i in model._UnitOnT0State.values()]):
        logger.error('The Unit on status value must be nonzero')

    def t0_unit_on_rule(m, g):
        return int(m._UnitOnT0State[g] > 0.)


    model._UnitOnT0 = tupledict({g: t0_unit_on_rule(model, g) for g in model._ThermalGenerators})


    #######################################################################################
    # the number of time periods that a generator must initally on-line (off-line) due to #
    # its minimum up time (down time) constraint.                                         #
    #######################################################################################

    def initial_time_periods_online_rule(m, g):
        if not m._UnitOnT0[g]:
            return 0
        else:
            return int(min(m._NumTimePeriods,
                 round(max(0, m._MinimumUpTime[g] - m._UnitOnT0State[g]) / m._TimePeriodLengthHours)))

    model._InitialTimePeriodsOnLine = tupledict({g: initial_time_periods_online_rule(model, g) for g in model._ThermalGenerators})

    def initial_time_periods_offline_rule(m, g):
        if m._UnitOnT0[g]:
            return 0
        else:
            return int(min(m._NumTimePeriods,
                           round(max(0, m._MinimumDownTime[g] + m._UnitOnT0State[g]) / m._TimePeriodLengthHours)))

    model._InitialTimePeriodsOffLine = tupledict({g: initial_time_periods_offline_rule(model, g) for g in model._ThermalGenerators})


    # ensure that the must-run flag and the t0 state are consistent. in partcular, make
    # sure that the unit has satisifed its minimum down time condition if UnitOnT0 is negative.

    def verify_must_run_t0_state_consistency_rule(m, g):
        t0_state = m._UnitOnT0State[g] / m._TimePeriodLengthHours
        if t0_state < 0:
            min_down_time = m._ScaledMinimumDownTime[g]
            if abs(t0_state) < min_down_time:
                for t in range(m._TimePeriods[0], m._InitialTimePeriodsOffLine[g] + m._TimePeriods[0]):
                    fixed_commitment = m._FixedCommitment[g, t]
                    if (fixed_commitment is not None) and (fixed_commitment == 1):
                        print(
                            "DATA ERROR: The generator %s has been flagged as must-run at time %d, but its T0 state=%d is inconsistent with its minimum down time=%d" % (
                            g, t, t0_state, min_down_time))
                        return False
        else:  # t0_state > 0
            min_up_time = m._ScaledMinimumUpTime[g]
            if abs(t0_state) < min_up_time:
                for t in range(m._TimePeriods[0], m._InitialTimePeriodsOnLine[g] + m._TimePeriods[0]):
                    fixed_commitment = m._FixedCommitment[g, t]
                    if (fixed_commitment is not None) and (fixed_commitment == 0):
                        print(
                            "DATA ERROR: The generator %s has been flagged as off at time %d, but its T0 state=%d is inconsistent with its minimum up time=%d" % (
                            g, t, t0_state, min_up_time))
                        return False
        return True

    model._InitialTimePeriodsOnLine = tupledict({g: initial_time_periods_online_rule(model, g) for g in model._ThermalGenerators})
    model._InitialTimePeriodsOffLine = tupledict({g: initial_time_periods_offline_rule(model, g) for g in model._ThermalGenerators})
    model._VerifyMustRunT0StateConsistency = tupledict({g: verify_must_run_t0_state_consistency_rule(model, g) for g in model._ThermalGenerators})



    # For future shutdowns/startups beyond the time-horizon
    # Like UnitOnT0State, a postive quantity means the generator
    # *will start* in 'future_status' hours, and a negative quantity
    # means the generator *will stop* in -('future_status') hours.
    # The default of 0 means we have no information
    model._FutureStatus = tupledict(thermal_gen_attrs.get('future_status', {g: 0 for g in model._ThermalGenerators}))


    def time_periods_since_last_shutdown_rule(m, g):
        if m._UnitOnT0[g]:
            # longer than any time-horizon we'd consider
            return 10000
        else:
            return int(math.ceil(-m._UnitOnT0State[g]/ m._TimePeriodLengthHours))


    model._TimePeriodsSinceShutdown = tupledict({g: time_periods_since_last_shutdown_rule(model, g) for g in model._ThermalGenerators})


    def time_periods_before_startup_rule(m, g):
        if m._FutureStatus[g] <= 0:
            # longer than any time-horizon we'd consider
            return 10000
        else:
            return int(math.ceil(m._FutureStatus[g]/m._TimePeriodLengthHours))


    model._TimePeriodsBeforeStartup = tupledict({g: time_periods_before_startup_rule(model, g) for g in model._ThermalGenerators})


    ###############################################
    # startup/shutdown curves for each generator. #
    # These are specified in the same time scales #
    # as 'time_period_length_minutes' and other   #
    # time-vary quantities.                       #
    ###############################################

    def startup_curve_init_rule(m, g):
        startup_curve = thermal_gens[g].get('startup_curve')

        if startup_curve is None:
            return ()

        min_down_time = int(math.ceil(m._MinimumDownTime[g]
                                      / m._TimePeriodLengthHours))

        if len(startup_curve) > min_down_time:
            logger.warn(
                "Truncating startup_curve longer than scaled minimum down "
                "time {} for generator {}".format(min_down_time, g)
            )

        return startup_curve[0:min_down_time]

    model._StartupCurve = {g: startup_curve_init_rule(model, g) for g in model._ThermalGenerators}


    def shutdown_curve_init_rule(m, g):
        shutdown_curve = thermal_gens[g].get('shutdown_curve')

        if shutdown_curve is None:
            return ()

        min_down_time = int(math.ceil(m._MinimumDownTime[g]
                                      / m._TimePeriodLengthHours))

        if len(shutdown_curve) > min_down_time:
            logger.warn(
                "Truncating shutdown_curve longer than scaled minimum down "
                "time {} for generator {}".format(min_down_time, g)
            )

        return shutdown_curve[0:min_down_time]

    model._ShutdownCurve = {g: shutdown_curve_init_rule(model, g) for g in model._ThermalGenerators}

    ####################################################################
    # generator power output at t=0 (initial condition). units are MW. #
    ####################################################################

    def power_generated_t0_validator(m):
        for g in m._ThermalGenerators:
            v = m._PowerGeneratedT0[g]
            t = m._TimePeriods[0]

            if m._UnitOnT0[g]:
                v_less_max = v <= m._MaximumPowerOutput[g, t]+ m._NominalRampDownLimit[g]* m._TimePeriodLengthHours

                if not v_less_max:
                    logger.error("Generator {} has more output at T0 than is "
                                 "feasible to ramp down to".format(g))
                    return False

                # TODO: double-check that `if not v_less_max` in original is wrong
                v_greater_min = v >=m._MinimumPowerOutput[g, t] - m._NominalRampUpLimit[g]* m._TimePeriodLengthHours

                if not v_less_max:
                    logger.error("Generator {} has less output at T0 than is "
                                 "feasible to ramp up to".format(g))
                    return False

                return True

            else:
                # Generator was off, but could have residual power due to
                # start-up/shut-down curve. Therefore, do not be too picky
                # as the value doesn't affect any constraints directly
                return True

    model._PowerGeneratedT0 = thermal_gen_attrs['initial_p_output']
    power_generated_t0_validator(model)


    # limits for time periods in which generators are brought on or off-line.
    # must be no less than the generator minimum output.
    def ramp_limit_validator(m):
        for g in model._ThermalGenerators:
            for t in model._TimePeriods:
                for v in [model._StartupRampLimit[g, t], model._ShutdownRampLimit[g, t], model._ScaledStartupRampLimit[g, t],model._ScaledShutdownRampLimit[g, t]]:
                    if v < m._MinimumPowerOutput[g, t]:
                        logger.error("Scaled/Not Scaled Start Up/Shut Down Ramp Limit is less than the minimum power output for power {} at time {}".format(g, t))
                        return False
        return True


    ## These defaults follow what is in most market manuals
    ## We scale this for the time period below
    def startup_ramp_default(m, g, t):
        return m._MinimumPowerOutput[g, t] + m._NominalRampUpLimit[g] / 2.


    ## shutdown is based on the last period *on*
    def shutdown_ramp_default(m, g, t):
        return m._MinimumPowerOutput[g, t] + m._NominalRampDownLimit[g] / 2.

    startup_ramp_default_dict = tupledict({(g, t): startup_ramp_default(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods})
    shutdown_ramp_default_dict = tupledict({(g, t): shutdown_ramp_default(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods})

    startupramplimit = thermal_gen_attrs.get('startup_capacity', 0)
    if startupramplimit == 0:
        startupramplimit = {(g, t): startup_ramp_default(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods}
    else:
        startupramplimit = TimeMapper(startupramplimit)

    model._StartupRampLimit = tupledict(startupramplimit)

    shutdownramplimit =  thermal_gen_attrs.get('shutdown_capacity', 0)
    if shutdownramplimit == 0:
        shutdownramplimit =  {(g, t): shutdown_ramp_default(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods}
    else:
        shutdownramplimit = TimeMapper(shutdownramplimit)

    model._ShutdownRampLimit = tupledict(shutdownramplimit)


    ## These get used in the basic UC constraints, which implicity assume RU, RD <= Pmax
    ## Ramping constraints look backward, so these will accordingly as well
    ## NOTES: can't ramp up higher than the current pmax from the previous value
    ##        can't ramp down more than the pmax from the prior time period
    def scale_ramp_up(m, g, t):
        temp = m._NominalRampUpLimit[g] * m._TimePeriodLengthHours

        if temp > m._MaximumPowerOutput[g, t]:
            return m._MaximumPowerOutput[g, t]
        else:
            return temp


    model._ScaledNominalRampUpLimit = tupledict({(g, t): scale_ramp_up(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods})


    def scale_ramp_down(m, g, t):
        temp = m._NominalRampDownLimit[g] * m._TimePeriodLengthHours

        if t == m._InitialTime:
            param = max(m._PowerGeneratedT0[g],
                        m._MaximumPowerOutput[g, t])
        else:
            param = m._MaximumPowerOutput[g, t - 1]

        if temp > param:
            return param
        else:
            return temp

    model._ScaledNominalRampDownLimit = tupledict({(g, t): scale_ramp_down(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods})


    def scale_startup_limit(m, g, t):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length,
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = m._StartupRampLimit[g, t] - m._MinimumPowerOutput[g, t]
        temp *= m._TimePeriodLengthHours

        if temp > m._MaximumPowerOutput[g, t]- m._MinimumPowerOutput[g, t]:
            return m._MaximumPowerOutput[g, t]
        else:
            return temp + m._MinimumPowerOutput[g, t]

    model._ScaledStartupRampLimit = tupledict({(g, t): scale_startup_limit(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods})

    def scale_shutdown_limit(m, g, t):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = m._ShutdownRampLimit[g, t] - m._MinimumPowerOutput[g, t]
        temp *= m._TimePeriodLengthHours

        if temp > (m._MaximumPowerOutput[g, t]
                                      - m._MinimumPowerOutput[g, t]):
            return m._MaximumPowerOutput[g, t]
        else:
            return temp + m._MinimumPowerOutput[g, t]

    model._ScaledShutdownRampLimit = tupledict({(g, t): scale_shutdown_limit(model, g, t) for g in model._ThermalGenerators for t in model._TimePeriods})


    ramp_limit_validator(model)

    ## Some additional ramping parameters to
    ## deal with shutdowns at time=1

    def _init_p_min_t0(m, g):
        if 'initial_p_min' in thermal_gen_attrs and \
                g in thermal_gen_attrs['initial_p_min']:
            return thermal_gen_attrs['initial_p_min'][g]
        else:
            return m._MinimumPowerOutput[g, m._InitialTime]


    model._MinimumPowerOutputT0 = tupledict({g: _init_p_min_t0(model, g) for g in model._ThermalGenerators})

    def _init_sd_t0(m, g):
        if 'initial_shutdown_capacity' in thermal_gen_attrs and \
                g in thermal_gen_attrs['initial_shutdown_capacity']:
            return thermal_gen_attrs['initial_shutdown_capacity'][g]

        return m._ShutdownRampLimit[g, m._InitialTime]


    model._ShutdownRampLimitT0 = tupledict({g: _init_sd_t0(model, g) for g in model._ThermalGenerators})


    def scale_shutdown_limit_t0(m, g):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = m._ShutdownRampLimitT0[g] - m._MinimumPowerOutputT0[g]
        temp *= m._TimePeriodLengthHours

        if temp > m._PowerGeneratedT0[g]- m._MinimumPowerOutputT0[g]:
            return m._PowerGeneratedT0[g]
        else:
            return temp + m._MinimumPowerOutputT0[g]


    model._ScaledShutdownRampLimitT0 = tupledict({g: scale_shutdown_limit_t0(model, g) for g in model._ThermalGenerators})


    ###############################################
    # startup cost parameters for each generator. #
    ###############################################

    # startup costs are conceptually expressed as pairs (x, y), where x represents the number of hours that a unit has been off and y represents
    # the cost associated with starting up the unit after being off for x hours. these are broken into two distinct ordered sets, as follows.

    def _get_startup_lag(startup, default):
        try:
            iter(startup)
        except TypeError:
            return [default]
        else:
            return [i[0] for i in startup]


    def startup_lags_init_rule(m, g):
        startup_cost = thermal_gens[g].get('startup_cost')
        startup_fuel = thermal_gens[g].get('startup_fuel')

        if startup_cost is not None and startup_fuel is not None:
            logger.warning("WARNING: found startup_fuel for generator }, "
                           "ignoring startup_cost".format(g))

        if startup_fuel is None and startup_cost is None:
            return [model.__MinimumDownTime[g]]

        elif startup_cost is None:
            return _get_startup_lag(startup_fuel,
                                    model.__MinimumDownTime[g])

        else:
            return _get_startup_lag(startup_cost,
                                    m._MinimumDownTime[g])


    # units are hours / time periods.
    model._StartupLags = {g: startup_lags_init_rule(model, g) for g in model._ThermalGenerators}

    def _get_startup_cost(startup, fixed_adder, multiplier):
        try:
            iter(startup)

        except TypeError:
            return [fixed_adder + multiplier * startup]
        else:
            return [fixed_adder + multiplier * i[1] for i in startup]


    def startup_costs_init_rule(m, g):
        startup_cost = thermal_gens[g].get('startup_cost')
        startup_fuel = thermal_gens[g].get('startup_fuel')
        fixed_startup_cost = thermal_gens[g].get('non_fuel_startup_cost')

        if fixed_startup_cost is None:
            fixed_startup_cost = 0.

        if startup_fuel is None and startup_cost is None:
            return [fixed_startup_cost]

        elif startup_cost is None:
            fuel_cost = thermal_gens[g].get('fuel_cost')

            if fuel_cost is None:
                raise ModelError("No fuel cost for generator }, but data is "
                                 "provided for fuel tracking".format(g))

            return _get_startup_cost(startup_fuel,
                                     fixed_startup_cost, fuel_cost)

        else:
            return _get_startup_cost(startup_cost, fixed_startup_cost, 1.)


    # units are $.
    model._StartupCosts = {g: startup_costs_init_rule(model, g) for g in model._ThermalGenerators}


    # startup lags must be monotonically increasing...
    def validate_startup_lags_rule(m, g):
        startup_lags = list(m._StartupLags[g])

        if len(startup_lags) == 0:
            raise ModelError("DATA ERROR: The number of startup lags for "
                             "thermal generator `}` must be >= 1!".format(g))

        if startup_lags[0] != m._MinimumDownTime[g]:
            raise ModelError(
                "DATA ERROR: The first startup lag for thermal generator `}` "
                "must be equal the minimum down time }!".format(
                    g, m._MinimumDownTime[g])
            )

        for i in range(0, len(startup_lags) - 1):
            if startup_lags[i] >= startup_lags[i + 1]:
                raise ModelError(
                    "DATA ERROR: Startup lags for thermal generator `}` must "
                    "be monotonically increasing!".format(g)
                )

    [validate_startup_lags_rule(model, g) for g in model._ThermalGenerators]

    # while startup costs must be monotonically non-decreasing!
    def validate_startup_costs_rule(m, g):
        startup_costs = m._StartupCosts[g]

        for i in range(1, len(startup_costs) - 1):
            if startup_costs[i] > startup_costs[i + 1]:
                raise ModelError(
                    "DATA ERROR: Startup costs for thermal generator `}` must "
                    "be monotonically non-decreasing!".format(g)
                )

    [validate_startup_costs_rule(model, g) for g in  model._ThermalGenerators]

    def validate_startup_lag_cost_cardinalities(m, g):
        if len(m._StartupLags[g]) != len(m._StartupCosts[g]):
            raise ModelError(
                "DATA ERROR: The number of startup lag entries (}) for "
                "thermal generator `}` must equal the number of startup cost "
                "entries (})".format(
                    len(m._StartupLags[g]), g, len(m._StartupCosts[g]))
            )

    [validate_startup_lag_cost_cardinalities(model, g) for g in model._ThermalGenerators]


    # for purposes of defining constraints, it is useful to have a set to index the various startup costs parameters.
    # entries are 1-based indices, because they are used as indicies into Pyomo sets - which use 1-based indexing.
    #change to 0-based indices when rewriting it directly in Gurobi

    def startup_cost_indices_init_rule(m, g):
        return range(0, len(m._StartupLags[g]))

    model._StartupCostIndices = {g: startup_cost_indices_init_rule(model, g) for g in model._ThermalGenerators}

    ## scale the startup lags
    ## Again, assert that this must be at least one in the time units of the model
    def scaled_startup_lags_rule(m, g):
        return [max(int(round(this_lag / m._TimePeriodLengthHours)), 1)
                for this_lag in m._StartupLags[g]]

    model._ScaledStartupLags = {g: scaled_startup_lags_rule(model, g) for g in model._ThermalGenerators}

    ##################################################################################
    # shutdown cost for each generator. in the literature, these are often set to 0. #
    ##################################################################################

    # units are $.
    model._ShutdownFixedCost = tupledict(thermal_gen_attrs.get('shutdown_cost', {g: 0 for g in model._ThermalGenerators}))


    ## FUEL-SUPPLY Sets
    def fuel_supply_gens_init(m):
        if 'fuel_supply' not in thermal_gen_attrs:
            thermal_gen_attrs['fuel_supply'] = tupledict()

        if 'aux_fuel_supply' not in thermal_gen_attrs:
            thermal_gen_attrs['aux_fuel_supply'] = tupledict()

        fuel_supply = thermal_gen_attrs['fuel_supply']
        for g in fuel_supply:
            yield g

        for g in thermal_gen_attrs['aux_fuel_supply']:
            if g not in fuel_supply:
                yield g


    def gen_cost_fuel_validator(g):
        # validators may get called once
        # with key None for empty sets
        if len(g) == 0:
            return True

        if 'p_fuel' in thermal_gen_attrs and g in thermal_gen_attrs['p_fuel']:
            pass

        else:
            raise ModelError("All fuel-constrained generators must have the "
                             "<p_fuel> attribute which tracks their fuel "
                             "consumption, could not find such an attribute "
                             "for generator `{}`!'".format(g))

        return True


    model._FuelSupplyGenerators = OrderedSet(fuel_supply_gens_init(model))
    gen_cost_fuel_validator(model._FuelSupplyGenerators)


    ## DUAL-FUEL Sets

    def dual_fuel_init(m):
        for gen, g_data in thermal_gens.items():
            if 'aux_fuel_capable' in g_data and g_data['aux_fuel_capable']:
                yield gen

    model._DualFuelGenerators =   OrderedSet(dual_fuel_init(model))

    ## This set is for modeling elements that are exhanged
    ## in whole for the dual-fuel model
    model._SingleFuelGenerators = (model._ThermalGenerators
                                  - model._DualFuelGenerators)


    ## BEGIN PRODUCTION COST
    ## NOTE: For better or worse, we handle scaling this to the time period length in the objective function.
    ##       In particular, this is done in objective.py.

    def _check_curve(m, g, curve, curve_type):
        for i, t in enumerate(m._TimePeriods):
            ## first, get a cost_curve out of time series
            if curve['data_type'] == 'time_series':
                curve_t = curve['values'][i]
                _t = t

            else:
                curve_t = curve
                _t = None

            tx_utils.validate_and_clean_cost_curve(
                curve_t, curve_type=curve_type,
                p_min=m._MinimumPowerOutput[g, t],
                p_max=m._MaximumPowerOutput[g, t],
                gen_name=g, t=_t
            )

            # if no curve_type+'_type' is specified, we assume piecewise
            # (for backwards compatibility with no 'fuel_curve_type')
            if (curve_type + '_type' in curve
                    and curve_t[curve_type + '_type'] == 'polynomial'):
                if not _check_curve.warn_piecewise_approx:
                    logger.warning("WARNING: Polynomial cost curves will be "
                                   "approximated using piecewise segments")
                    _check_curve.warn_piecewise_approx = True

            if curve['data_type'] != 'time_series':
                break


    ## set "static" variable for this function
    _check_curve.warn_piecewise_approx = False


    def validate_cost_rule(m, g):
        gen_dict = thermal_gens[g]
        cost = gen_dict.get('p_cost')
        fuel = gen_dict.get('p_fuel')
        fuel_cost = gen_dict.get('fuel_cost')

        if cost is None and fuel is None:
            logger.warning("WARNING: Generator {} has no cost information "
                           "associated with it".format(g))
            return True

        if cost is not None and fuel is not None:
            logger.warning("WARNING: ignoring provided p_cost and using fuel "
                           "cost data from p_fuel for generator {}".format(g))

        ## look at p_cost through time
        if fuel is None:
            _check_curve(m, g, cost, 'cost_curve')

        else:
            if fuel_cost is None:
                raise ModelError("Found fuel_curve but not fuel_cost "
                                 "for generator {}".format(g))

            _check_curve(m, g, fuel, 'fuel_curve')
            for i, t in enumerate(m._TimePeriods):
                if fuel_cost is dict:
                    if fuel_cost['data_type'] != 'time_series':
                        raise ModelError("fuel_cost must be either numeric "
                                         "or time_series")

                    fuel_cost_t = fuel_cost['values'][i]

                else:
                    fuel_cost_t = fuel_cost

                if fuel_cost_t < 0:
                    raise ModelError(
                        "fuel_cost must be non-negative, found negative "
                        "fuel_cost for generator `{}`!".format(g)
                    )

                if fuel_cost_t == fuel_cost:
                    break

        return True


    model._ValidateGeneratorCost = tupledict({g: validate_cost_rule(model, g) for g in model._ThermalGenerators})

    ##############################################################################################
    # number of pieces in the linearization of each generator's quadratic cost production curve. #
    ##############################################################################################
    ## TODO: option-drive with Egret, either globally or per-generator

    model._NumGeneratorCostCurvePieces = 2

    #######################################################################
    # points for piecewise linearization of power generation cost curves. #
    #######################################################################

    # BK -- changed to reflect that the generator's power output variable is always above minimum in the ME model
    #       this simplifies things quite a bit..

    # maps a (generator, time-index) pair to a list of points defining the piecewise cost linearization breakpoints.
    # the time index is redundant, but required - in the current implementation of the Piecewise construct, the
    # breakpoints must be indexed the same as the Piecewise construct itself.

    # the points are expected to be on the interval [0, maxpower-minpower], and must contain both endpoints.
    # power generated can always be 0, and piecewise expects the entire variable domain to be represented.
    model._PowerGenerationPiecewisePoints = tupledict()

    # NOTE: the values are relative to the minimum production cost, i.e., the values represent
    # incremental costs relative to the minimum production cost.
    model._PowerGenerationPiecewiseCostValues = tupledict()

    # NOTE; these values are relative to the minimum fuel conumption
    model._PowerGenerationPiecewiseFuelValues = tupledict()

    _minimum_production_cost = tupledict()
    _minimum_fuel_consumption = tupledict()


    def _eliminate_piecewise_duplicates(input_func):
        if len(input_func) <= 1:
            return input_func

        new = [input_func[0]]
        for (o1, c1), (o2, c2) in zip(input_func, input_func[1:]):
            if not math.isclose(o1, o2) and not math.isclose(c1, c2):
                new.append((o2, c2))

        return new


    def _much_less_than(v1, v2):
        return v1 < v2 and not math.isclose(v1, v2)


    def _piecewise_adjustment_helper(m, p_min, p_max, input_func):
        minimum_val = 0.
        new_points = []
        new_vals = []

        input_func = _eliminate_piecewise_duplicates(input_func)
        set_p_min = False

        # NOTE: this implicitly inserts a (0.,0.)
        #       into every cost array
        prior_output, prior_cost = 0., 0.

        for output, cost in input_func:
            ## catch this case
            if math.isclose(output, p_min) and math.isclose(output, p_max):
                new_points.append(0.)
                new_vals.append(0.)
                minimum_val = cost
                break

            ## output < p_min
            elif _much_less_than(output, p_min):
                pass

            ## p_min == output
            elif math.isclose(output, p_min):
                assert set_p_min is False
                new_points.append(0.)
                new_vals.append(0.)
                minimum_val = cost
                set_p_min = True

            ## p_min < output
            elif (_much_less_than(p_min, output)
                  and _much_less_than(output, p_max)):
                if not set_p_min:
                    new_points.append(0.)
                    new_vals.append(0.)

                    price = (cost - prior_cost) / (output - prior_output)
                    minimum_val = (p_min - prior_output) * price + prior_cost

                    new_points.append(output - p_min)
                    new_vals.append((output - p_min) * price)
                    set_p_min = True

                else:
                    new_points.append(output - p_min)
                    new_vals.append(cost - minimum_val)

            elif math.isclose(output, p_max) or _much_less_than(p_max, output):
                if not set_p_min:
                    new_points.append(0.)
                    new_vals.append(0.)

                    price = (cost - prior_cost) / (output - prior_output)
                    minimum_val = (p_min - prior_output) * price + prior_cost
                    new_points.append(p_max - p_min)

                    if math.isclose(output, p_max):
                        new_vals.append(cost - minimum_val)
                    else:
                        new_vals.append((p_max - p_min) * price)

                    set_p_min = True

                else:
                    new_points.append(p_max - p_min)

                    if math.isclose(output, p_max):
                        new_vals.append(cost - minimum_val)
                    else:
                        price = (cost - prior_cost) / (output - prior_output)
                        new_vals.append((p_max - prior_output)
                                        * price + prior_cost - minimum_val)

                break

            else:
                raise ModelError(
                    "Unexpected case in _piecewise_adjustment_helper, "
                    "p_min={}, p_max={}, output={}".format(
                        p_min, p_max, output)
                )

            prior_output, prior_cost = output, cost

        return new_points, new_vals, minimum_val


    def _polynomial_to_piecewise_helper(m, p_min, p_max, input_func):
        segment_max = m._NumGeneratorCostCurvePieces

        for key in {0, 1, 2}:
            if key not in input_func:
                input_func[key] = 0.

        poly_func = lambda x: (input_func[0] + input_func[1] * x
                               + input_func[2] * x * x)

        if p_min >= p_max:
            minimum_val = poly_func(p_min)
            new_points = [0.]
            new_vals = [0.]

            return new_points, new_vals, minimum_val

        elif input_func[2] == 0.:  ## not actually quadratic
            minimum_val = poly_func(p_min)
            new_points = [0., p_max - p_min]
            new_vals = [0., poly_func(p_max) - minimum_val]

            return new_points, new_vals, minimum_val

        ## actually quadratic
        width = (p_max - p_min) / float(segment_max)

        new_points = [i * width for i in range(0, segment_max + 1)]

        ## replace the last with (p_max - p_min)
        new_points[-1] = p_max - p_min

        minimum_val = poly_func(p_min)
        new_vals = [poly_func(pnt + p_min) - minimum_val for pnt in new_points]

        return new_points, new_vals, minimum_val


    def _piecewise_helper(m, p_min, p_max, curve, curve_type):
        if (curve_type not in curve
                or curve[curve_type] == 'piecewise'):
            return _piecewise_adjustment_helper(m, p_min, p_max,
                                                curve['values'])

        else:
            assert curve[curve_type] == 'polynomial'
            return _polynomial_to_piecewise_helper(m, p_min, p_max,
                                                   curve['values'])


    def power_generation_piecewise_points_rule(m, g):
        ## NOTE: it is often (usually) the case that cost curves
        ##       are the same in every time period, This function
        ##       is optimized to avoid data redunancy and recalculation
        ##       for that case

        gen_dict = gens[g]
        fuel_curve = gen_dict.get('p_fuel')
        cost_curve = gen_dict.get('p_cost')
        fuel_cost = gen_dict.get('fuel_cost', 0.)
        no_load_cost = gen_dict.get('non_fuel_no_load_cost', 0.)

        if isinstance(fuel_cost, tupledict):
            fuel_costs = fuel_cost['values']
        else:
            fuel_costs = (fuel_cost for t in m._TimePeriods)

        if isinstance(no_load_cost, tupledict):
            no_load_costs = no_load_cost['values']
        else:
            no_load_costs = (no_load_cost for _ in m._TimePeriods)

        _curve_cache = tupledict()
        if fuel_curve is not None:
            g_in_fuel_supply_generators = g in m._FuelSupplyGenerators
            g_in_single_fuel_generators = g in m._SingleFuelGenerators

            if (isinstance(fuel_curve, tupledict)
                    and fuel_curve['data_type'] == 'time_series'):
                fuel_curves = fuel_curve['values']
                one_fuel_curve = False

            else:
                fuel_curves = (fuel_curve for _ in m._TimePeriods)
                one_fuel_curve = True

            for fuel_curve, fuel_cost, nlc, t in zip(fuel_curves, fuel_costs,
                                                     no_load_costs,
                                                     m._TimePeriods):
                p_min = m._MinimumPowerOutput[g, t]
                p_max = m._MaximumPowerOutput[g, t]

                if (p_min, p_max, fuel_cost, nlc) in _curve_cache:
                    curve = _curve_cache[p_min, p_max, fuel_cost, nlc]

                    if one_fuel_curve or curve['fuel_curve'] == fuel_curve:
                        m._PowerGenerationPiecewisePoints[g, t] = curve['points']

                        if g_in_fuel_supply_generators:
                            _minimum_fuel_consumption[g, t] = curve[
                                'min_fuel_consumption']
                            m._PowerGenerationPiecewiseFuelValues[g, t] = curve[
                                'fuel_values']

                        if g_in_single_fuel_generators:
                            _minimum_production_cost[g, t] = curve[
                                'min_production_cost']
                            m._PowerGenerationPiecewiseCostValues[g, t] = curve[
                                'cost_values']

                        continue

                points, values, minimum_val = _piecewise_helper(
                    m, p_min, p_max, fuel_curve, 'fuel_curve_type')
                curve = {'points': points}

                if not one_fuel_curve:
                    curve['fuel_curve'] = fuel_curve

                m._PowerGenerationPiecewisePoints[g, t] = points
                if g_in_fuel_supply_generators:
                    _minimum_fuel_consumption[g, t] = minimum_val
                    curve['min_fuel_consumption'] = minimum_val

                    m._PowerGenerationPiecewiseFuelValues[g, t] = values
                    curve['fuel_values'] = values

                if g_in_single_fuel_generators:
                    min_production_cost = (minimum_val
                                           * fuel_cost + no_load_cost)
                    _minimum_production_cost[g, t] = min_production_cost
                    curve['min_production_cost'] = min_production_cost

                    cost_values = [fuel_cost * val for val in values]
                    m._PowerGenerationPiecewiseCostValues[g, t] = cost_values
                    curve['cost_values'] = cost_values

                _curve_cache[p_min, p_max, fuel_cost, nlc] = curve

            return  ## we can assume below that we don't have a fuel curve

        if (isinstance(cost_curve, tupledict)
                and cost_curve['data_type'] == 'time_series'):
            cost_curves = cost_curve['values']
            one_cost_curve = False

        else:
            cost_curves = (cost_curve for _ in m._TimePeriods)
            one_cost_curve = True

        for cost_curve, nlc, t in zip(cost_curves, no_load_costs,
                                      m._TimePeriods):
            p_min = m._MinimumPowerOutput[g, t]
            p_max = m._MaximumPowerOutput[g, t]

            if (p_min, p_max, nlc) in _curve_cache:
                curve = _curve_cache[p_min, p_max, nlc]

                if one_cost_curve or curve['cost_curve'] == cost_curve:
                    m._PowerGenerationPiecewisePoints[g, t] = curve['points']
                    m._PowerGenerationPiecewiseCostValues[g, t] = curve[
                        'cost_values']
                    _minimum_production_cost[g, t] = curve['min_production']

                    continue

            if cost_curve is None:
                if p_min >= p_max:  ## only one point
                    points = [0.]
                    values = [0.]

                else:
                    points = [0., p_max - p_min]
                    values = [0., 0.]

                min_production = nlc

            else:
                points, values, minimum_val = _piecewise_helper(
                    m, p_min, p_max, cost_curve, 'cost_curve_type')
                min_production = minimum_val + nlc

            curve = {'points': points, 'cost_values': values,
                     'min_production': min_production}

            if not one_cost_curve:
                curve['cost_curve'] = cost_curve

            _curve_cache[p_min, p_max, nlc] = curve

            m._PowerGenerationPiecewisePoints[g, t] = points
            m._PowerGenerationPiecewiseCostValues[g, t] = values
            _minimum_production_cost[g, t] = min_production


    model._CreatePowerGenerationPiecewisePoints = [power_generation_piecewise_points_rule(model, g) for g in cost_gens]

    # Minimum production cost (needed because Piecewise constraint on
    # ProductionCost has to have lower bound of 0, so the unit can cost 0 when
    # off -- this is added back in to the objective if a unit is on

    if renew_costs:
        model._MinimumProductionCost = tupledict(((g, t), _minimum_production_cost[g, t]) for g in model._SingleFuelGenerators | model._AllNondispatchableGenerators
                                               for t in model._TimePeriods if (g, t) in _minimum_production_cost.keys())

        model._MinimumFuelConsumption = tupledict(((g, t), _minimum_fuel_consumption[g, t]) for g in model._FuelSupplyGenerators | model._AllNondispatchableGenerators
                                               for t in model._TimePeriods if (g, t) in _minimum_fuel_consumption.keys())

    else:
        model._MinimumProductionCost = tupledict(((g, t), _minimum_production_cost[g, t]) for g in model._SingleFuelGenerators
                                               for t in model._TimePeriods if (g, t) in _minimum_production_cost.keys())

        model._MinimumFuelConsumption = tupledict(((g, t), _minimum_fuel_consumption[g, t]) for g in model._FuelSupplyGenerators
                                               for t in model._TimePeriods if (g, t) in _minimum_fuel_consumption.keys())


    ## END PRODUCTION COST CALCULATIONS

    #########################################
    # penalty costs for constraint violation #
    #########################################

    kinda_big_penalty = 1e3 * model_data.get_system_attr('baseMVA')
    big_penalty = 1e4 * model_data.get_system_attr('baseMVA')

    model._ReserveShortfallPenalty = model_data.get_system_attr('reserve_shortfall_cost',
                                              kinda_big_penalty)


    model._LoadMismatchPenalty = model_data.get_system_attr('load_mismtch_cost',
                                              big_penalty)

    model._LoadMismatchPenaltyReactive = model_data.get_system_attr('q_load_mismatch_cost',
                                              big_penalty / 2.)


    model._Contingencies = OrderedSet(contingencies.keys())

    # leaving this unindexed for now for simpility
    model._ContingencyLimitPenalty = model_data.get_system_attr(
            'contingency_flow_violation_cost', big_penalty / 2.)

    #
    # STORAGE parameters
    #

    model._Storage = OrderedSet(storage_attrs['names'])
    model._StorageAtBus = tupledict(storage_by_bus)

    def verify_storage_buses_rule(m):
        assert set(m._Storage) == {store for bus in m._Buses
                                  for store in m._StorageAtBus[bus]}

    model._VerifyStorageBuses = verify_storage_buses_rule(model)

    ####################################################################################
    # minimum and maximum power ratings, for each storage unit. units are MW.          #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################

    # Storage power output >0 when discharging

    model._MinimumPowerOutputStorage = tupledict(storage_attrs.get('min_discharge_rate', {storage: 0 for storage in model._Storage}))

    def maximum_power_output_validator_storage(m, v, s):
        return v >= m._MinimumPowerOutputStorage[s]

    model._MaximumPowerOutputStorage = tupledict(storage_attrs.get('max_discharge_rate', {storage: 0 for storage in model._Storage}))
    if model._MaximumPowerOutputStorage:
        maximumpoweroutput_validate = [maximum_power_output_validator_storage(model, model._MaximumPowerOutputStorage[s], s) for s in model._Storage]
        if not any(maximumpoweroutput_validate):
            logger.warning('Maximum power output should be less than Minimum power output')

    #Storage power input >0 when charging
    model._MinimumPowerInputStorage = tupledict(storage_attrs.get('min_charge_rate', {storage: 0 for storage in model._Storage}))

    def maximum_power_input_validator_storage(m, v, s):
        return v >= m._MinimumPowerInputStorage[s]
    model._MaximumPowerInputStorage = tupledict(storage_attrs.get('max_charge_rate', {storage: 0 for storage in model._Storage}))

    def maximum_power_input_validator_storage(m, v, s):
        return v >= m._MinimumPowerInputStorage[s]
    if model._MaximumPowerInputStorage:
        maximumpowerinput_validate = [maximum_power_input_validator_storage(model, model._MaximumPowerInputStorage[s], s) for s in model._Storage]
        if not any(maximumpowerinput_validate):
            logger.warning('Maximum power input should be less than Minimum power output')
    ###############################################
    # storage ramp up/down rates. units are MW/h. #
    ###############################################

    # ramp rate limits when discharging
    model._NominalRampUpLimitStorageOutput = tupledict(storage_attrs.get('ramp_up_output_60min', {storage: None for storage in model._Storage}))

    model._NominalRampDownLimitStorageOutput = tupledict(
        storage_attrs.get('ramp_down_output_60min', {storage: None for storage in model._Storage}))


    # ramp rate limits when charging
    model._NominalRampUpLimitStorageInput  = tupledict(
        storage_attrs.get('ramp_up_input_60min', {storage: None for storage in model._Storage}))

    model._NominalRampDownLimitStorageInput = tupledict(
        storage_attrs.get('ramp_down_input_60min', {storage: None for storage in model._Storage}))


    def scale_storage_ramp_up_out(m, s):
        return m._NominalRampUpLimitStorageOutput[s] * m._TimePeriodLengthHours

    model._ScaledNominalRampUpLimitStorageOutput = tupledict({s: scale_storage_ramp_up_out(model, s) for s in model._Storage})

    def scale_storage_ramp_down_out(m, s):
        return m._NominalRampDownLimitStorageOutput[s] * m._TimePeriodLengthHours

    model._ScaledNominalRampUpLimitStorageOutput = tupledict({s: scale_storage_ramp_down_out(model, s) for s in model._Storage})


    def scale_storage_ramp_up_in(m, s):
        return m._NominalRampUpLimitStorageInput[s] * m._TimePeriodLengthHours

    model._ScaledNominalRampUpLimitStorageInput  = tupledict(
        {s: scale_storage_ramp_up_in(model, s) for s in model._Storage})


    def scale_storage_ramp_down_in(m, s):
        return m._NominalRampDownLimitStorageInput[s] * m._TimePeriodLengthHours

    model._ScaledNominalRampDownLimitStorageInput = tupledict(
        {s: scale_storage_ramp_down_in(model, s) for s in model._Storage})

    ####################################################################################
    # minimum state of charge (SOC) and maximum energy ratings, for each storage unit. #
    # units are MWh for energy rating and p.u. (i.e. [0,1]) for SOC     #
    ####################################################################################

    # you enter storage energy ratings once for each storage unit

    model._MaximumEnergyStorage = tupledict(storage_attrs.get('energy_capacity', {s: 0 for s in model._Storage}))

    model._MinimumSocStorage = tupledict(storage_attrs.get('minimum_state_of_charge', {s: 0 for s in model._Storage}))

    ################################################################################
    # round trip efficiency for each storage unit given as a fraction (i.e. [0,1]) #
    ################################################################################
    model._InputEfficiencyEnergy = tupledict(storage_attrs.get('charge_efficiency', {s: 1 for s in model._Storage}))
    model._OutputEfficiencyEnergy = tupledict(storage_attrs.get('discharge_efficiency', {s: 1 for s in model._Storage}))



    ## assumed to be %/hr
    model._RetentionRate = tupledict(storage_attrs.get('retention_rate_60min', {s: 1 for s in model._Storage}))

    model._ChargeCost = tupledict(storage_attrs.get('charge_cost', {s: 0 for s in model._Storage}))

    model._DisChargeCost = tupledict(storage_attrs.get('discharge_cost', {s: 0 for s in model._Storage}))


    # this will be multiplied by itself 1/m.TimePeriodLengthHours times, so
    # this is the scaling to get us back to %/hr
    def scaled_retention_rate(m,s):
        return m._RetentionRate[s] ** m._TimePeriodLengthHours

    model._DisChargeCost = tupledict(storage_attrs.get('discharge_cost', {s: None for s in model._Storage}))

    model._ScaledRetentionRate = tupledict({s: scaled_retention_rate(model, s) for s in model._Storage})

    ########################################################################
    # end-point SOC for each storage unit. units are in p.u. (i.e. [0,1])  #
    ########################################################################

    # end-point values are the SOC targets at the final time period. With no
    # end-point constraints storage units will always be empty at the final
    # time period.
    def _end_point_soc(m, s):
        if s is None:
            return

        s_dict = storage[s]
        if 'end_state_of_charge' in s_dict:
            return s_dict['end_state_of_charge']
        if 'initial_state_of_charge' in s_dict:
            return s_dict['initial_state_of_charge']

        return 0.5

    model._EndPointSocStorage = tupledict({s: _end_point_soc(model, s) for s in model._Storage})

    ############################################################
    # storage initial conditions: SOC, power output and input  #
    ############################################################

    def t0_storage_power_input_validator(m, v, s):
        return (m._MinimumPowerInputStorage[s] <= v) and (v<= m._MaximumPowerInputStorage[s])

    def t0_storage_power_output_validator(m, v, s):
        return (m._MinimumPowerOutputStorage[s] <= v) and (v <= m._MaximumPowerOutputStorage[s])

    model._StoragePowerOutputOnT0 = tupledict(storage_attrs.get('initial_discharge_rate', {s: 0 for s in model._Storage}))
    model._StoragePowerInputOnT0 = tupledict(storage_attrs.get('initial_charge_rate', {s: 0 for s in model._Storage}))

    if model._StoragePowerInputOnT0:
        if not any([t0_storage_power_input_validator(model, model._StoragePowerInputOnT0[s], s) for s in model._Storage]):
            logger.warning('Storage Power Input should between its min and max')

    if model._StoragePowerOutputOnT0:
        if not any([t0_storage_power_output_validator(model, model._StoragePowerOutputOnT0[s], s) for s in model._Storage]):
            logger.warning('Storage Power Output should between its min and max')

    model._StorageSocOnT0 = tupledict(storage_attrs.get('initial_state_of_charge', {s: 0.5 for s in model._Storage}))
    return model


# @add_model_attr(component_name)
def default_params(
        model: gp.Model,
        model_data: Optional[VaticModelData] = None
        ) -> gp.Model:
    """This loads unit commitment params from a GridModel object."""
    return load_base_params(model, model_data, renew_costs=False)


# @add_model_attr(component_name)
def renew_cost_params(
        model: gp.Model,
        model_data: Optional[VaticModelData] = None
        ) -> gp.Model:
    """This loads unit commitment params from a GridModel object."""
    return load_base_params(model, model_data, renew_costs=True)
